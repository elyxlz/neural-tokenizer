from dataclasses import dataclass
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from accelerate import Accelerator
from transformers import get_inverse_sqrt_schedule
import wandb

from neural_tokenizer.model import NeuralTokenizer


@dataclass
class TrainConfig:
    # main
    name: str = "neural-tokenizer_test"
    mixed_precision: str = "no"  # "no", "fp16", "bf16"
    gradient_checkpointing: bool = False
    cpu: bool = False

    # dataset
    dataset_namespace: str = "QuyenAnhDE/Diseases_Symptoms"
    dataset_column: str = "Name"
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True

    # optimizer
    lr: float = 1e-4
    weight_decay: float = 0.1
    lr_eps: float = 1e-8
    lr_betas: tuple = (0.9, 0.99)
    fused_adam: bool = False
    grad_norm: float = 1.0

    # scheduler
    warmup_steps: int = 1000
    timescale: int = 1e8
    last_epoch: int = -1

    # logging and misc
    log_every: int = 100
    save_every: int = 1000
    push_every: int = None
    val_every: int = None
    resume_from_ckpt: str = None
    use_wandb: bool = False
    wandb_project_name: str = "neural-tokenizer"

    def to_dict(self):
        return vars(self)


class Trainer:
    def __init__(
        self,
        model: NeuralTokenizer,
        train_config: TrainConfig,
    ):
        self.train_config = train_config

        self.completed_steps = -1
        self.run_id = wandb.util.generate_id()

        # ckpt
        if train_config.resume_from_ckpt is not None:
            self.run_id = os.path.basename(
                os.path.dirname(train_config.resume_from_ckpt)
            )

        # log dir
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.log_dir = os.path.join(os.path.join(root_dir, "logs"), self.run_id)

        self.accelerator = Accelerator(
            mixed_precision=train_config.mixed_precision,
            cpu=train_config.cpu,
        )

        # wandb
        if train_config.use_wandb and self.accelerator.is_local_main_process:
            config = train_config.to_dict() | model.config.to_dict()
            config["seed"] = os.getenv("GLOBAL_SEED")
            assert (
                train_config.wandb_project_name is not None
            ), "Please provide a wandb project name"
            self.accelerator.init_trackers(
                project_name=train_config.wandb_project_name,
                config=config,
                init_kwargs=dict(
                    wandb=dict(
                        name=train_config.name,
                        dir=os.path.join(root_dir, "logs"),
                        id=self.run_id,
                        resume="allow",
                    )
                ),
            )

        # model
        self.model: NeuralTokenizer = self.accelerator.prepare(model)
        self.model.train()

        if train_config.gradient_checkpointing:
            raise NotImplementedError
            self.model.gradient_checkpointing_enable()

        # optimizer
        nodecay_params = [p for p in self.model.parameters() if p.dim() == 1]
        decay_params = [p for p in self.model.parameters() if p.dim() != 1]

        self.optimizer = torch.optim.AdamW(
            [
                {"params": nodecay_params, "weight_decay": 0.0},
                {"params": decay_params, "weight_decay": train_config.weight_decay},
            ],
            lr=train_config.lr,
            betas=train_config.lr_betas,
            eps=train_config.lr_eps,
            fused=train_config.fused_adam,
        )
        self.optimizer = self.accelerator.prepare(self.optimizer)

        # scheduler
        self.scheduler = get_inverse_sqrt_schedule(
            optimizer=self.optimizer,
            num_warmup_steps=train_config.warmup_steps,
            timescale=train_config.timescale,
            last_epoch=train_config.last_epoch,
        )
        self.accelerator.register_for_checkpointing(self.scheduler)

        # get dataset and loaders
        ds = load_dataset(train_config.dataset_namespace)["train"]
        ds = ds.remove_columns(
            [i for i in ds.column_names if i != train_config.dataset_column]
        )
        ds = ds.rename_column(train_config.dataset_column, "text")
        ds = ds.train_test_split(test_size=0.1)

        self.train_loader = DataLoader(
            ds["train"],
            batch_size=train_config.batch_size,
            num_workers=train_config.num_workers,
            pin_memory=train_config.pin_memory,
            shuffle=True,
        )

        self.val_loader = DataLoader(
            ds["test"],
            batch_size=train_config.batch_size,
            num_workers=train_config.num_workers,
            pin_memory=train_config.pin_memory,
            shuffle=False,
        )

        if train_config.resume_from_ckpt is not None:
            self.resume()

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.accelerator.print("Trainable parameters: ", trainable_params / 1e6, "M")

    def training_step(self, batch):
        loss = self.model.forward(**batch)
        self.accelerator.backward(loss)
        self.accelerator.clip_grad_norm_(
            self.model.parameters(), self.train_config.grad_norm
        )  # Gradient clipping
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        if self.time_to_log():
            self.accelerator.log({"train_loss": loss}, step=self.completed_steps)
            self.epoch_bar.set_postfix({"loss": loss.item()})

    @torch.no_grad()
    def validation(self, val_loader):
        """Validation loop"""
        self.model.eval()
        val_loss = 0.0
        for batch in tqdm(val_loader, desc="Validation"):
            loss = self.model.forward(**batch)
            val_loss += loss.item()
        val_loss /= len(val_loader)
        self.accelerator.log({"val_loss": loss})
        self.model.train()

    def train(self):
        """Basic training loop"""

        epochs = -1
        while True:
            epochs += 1

            self.accelerator.log({"epoch": epochs}, step=self.completed_steps)

            self.epoch_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epochs}",
                disable=not self.accelerator.is_local_main_process,
                initial=self.completed_steps + 1 % len(self.train_loader),
            )

            for batch in self.epoch_bar:
                self.completed_steps += 1

                with self.accelerator.accumulate(self.model):
                    self.training_step(batch)

                # validation and evaluation
                if self.time_to_val():
                    self.accelerator.print(f"Validation at step {self.completed_steps}")
                    self.validation(self.val_loader)

                # checkpoint
                if self.time_to_save():
                    self.save()

                # push to hub
                if self.time_to_push():
                    self.push(self.model)

    def time_to_save(self) -> bool:
        """Checks if it's time to save"""
        save: bool = (
            self.train_config.save_every is not None
            and self.completed_steps % self.train_config.save_every == 0
            and self.completed_steps != 0
        )
        return save

    def time_to_push(self) -> bool:
        """Checks if it's time to push"""
        push: bool = (
            self.train_config.push_every is not None
            and self.completed_steps % self.train_config.push_every == 0
            and self.completed_steps != 0
        )
        return push

    def time_to_log(self) -> bool:
        """Checks if it's time to log"""
        log: bool = (
            self.train_config.log_every is not None
            and self.completed_steps % self.train_config.log_every == 0
        )
        return log

    def time_to_val(self) -> bool:
        """Checks if it's time to validate"""
        val: bool = (
            self.train_config.val_every is not None
            and self.completed_steps % self.train_config.val_every == 0
        )
        return val

    def save(self) -> None:
        """Saves model to path"""

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if self.accelerator.is_local_main_process:
            self.accelerator.print(f"Saving model to {self.log_dir}")
            save_dir = os.path.join(self.log_dir, f"step_{self.completed_steps}")
            # check if there are < 5 checkpoints
            if len(os.listdir(self.log_dir)) > 5:
                # remove oldest checkpoint
                oldest_step = sorted(
                    [
                        int(os.path.splitext(f)[0].replace("step_", ""))
                        for f in os.listdir(self.log_dir)
                    ]
                )[0]
                oldest_step_dir = os.path.join(self.log_dir, f"step_{oldest_step}")
                self.accelerator.print(f"Removing oldest checkpoint {oldest_step_dir}")
                os.system(f"rm -rf {oldest_step_dir}")
            self.accelerator.save_state(save_dir)

    def resume(self) -> None:
        """Resumes from checkpoint state, sets self.resume_step and adds to self.completed_steps"""
        assert (
            self.train_config.resume_from_ckpt is not None
        ), "Please provide a checkpoint path to resume from"
        self.accelerator.print(
            f"Resuming from checkpoint {self.train_config.resume_from_ckpt}"
        )
        self.accelerator.load_state(self.train_config.resume_from_ckpt)
        path = os.path.basename(self.train_config.resume_from_ckpt)
        training_basename = os.path.splitext(path)[0]
        resume_step = int(training_basename.replace("step_", ""))
        self.completed_steps += resume_step

    def push(self, model: NeuralTokenizer) -> None:
        """Takes care of pushing the model to hub, multi-rank safe"""
        try:
            if self.completed_steps == 0:
                return

            if self.accelerator.is_main_process:
                self.accelerator.print("Pushing model to hub...")
                unwrapped_model: NeuralTokenizer = self.accelerator.unwrap_model(model)

                unwrapped_model.push_to_hub(
                    self.train_config.hub_namespace,
                    commit_message=f"Run {self.run_id}, step {self.completed_steps}",
                    private=True,
                    token=os.environ["HUGGINGFACE_TOKEN"],
                )
            self.accelerator.wait_for_everyone()
        except Exception:
            print("Push failed")
            self.accelerator.wait_for_everyone()
