# Loading

# neural-tokenizer
High compression text tokenizers via VQAEs for efficient and democratic language modeling.

Language models struggle with semantic modeling due to high frequency details in tokens from typical tokenizers, employing stronger textual compression via neural tokenizers may solve alleviate this problem.

# Loading
```py
from neural_tokenizer import NeuralTokenizer, NeuralTokenizerConfig

# Random initialization
model = NeuralTokenizer(NeuralTokenizerConfig())

# Push to huggingface hub
model = model.push_to_hub("elyxlz/neural-tokenizer-v1")

# Load pretrained model
model = NeuralTokenizer.from_pretrained("elyxlz/neural-tokenizer-v1")
```


# Usage
```python
text = ["Hello", "World :)"]
tokens = model.encode(text)
print(tokens.data)
# [[0, 1235, 1236, 1], [0, 1237, 1238, 1239, 1240, 1]]

recon = model.decode(tokens)
print(recon)
# ["Hello", "World :)"]

loss = model.forward(text, max_len=2048)
```

# Training
Install train dependencies
```zsh
pip install -e '.[train]'
```

Setup accelerate config
```sh
accelerate config
```

Train with a huggingface dataset
```py
from neural_tokenizer import Trainer, TrainConfig

train_config = TrainConfig()

trainer = Trainer(
    model=model,
    train_config=train_config
)

trainer.train()

```

# TODO
- [ ] Dataloader with HF datasets
- [ ] Add training 
- [ ] GAN training
- [ ] Variational + continuous bottleneck
