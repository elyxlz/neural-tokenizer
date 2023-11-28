import functools

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange
import flash_attn
from transformers import PretrainedConfig, PreTrainedModel

from neural_tokenizer.fsq import FSQ

def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def xavier_init(layer):
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def rms_norm(x, scale, eps):
    dtype = functools.reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


def rotate_half(x: Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, position_ids: Tensor
) -> tuple[Tensor, Tensor]:
    cos = cos[position_ids].unsqueeze(
        1
    )  # [seq_len, dim] -> [batch_size, 1, seq_len, head_dim]
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 240000,
        base: int = 10000,
        device: str = None,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.inv_freq: Tensor

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        self.cos_cached: Tensor
        self.sin_cached: Tensor

    def forward(self, x: Tensor, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


class Patch(nn.Module):
    """ Resample and resize the input """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        factor: int,
    ):
        super().__init__()

        self.is_encoder = factor > 1
        self.factor = factor
        if not self.is_encoder:
            self.factor = 1 / factor
            
        self.proj = nn.Linear(in_size, out_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.is_encoder:
            x = self.proj.forward(x)
            x = rearrange(x, "b s (p d) -> b (s p) d", p=self.factor)
        else:
            x = rearrange(x, "b (s p) d -> b s (p d)", p=self.factor)
            x = self.proj.forward(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        mlp_factor: int,
        norm: bool,
        zero_out: bool = False,
    ):
        super().__init__()
        out_init = xavier_init if not zero_out else zero_init
        self.net = nn.Sequential(
            xavier_init(nn.Linear(in_size, in_size * mlp_factor, bias=False)),
            nn.GELU(),
            out_init(nn.Linear(in_size * mlp_factor, out_size, bias=False)),
        )
        self.norm = norm
        if self.norm:
            self.norm1 = RMSNorm(in_size)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm1.forward(x) if self.norm else x
        return self.net(x) + residual


class Attention(nn.Module):
    def __init__(
        self,
        size: int,
        num_heads: int,
        window_size: int,
        embed_pos: bool,
        norm: bool,
    ):
        super().__init__()

        self.qkv_proj = xavier_init(nn.Linear(size, size * 3, bias=False))
        self.num_heads = num_heads
        self.head_size = size // num_heads
        self.out_proj = xavier_init(nn.Linear(size, size, bias=False))

        self.embed_pos = embed_pos
        if self.embed_pos:
            self.rotary_emb = RotaryEmbedding(self.head_size)

        self.norm = norm
        if self.norm:
            self.norm1 = RMSNorm(self.head_size)

        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm1.forward(x) if self.norm else x
        qkv = self.qkv_proj.forward(x)
        qkv = rearrange(qkv, "b s (n h d) -> b n h s d", n=3, h=self.num_heads)
        q, k, v = qkv.unbind(dim=1)

        if self.embed_pos:
            cos, sin = self.rotary_emb.forward(v, seq_len=v.shape[1])
            position_ids = torch.arange(v.shape[1], device=x.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        q, k, v = map(lambda t: rearrange(t, "b h s d -> b s h d"), (q, k, v))

        x = flash_attn.flash_attn_func(
            q=q,
            k=k,
            v=v,
            window_size=(self.window_size, self.window_size),
        )
        x = rearrange(x, "b s h d -> b s (h d)")
        x = self.out_proj.forward(x)
        return x + residual


class Block(nn.Module):
    """ patch, attention, and mlp """
    def __init__(
        self,
        in_size: Tensor,
        out_size: Tensor,
        num_heads: int,
        factor: int,
        mlp_factor: int,
        window_size: int,
        embed_pos: bool,
        norm: bool,
        zero_out: bool = False,
    ):
        super().__init__()

        self.patch = Patch(factor=factor) if factor > 1 else nn.Identity()
        self.attn = Attention(
            size=in_size,
            num_heads=num_heads,
            window_size=window_size,
            embed_pos=embed_pos,
            norm=norm,
        )
        self.mlp = Mlp(
            in_size=in_size,
            out_size=out_size,
            mlp_factor=mlp_factor,
            norm=norm,
            zero_out=zero_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch.forward(x) if self.patch.is_encoder else x
        x = self.attn.forward(x)
        x = self.mlp.forward(x)
        x = self.patch.forward(x) if not self.patch.is_encoder else x
        return x


class NeuralTokenizerConfig(PretrainedConfig):
    model_type = "neural_tokenizer"

    def __init__(
        self,
        char_vocab_size: int = 256,
        quantizer_levels: list[int] = [8, 8, 5, 3],
        num_heads: int = 4,
        hidden_sizes: list[int] = [128, 256, 512, 512],
        latent_size: int = 512,
        factors: list[int] = [2, 2, 2, 2],
        mlp_factor: int = 4,
        window_size: int = 128,
        embed_pos: bool = True,
        norm: bool = True,
        zero_out: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.char_vocab_size = char_vocab_size
        self.quantizer_levels = quantizer_levels
        self.num_heads = num_heads
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size
        self.mlp_factor = mlp_factor
        self.factors = factors
        self.window_size = window_size
        self.embed_pos = embed_pos
        self.norm = norm
        self.zero_out = zero_out


class NeuralTokenizerModel(PreTrainedModel):
    
    config_class = NeuralTokenizerConfig
    
    def __init__(self, config: NeuralTokenizerConfig):
        num_layers = len(config.hidden_sizes)
        self.encoder = nn.Sequential(
            nn.Embedding(config.char_vocab_size, config.hidden_sizes[0]),
            *[Block(
                in_size=config.hidden_sizes[i],
                out_size=config.hidden_sizes[i + 1] if i < num_layers - 1 else config.latent_size,
                num_heads=config.num_heads,
                factor=config.factors[i],
                mlp_factor=config.mlp_factor,
                window_size=config.window_size,
                embed_pos=config.embed_pos,
                norm=config.norm,
                zero_out=False,
            ) for i in range(num_layers)],
        )

        self.quantizer = FSQ(
            in_size=config.latent_size,
            levels=config.quantizer_levels,
        )
        
        self.decoder = nn.Sequential(
            *[Block(
                in_size=config.latent_size,
                out_size=config.hidden_sizes[i],
                num_heads=config.num_heads,
                factor=config.factors[i],
                mlp_factor=config.mlp_factor,
                window_size=config.window_size,
                embed_pos=config.embed_pos,
                norm=config.norm,
                zero_out=config.zero_out if i < num_layers - 1 else False,
            ) for i in range(num_layers - 1, -1, -1)],
        )
        
        head_init = zero_init if config.zero_out else xavier_init
        self.lm_head = head_init(nn.Linear(config.hidden_sizes[0], config.char_vocab_size, bias=False))
        
    @torch.inference_mode()
    def encode(self, x: list[str]) -> Tensor:
        latent = self.encoder.forward(x)
        indices = self.quantizer.codes_to_indices(latent)
        return indices
    
    @torch.inference_mode()
    def decode(self, indices: Tensor) -> list[str]:
        latent = self.quantizer.indices_to_codes(indices)
        x = self.decoder.forward(latent)
        x = self.lm_head.forward(x)
        return x
    
    def forward(self, x: list[str]) -> Tensor:
        latent = self.encoder.forward(x)
        q_latent = self.quantizer.forward(latent)
        recon = self.decoder.forward(q_latent)
        xe_loss = F.cross_entropy(recon, torch.tensor(x))
        return xe_loss