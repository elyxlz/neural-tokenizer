import functools

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange
import math
from transformers import PreTrainedModel

from .config import NeuralTokenizerConfig

try:
    import flash_attn

    flash_attn_available = True
except ImportError:
    flash_attn_available = False
    print("\033[91mWARNING: flash_attn not installed, expect non valid results\033[0m")


import inspect
import pdb
import functools

""" debug """

def debug_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            print(
                "Error encountered! Starting debug session from the beginning of the function."
            )
            pdb.runcall(func, *args, **kwargs)

    return wrapper


def trace():
    frame = inspect.currentframe().f_back
    pdb.Pdb().set_trace(frame)



""" helper """


def find_multiple(x: int, n: int) -> int:
    """Finds the smallest multiple of n greater than or equal to x, e.g. find_multiple(5, 3) = 6"""
    return math.ceil(x / n) * n


def pad_tensors(
    tensors: list[Tensor],
    max_len: int = None,
    pad_value: int = -1,
) -> tuple[Tensor, Tensor]:
    """
    Pad every item in the sequence to a the longest item in the batch.
    """

    longest = max([t.size(0) for t in tensors]) if max_len is None else max_len
    tensors = torch.stack(
        [F.pad(t, (0, longest - t.shape[0]), value=pad_value) for t in tensors]
    )
    mask = torch.stack([t != pad_value for t in tensors])
    return tensors, mask


def unpad_tensors(tensors: Tensor, mask: Tensor) -> list[Tensor]:
    """Unbinds and unpads stacked tensors"""
    tensors = tensors.unbind(dim=0)
    mask = mask.unbind(dim=0)
    out = [t[m] for t, m in zip(tensors, mask)]
    return out


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
    """Resample and resize the input with a linear proj"""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        factor: int,
        zero_out: bool = False,
    ):
        super().__init__()
        self.factor = factor
        self.is_encoder = out_size >= in_size

        proj_size = out_size // factor if self.is_encoder else out_size * factor
        self.proj = xavier_init(nn.Linear(in_size, proj_size))

        if zero_out:
            nn.init.zeros_(self.proj.weight)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        x = self.proj.forward(x)
        if self.is_encoder:
            x = rearrange(x, "b (s p) d -> b s (p d)", p=self.factor)
            mask = mask[:, :: self.factor]
        else:
            x = rearrange(x, "b s (p d) -> b (s p) d", p=self.factor)
            mask = torch.repeat_interleave(mask, self.factor, dim=1)

        return x, mask


class Mlp(nn.Module):
    def __init__(
        self,
        size: int,
        mlp_factor: int,
        norm: bool,
    ):
        super().__init__()
        self.net = nn.Sequential(
            xavier_init(nn.Linear(size, size * mlp_factor, bias=False)),
            nn.GELU(),
            xavier_init(nn.Linear(size * mlp_factor, size, bias=False)),
        )
        self.norm = norm
        if self.norm:
            self.norm1 = RMSNorm(size)

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
            self.norm1 = RMSNorm(size)

        self.window_size = window_size

    # TODO: varlen flash attn func
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
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

        # use flash attn package until F.sdpa supports windowed attention
        if flash_attn_available:
            x = flash_attn.flash_attn_func(
                q=q,
                k=k,
                v=v,
                window_size=(self.window_size, self.window_size),
                # mask=mask,
            )
            x = rearrange(x, "b s h d -> b s (h d)")

        x = self.out_proj.forward(x)
        x += residual
        return x


class Block(nn.Module):
    """patch, attention, and mlp"""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        num_heads: int,
        factor: int,
        mlp_factor: int,
        window_size: int,
        embed_pos: bool,
        norm: bool,
        zero_out: bool,
    ):
        super().__init__()

        self.patch = (
            Patch(
                in_size=in_size,
                out_size=out_size,
                factor=factor,
                zero_out=zero_out,
            )
            if factor > 1
            else nn.Identity()
        )

        self.attn = Attention(
            size=out_size if self.patch.is_encoder else in_size,
            num_heads=num_heads,
            window_size=window_size,
            embed_pos=embed_pos,
            norm=norm,
        )
        self.mlp = Mlp(
            size=out_size if self.patch.is_encoder else in_size,
            mlp_factor=mlp_factor,
            norm=norm,
        )

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        x, mask = self.patch.forward(x, mask) if self.patch.is_encoder else (x, mask)
        x = self.attn.forward(x, mask)
        x = self.mlp.forward(x)
        x, mask = (
            self.patch.forward(x, mask) if not self.patch.is_encoder else (x, mask)
        )
        return x, mask


class Encoder(nn.Module):
    def __init__(
        self,
        config: NeuralTokenizerConfig,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Block(
                    in_size=config.hidden_sizes[i],
                    out_size=config.hidden_sizes[i + 1]
                    if i < config.num_layers - 1
                    else config.latent_size,
                    num_heads=config.num_heads,
                    factor=config.factors[i],
                    mlp_factor=config.mlp_factor,
                    window_size=config.window_size,
                    embed_pos=config.embed_pos,
                    norm=config.norm,
                    zero_out=False,
                )
                for i in range(config.num_layers)
            ],
        )

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            block: Block
            x, mask = block.forward(x, mask)
        return x, mask


class Decoder(nn.Module):
    def __init__(
        self,
        config: NeuralTokenizerConfig,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Block(
                    in_size=config.hidden_sizes[i + 1]
                    if i < config.num_layers - 1
                    else config.latent_size,
                    out_size=config.hidden_sizes[i],
                    num_heads=config.num_heads,
                    factor=config.factors[i],
                    mlp_factor=config.mlp_factor,
                    window_size=config.window_size,
                    embed_pos=config.embed_pos,
                    norm=config.norm,
                    zero_out=config.zero_out if i < config.num_layers - 1 else False,
                )
                for i in range(config.num_layers - 1, -1, -1)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            x, mask = block.forward(x, mask)
        return x, mask


""" Quantizer """


class FSQ(nn.Module):
    def __init__(
        self,
        config: NeuralTokenizerConfig,
    ):
        super().__init__()

        levels = config.quantizer_levels
        dim = config.latent_size

        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels)

        self.dim = dim

        _basis = torch.cumprod(
            torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32
        )
        self.register_buffer("_basis", _basis)

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        self.in_proj = nn.Linear(self.dim, codebook_dim)
        self.out_proj = nn.Linear(codebook_dim, self.dim)

        self.n_codes = self._levels.prod().item()
        implicit_codebook = self.indices_to_codes(torch.arange(self.n_codes))
        self.register_buffer("implicit_codebook", implicit_codebook)

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        z = self.bound(z)
        zhat = z.round()
        quantized = zhat + (z - zhat).detach()  # STE
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, z: Tensor) -> Tensor:
        """Converts a raw `code` to an index in the codebook."""

        z = self.in_proj.forward(z)
        zhat = self.quantize(z)

        assert (
            zhat.shape[-1] == self.codebook_dim
        ), f"expected dimension of {self.codebook_dim} but found dimension of {zhat.shape[-1]}"
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def indices_to_codes(
        self,
        indices: Tensor,
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""

        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        return self.out_proj(codes)

    def forward(self, z: Tensor) -> Tensor:
        assert (
            z.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"
        z = self.in_proj(z)
        codes = self.quantize(z)
        out = self.out_proj(codes)
        return out


""" Main """


class NeuralTokenizer(PreTrainedModel):
    config_class = NeuralTokenizerConfig

    def __init__(self, config: NeuralTokenizerConfig):
        super().__init__(config)

        self.char_embed = nn.Embedding(
            config.char_vocab_size+1,
            config.hidden_sizes[0],
            padding_idx=config.char_vocab_size,
        )

        self.encoder = Encoder(config)

        self.quantizer = FSQ(config)

        self.decoder = Decoder(config)

        head_init = zero_init if config.zero_out else xavier_init
        self.head = head_init(
            nn.Linear(config.hidden_sizes[0], config.char_vocab_size, bias=False)
        )

    def char_tokenize(self, x: list[str]) -> tuple[Tensor, Tensor]:
        """Takes a list of strings and returns cumprod(factors) block sized utf8 indices to resist the patching mechanism, then globally pads to longest for varlen"""
        x = [torch.tensor(list(map(ord, s)), dtype=torch.long) for s in x]

        # fakepad to block size
        block_size = find_multiple(torch.cumprod(torch.tensor(self.config.factors), dim=0)[-1].item(), 8)
        x = [torch.nn.functional.pad(s, (0, find_multiple(s.size(-1), block_size) - s.size(-1)), value=0) for s in x]
        # assert x[0].size(-1) > 16, x[0].size(-1)

        # real pad
        x, mask = pad_tensors(x, pad_value=self.config.char_vocab_size)
        return x, mask

    def char_detokenize(self, z: Tensor, mask: Tensor) -> list[str]:
        """Takes char lvl utf 8 indices and returns list of strings, removes fakepadding"""

        z_unpadded = unpad_tensors(z, mask)
        out = ["".join(map(chr, t.tolist())) for t in z_unpadded]

        # remove fakepadding (\x00)
        out = [s.replace("\x00", "") for s in out]
        return out

    @torch.inference_mode()
    def encode(self, x: list[str]) -> tuple[Tensor, Tensor]:
        char_indices, mask = self.char_tokenize(x)   
        char_emb = self.char_embed.forward(char_indices)
        codes, mask = self.encoder.forward(char_emb, mask)
        indices = self.quantizer.codes_to_indices(codes)
        return indices, mask

    @torch.inference_mode()
    def decode(self, indices: Tensor, mask: Tensor) -> list[str]:
        latent = self.quantizer.indices_to_codes(indices)
        hidden, mask = self.decoder.forward(latent, mask)
        logits = self.head.forward(hidden)

        # sample logits # TODO
        y = torch.argmax(logits, dim=-1)

        y = self.char_detokenize(y, mask)
        return y

    def forward(self, x: list[str]) -> Tensor:
        char_indices = self.char_tokenize(x)
        codes = self.encoder.forward(char_indices)
        latent = self.quantizer.forward(codes)
        logits = self.decoder.forward(latent)
        xe_loss = F.cross_entropy(logits, char_indices).mean()
        return xe_loss
