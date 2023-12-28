import functools

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange
from transformers import PretrainedConfig, PreTrainedModel

try:
    import flash_attn

    flash_attn_available = True
except ImportError:
    print("WARNING: flash_attn not installed, expect non valid results")
    flash_attn_available = False


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


def find_multiple(x: int, n: int) -> int:
    """Finds the smallest multiple of n greater than x, e.g. find_multiple(5, 3) = 6"""
    return (x // n + 1) * n


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


class SparseTensor:
    """Abstract class to conveniently managed padded tensors with masks, used for varlen inputs """

    def __init__(
        self,
        data: Tensor,
        mask: Tensor,
    ):
        self.data = data
        self.mask = mask

        assert mask.dtype == torch.bool

    def __iter__(self):
        return iter((self.data, self.mask))

    @classmethod
    def from_unbinded(cls, data: list[Tensor], max_len: int = None) -> "SparseTensor":
        """Pad list of tensors and create mask to return SparseTensor"""
        longest = max([t.shape[0] for t in data]) if max_len is None else max_len
        # left pad with 0s
        data = [F.pad(t, (0,  longest - t.shape[0])) for t in data]
        mask = [t != 0 for t in data]
        return cls(torch.stack(data), torch.stack(mask))

    def unbind(self) -> list[Tensor]:
        """Unbinds and unpads self"""
        data = self.data.unbind(dim=0)
        mask = self.mask.unbind(dim=0)
        out = [t[m] for t, m in zip(data, mask)]
        return out


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

        self.is_encoder = factor > 1
        self.factor = factor
        if not self.is_encoder:
            self.factor = 1 / factor

        init = xavier_init if not zero_out else zero_init
        self.proj = init(nn.Linear(in_size, out_size, bias=False))

    def forward(self, x: SparseTensor) -> SparseTensor:
        data, mask = x
        if self.is_encoder:
            data = self.proj.forward(data)
            data = rearrange(data, "b s (p d) -> b (s p) d", p=self.factor)
            mask = mask[:, :: self.factor]
        else:
            data = rearrange(data, "b (s p) d -> b s (p d)", p=self.factor)
            data = self.proj.forward(data)
            mask = torch.repeat_interleave(mask, self.factor, dim=1)
        return SparseTensor(data, mask)


class Mlp(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        mlp_factor: int,
        norm: bool,
    ):
        super().__init__()
        self.net = nn.Sequential(
            xavier_init(nn.Linear(in_size, in_size * mlp_factor, bias=False)),
            nn.GELU(),
            xavier_init(nn.Linear(in_size * mlp_factor, out_size, bias=False)),
        )
        self.norm = norm
        if self.norm:
            self.norm1 = RMSNorm(in_size)

    def forward(self, x: SparseTensor) -> SparseTensor:
        data, mask = x
        residual = data
        data = self.norm1.forward(data) if self.norm else data
        out = self.net(data) + residual
        return SparseTensor(out, mask)


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

    # TODO: involve mask
    # TODO: varlen flash attn func
    def forward(self, x: SparseTensor) -> SparseTensor:
        data, mask = x
        residual = data
        data = self.norm1.forward(data) if self.norm else data
        qkv = self.qkv_proj.forward(data)
        qkv = rearrange(qkv, "b s (n h d) -> b n h s d", n=3, h=self.num_heads)
        q, k, v = qkv.unbind(dim=1)

        if self.embed_pos:
            cos, sin = self.rotary_emb.forward(v, seq_len=v.shape[1])
            position_ids = torch.arange(v.shape[1], device=x.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        q, k, v = map(lambda t: rearrange(t, "b h s d -> b s h d"), (q, k, v))

        # use flash attn package until F.sdpa supports windowed attention
        if flash_attn_available:
            data = flash_attn.flash_attn_func(
                q=q,
                k=k,
                v=v,
                window_size=(self.window_size, self.window_size),
                #mask=mask,
            )

        data = rearrange(data, "b s h d -> b s (h d)")
        data = self.out_proj.forward(data)
        data += residual
        return SparseTensor(data, mask)


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
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        x = self.patch.forward(x) if self.patch.is_encoder else x
        x = self.attn.forward(x)
        x = self.mlp.forward(x)
        x = self.patch.forward(x) if not self.patch.is_encoder else x
        return x


""" Quantizer """


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


class FSQ(nn.Module):
    def __init__(self, in_size: int, levels: list[int]):
        super().__init__()
        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels)
        self._levels: Tensor

        _basis = torch.cumprod(
            Tensor([1] + levels[:-1]), dim=0, dtype=torch.int32
        )
        self.register_buffer("_basis", _basis)
        self._basis: Tensor

        self.dim = len(levels)

        self.in_proj = nn.Linear(in_size, self.dim, bias=True)
        self.out_proj = nn.Linear(self.dim, in_size, bias=True)

        self.n_codes = self._levels.prod().item()
        # implicit_codebook = self.indices_to_codes(torch.arange(self.n_codes))
        # self.register_buffer("implicit_codebook", implicit_codebook)

    def forward(self, z: SparseTensor) -> SparseTensor:
        data, mask = z
        data = self.in_proj.forward(data)
        zhat = self.quantize(data)
        zhat = self.out_proj.forward(zhat)
        return SparseTensor(zhat, mask)

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quanitzes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: SparseTensor) -> SparseTensor:
        """Converts a `code` to an index in the codebook."""
        zhat, mask = zhat
        assert zhat.shape[-1] == self.dim
        zhat = self._scale_and_shift(zhat)
        return SparseTensor((zhat * self._basis).sum(dim=-1).to(torch.int32), mask)

    def indices_to_codes(self, indices: SparseTensor) -> SparseTensor:
        indices, mask = indices
        """Inverse of `codes_to_indices`."""
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        out = self._scale_and_shift_inverse(codes_non_centered)
        return SparseTensor(self.out_proj.forward(out), mask)


""" Main """


class NeuralTokenizerConfig(PretrainedConfig):
    model_type = "neural_tokenizer"

    def __init__(
        self,
        char_vocab_size: int = 256,
        quantizer_levels: list[int] = [8, 8, 5, 3], # 960 vocab size
        num_heads: int = 4,
        hidden_sizes: list[int] = [32, 32, 64, 64],
        latent_size: int = 128,
        factors: list[int] = [2, 2, 2, 2],
        mlp_factor: int = 2,
        window_size: int = 128,
        embed_pos: bool = True,
        norm: bool = True,
        zero_out: bool = False,
        **kwargs,
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


class NeuralTokenizer(PreTrainedModel):
    config_class = NeuralTokenizerConfig

    def __init__(self, config: NeuralTokenizerConfig):
        super().__init__(config)

        num_layers = len(config.hidden_sizes)

        self.char_emb = nn.Embedding(config.char_vocab_size, config.hdiden_sizes[0])

        self.encoder = nn.Sequential(
            *[
                Block(
                    in_size=config.hidden_sizes[i],
                    out_size=config.hidden_sizes[i + 1]
                    if i < num_layers - 1
                    else config.latent_size,
                    num_heads=config.num_heads,
                    factor=config.factors[i],
                    mlp_factor=config.mlp_factor,
                    window_size=config.window_size,
                    embed_pos=config.embed_pos,
                    norm=config.norm,
                    zero_out=False,
                )
                for i in range(num_layers)
            ],
        )

        self.quantizer = FSQ(
            in_size=config.latent_size,
            levels=config.quantizer_levels,
        )

        head_init = zero_init if config.zero_out else xavier_init
        self.decoder = nn.Sequential(
            *[
                Block(
                    in_size=config.latent_size,
                    out_size=config.hidden_sizes[i],
                    num_heads=config.num_heads,
                    factor=config.factors[i],
                    mlp_factor=config.mlp_factor,
                    window_size=config.window_size,
                    embed_pos=config.embed_pos,
                    norm=config.norm,
                    zero_out=config.zero_out if i < num_layers - 1 else False,
                )
                for i in range(num_layers - 1, -1, -1)
            ]
        )

        head_init = zero_init if config.zero_out else xavier_init
        self.head = head_init(nn.Linear(config.hidden_sizes[0]), config.char_vocab_size, bias=False)

    @staticmethod
    def char_tokenize(x: list[str]) -> SparseTensor:
        """Takes a list of strings and returns utf8 indices as SparseTensor for var len"""
        x = [torch.tensor(list(map(ord, s)), dtype=torch.long) for s in x]
        x = SparseTensor.from_unbinded(x)
        return x

    @staticmethod
    def char_detokenize(z: SparseTensor) -> list[str]:
        """Takes char lvl utf 8 nested tensor indices and returns list of strings"""
        out = ["".join(map(chr, t.tolist())) for t in z.unbind()]
        return out

    @torch.inference_mode()
    def encode(self, x: list[str]) -> SparseTensor:
        char_indices = self.char_tokenize(x)
        codes = self.encoder.forward(char_indices)
        indices = self.quantizer.codes_to_indices(codes)
        return indices

    @torch.inference_mode()
    def decode(self, indices: SparseTensor) -> list[str]:
        latent = self.quantizer.indices_to_codes(indices)
        logits = self.decoder.forward(latent)

        # sample logits # TODO
        y = torch.argmax(logits, dim=-1)

        y = self.char_detokenize(y)
        return y

    def forward(self, x: list[str]) -> Tensor:
        char_indices = self.char_tokenize(x)
        codes = self.encoder.forward(char_indices)
        latent = self.quantizer.forward(codes)
        logits = self.decoder.forward(latent)
        xe_loss = F.cross_entropy(logits, char_indices).mean()
        return xe_loss
