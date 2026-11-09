"""Official CrossFlow backbone components (crossflowDiT).

Vendored (adapted for import hygiene) from the official C-Fun/cross-flow repo:
    https://github.com/C-Fun/cross-flow.git
    models/crossflowDiT.py, models/embedder.py, models/torch_models.py

This is the *authoritative* backbone we must match for a fair comparison against
CrossFlow: RMSNorm + SwiGLU + 2D RoPE + QK-norm + zero-initialized vector gates,
with in-context prefix tokens (class + target time r) and a pixel unpatchify head.

Reference parameter counts (B/L/XL at latent 32x32x4 -> 256x256x3):
    B  (depth 12, hidden 768,  heads 12): 87.4M
    L  (depth 24, hidden 1024, heads 16): 305.6M
    XL (depth 28, hidden 1152, heads 16): 450.3M

Everything below is kept faithful to the original; only the module-qualified
import of ``TorchLinear``/``TorchEmbedding``/``RMSNorm``/``SwiGLUMlp`` was
inlined so the file is self-contained.
"""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# torch_models.py (official) — inline
# ============================================================
class TorchLinear(nn.Module):
    """A linear layer similar to torch.nn.Linear (Flax-style init)."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        weight_init="scaled_variance",
        init_constant=1.0,
        bias_init="zeros",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight_init = weight_init
        self.init_constant = init_constant
        self.bias_init = bias_init

        if self.weight_init == "scaled_variance":
            std = self.init_constant / math.sqrt(self.in_features)
            weight_initializer = partial(nn.init.normal_, std=std)
        elif self.weight_init == "zeros":
            weight_initializer = nn.init.zeros_
        else:
            raise ValueError(f"Invalid weight_init: {self.weight_init}")

        if self.bias_init == "zeros":
            bias_initializer = nn.init.zeros_
        else:
            raise ValueError(f"Invalid bias_init: {self.bias_init}")

        self._flax_linear = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias,
        )
        weight_initializer(self._flax_linear.weight)
        if self.bias:
            bias_initializer(self._flax_linear.bias)

    def forward(self, x):
        return self._flax_linear(x)


class TorchEmbedding(nn.Module):
    """An embedding layer similar to torch.nn.Embedding (Flax-style init)."""

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        weight_init="scaled_variance",
        init_constant=1.0,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight_init = weight_init
        self.init_constant = init_constant

        if self.weight_init is None:
            std = 0.02
        elif self.weight_init == "scaled_variance":
            std = self.init_constant / math.sqrt(self.embedding_dim)
        else:
            raise ValueError(f"Invalid weight_init: {self.weight_init}")

        self._flax_embedding = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
        )
        nn.init.normal_(self._flax_embedding.weight, std=std)

    def forward(self, x):
        return self._flax_embedding(x)


class RMSNorm(nn.Module):
    """Root Mean Square Normalization."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.dim))

    def _norm(self, x):
        mean_square = torch.mean(torch.square(x), dim=-1, keepdim=True)
        return x * torch.rsqrt(mean_square + self.eps)

    def forward(self, x):
        output = self._norm(x).to(x.dtype)
        return output * self.weight


class SwiGLUMlp(nn.Module):
    """Swish-Gated Linear Unit MLP."""

    def __init__(
        self,
        in_features,
        hidden_features,
        weight_init="scaled_variance",
        weight_init_constant=1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.weight_init = weight_init
        self.weight_init_constant = weight_init_constant

        init_kwargs = dict(
            bias=False,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )
        self.w1 = TorchLinear(self.in_features, self.hidden_features, **init_kwargs)
        self.w3 = TorchLinear(self.in_features, self.hidden_features, **init_kwargs)
        self.w2 = TorchLinear(self.hidden_features, self.in_features, **init_kwargs)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ============================================================
# embedder.py (official) — inline
# ============================================================
class TimestepEmbedder(nn.Module):
    """Embeds a scalar timestep (or scalar conditioning) into a vector."""

    def __init__(
        self,
        hidden_size,
        frequency_embedding_size=256,
        weight_init="scaled_variance",
        init_constant=1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.weight_init = weight_init
        self.init_constant = init_constant

        init_kwargs = dict(
            out_features=self.hidden_size,
            bias=True,
            weight_init=self.weight_init,
            init_constant=self.init_constant,
            bias_init="zeros",
        )
        self.mlp = nn.Sequential(
            TorchLinear(self.frequency_embedding_size, **init_kwargs),
            nn.SiLU(),
            TorchLinear(self.hidden_size, **init_kwargs),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        args = t[:, None].to(torch.float32) * freqs[None].to(t.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], axis=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations (num_classes + 1 for CFG)."""

    def __init__(
        self, num_classes, hidden_size, weight_init="scaled_variance", init_constant=1.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.weight_init = weight_init
        self.init_constant = init_constant

        self.embedding_table = TorchEmbedding(
            self.num_classes + 1,
            self.hidden_size,
            weight_init=self.weight_init,
            init_constant=self.init_constant,
        )

    def forward(self, labels):
        return self.embedding_table(labels)


class BottleneckPatchEmbedder(nn.Module):
    """Image to Patch Embedding (conv patch -> PCA channels -> hidden)."""

    def __init__(
        self, input_size, initial_patch_size, pca_channels, in_channels, hidden_size, bias=True
    ):
        super().__init__()
        self.input_size = input_size
        self.initial_patch_size = initial_patch_size
        self.in_channels = in_channels
        self.pca_channels = pca_channels
        self.hidden_size = hidden_size
        self.bias = bias

        self.patch_size = (self.initial_patch_size, self.initial_patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(self.input_size)

        self.flatten = True
        self.proj1 = nn.Conv2d(
            self.in_channels,
            self.pca_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=self.bias,
        )
        self.proj2 = nn.Conv2d(
            self.pca_channels,
            self.hidden_size,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=self.bias,
        )

        kh = kw = self.patch_size[0]
        fan_in = kh * kw * self.in_channels
        fan_out = self.pca_channels
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        nn.init.uniform_(self.proj1.weight, -limit, limit)

        fan_in = self.pca_channels
        fan_out = self.hidden_size
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        nn.init.uniform_(self.proj2.weight, -limit, limit)
        if self.bias:
            nn.init.zeros_(self.proj1.bias)
            nn.init.zeros_(self.proj2.bias)

    def _init_img_size(self, img_size: int):
        img_size = (img_size, img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W, f"{x.shape}"
        x = self.proj2(self.proj1(x))
        x = x.permute(0, 2, 3, 1).reshape(B, -1, x.shape[1])
        return x


# ============================================================
# crossflowDiT.py (official) — inline
# ============================================================
def unsqueeze(t, dim):
    return t.unsqueeze(dim)


def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0):
    """Precompute 2D-RoPE rotation *angles* (real), shape (seq_len, dim // 2)."""
    dim = dim // 2  # for 2d rotary embeddings
    T = int(seq_len**0.5)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(T, dtype=torch.float32)
    freqs_h = torch.einsum("i,j->ij", positions, freqs)
    freqs_w = torch.einsum("i,j->ij", positions, freqs)
    freqs = torch.cat(
        [
            torch.tile(freqs_h[:, None, :], (1, T, 1)),
            torch.tile(freqs_w[None, :, :], (T, 1, 1)),
        ],
        axis=-1,
    )
    angles = freqs.reshape(seq_len, dim)
    return angles


def apply_rotary_pos_emb(x, angles):
    """Apply 2D RoPE to the last P image-patch tokens; prefix tokens are untouched."""
    P = angles.shape[0]
    cos = torch.cos(angles).to(x.dtype)[None, :, None, :]
    sin = torch.sin(angles).to(x.dtype)[None, :, None, :]

    x_img = x[:, -P:]
    x1 = x_img[..., 0::2]
    x2 = x_img[..., 1::2]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    x_rot = torch.stack([rot1, rot2], dim=-1).flatten(-2)

    return torch.cat([x[:, :-P], x_rot], dim=1)


class RoPEAttention(nn.Module):
    """Multi-head self-attention with RoPE and QK RMS norm."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        weight_init="scaled_variance",
        weight_init_constant=1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.weight_init = weight_init
        self.weight_init_constant = weight_init_constant

        init_kwargs = dict(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=False,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )
        self.q_proj = TorchLinear(**init_kwargs)
        self.k_proj = TorchLinear(**init_kwargs)
        self.v_proj = TorchLinear(**init_kwargs)
        self.out_proj = TorchLinear(**init_kwargs)

        self.head_dim = self.hidden_size // self.num_heads
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope_angles):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rotary_pos_emb(q, rope_angles)
        k = apply_rotary_pos_emb(k, rope_angles)

        query = q / math.sqrt(self.head_dim)
        attn_weights = torch.einsum("bqhd,bkhd->bhqk", query, k)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        attn = attn.reshape(batch, seq_len, self.hidden_size)

        return self.out_proj(attn)


class TransformerBlock(nn.Module):
    """Transformer block with zero-initialized vector gates on residuals."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=8 / 3,
        weight_init="scaled_variance",
        weight_init_constant=1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.weight_init = weight_init
        self.weight_init_constant = weight_init_constant

        self.norm1 = RMSNorm(self.hidden_size)
        self.attn = RoPEAttention(
            self.hidden_size,
            num_heads=self.num_heads,
            weight_init=self.weight_init,
            weight_init_constant=self.weight_init_constant,
        )
        self.norm2 = RMSNorm(self.hidden_size)
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        if hidden_size > 1024:  # only for HSDP code (match official rounding)
            mlp_hidden_dim = (mlp_hidden_dim + 7) // 8 * 8
        self.mlp = SwiGLUMlp(
            self.hidden_size,
            mlp_hidden_dim,
            weight_init=self.weight_init,
            weight_init_constant=self.weight_init_constant,
        )

        self.attn_scale = nn.Parameter(torch.zeros(self.hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(self.hidden_size))

    def forward(self, x, rope_angles):
        x = x + self.attn(self.norm1(x), rope_angles) * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x


class FinalLayer(nn.Module):
    """Final projection layer with RMSNorm and zero init weights."""

    def __init__(self, hidden_size, out_patch_size, out_channels):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_patch_size = out_patch_size
        self.out_channels = out_channels

        self.norm = RMSNorm(self.hidden_size)
        self.linear = TorchLinear(
            self.hidden_size,
            self.out_patch_size * self.out_patch_size * self.out_channels,
            bias=True,
            weight_init="zeros",
            bias_init="zeros",
        )

    def forward(self, x):
        return self.linear(self.norm(x))


class CrossFlowDiT(nn.Module):
    """CrossFlow cross-space Transformer (official `crossflowDiT`).

    Input: noised latent z_t (latent_channels x latent_size x latent_size).
    Output: pixel image F_theta (out_channels x pixel_size x pixel_size).
    Conditioning: in-context prefix tokens (class + target time r).
    """

    def __init__(
        self,
        latent_size: int = 32,
        patch_size: int = 2,
        latent_channels: int = 4,
        out_patch_size: int = 16,
        out_channels: int = 3,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 8 / 3,
        num_classes: int = 1000,
        num_class_tokens: int = 8,
        num_time_tokens: int = 4,
        token_init_constant: float = 1.0,
        embedding_init_constant: float = 1.0,
        weight_init_constant: float = 0.32,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.out_patch_size = out_patch_size
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes

        self.num_class_tokens = num_class_tokens
        self.num_time_tokens = num_time_tokens

        self.token_init_constant = token_init_constant
        self.embedding_init_constant = embedding_init_constant
        self.weight_init_constant = weight_init_constant

        assert latent_size % patch_size == 0
        self.grid_size = latent_size // patch_size
        self.pixel_size = self.grid_size * self.out_patch_size

        self.x_embedder = BottleneckPatchEmbedder(
            self.latent_size,
            self.patch_size,
            128 if self.hidden_size <= 1024 else 256,
            self.latent_channels,
            self.hidden_size,
            bias=True,
        )

        embed_kwargs = dict(
            hidden_size=self.hidden_size,
            weight_init="scaled_variance",
            init_constant=self.embedding_init_constant,
        )
        self.r_embedder = TimestepEmbedder(**embed_kwargs)
        self.y_embedder = LabelEmbedder(self.num_classes, **embed_kwargs)

        token_initializer = partial(
            nn.init.normal_, std=self.token_init_constant / math.sqrt(self.hidden_size)
        )
        self.time_tokens = nn.Parameter(
            token_initializer(torch.empty(1, self.num_time_tokens, self.hidden_size))
        )
        self.class_tokens = nn.Parameter(
            token_initializer(torch.empty(1, self.num_class_tokens, self.hidden_size))
        )

        total_tokens = (
            self.x_embedder.num_patches + self.num_class_tokens + self.num_time_tokens
        )
        self.prefix_tokens = self.num_class_tokens + self.num_time_tokens
        self.head_dim = self.hidden_size // self.num_heads
        self.register_buffer(
            "rope_angles",
            precompute_rope_freqs(self.head_dim, self.x_embedder.num_patches),
        )
        self.pos_embed = nn.Parameter(
            nn.init.normal_(torch.empty(1, total_tokens, self.hidden_size), std=0.02)
        )

        block_kwargs = dict(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            weight_init="scaled_variance",
            weight_init_constant=self.weight_init_constant,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(**block_kwargs) for _ in range(self.depth)]
        )
        self.final_layer = FinalLayer(self.hidden_size, self.out_patch_size, self.out_channels)

    def unpatchify(self, x):
        """(B, num_patches, out_patch**2 * C) -> (B, C, pixel_size, pixel_size)."""
        c = self.out_channels
        p = self.out_patch_size
        h = w = self.grid_size
        assert h * w == x.shape[1]
        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape((x.shape[0], c, h * p, w * p))

    def _build_sequence(self, x, r, y):
        x_embed = self.x_embedder(x)
        r_embed = self.r_embedder(r)
        y_embed = self.y_embedder(y)

        time_tokens = self.time_tokens + unsqueeze(r_embed, 1)
        class_tokens = self.class_tokens + unsqueeze(y_embed, 1)

        seq = torch.cat([class_tokens, time_tokens, x_embed], axis=1)
        seq = seq + self.pos_embed
        return seq

    def forward(self, x, r, y):
        """x: (B, latent_channels, latent_size, latent_size); r: (B,); y: (B,)
        Returns F_theta: (B, out_channels, pixel_size, pixel_size)."""
        seq = self._build_sequence(x, r, y)
        for block in self.blocks:
            seq = block(seq, self.rope_angles)
        tokens = seq[:, self.prefix_tokens :]
        return self.unpatchify(self.final_layer(tokens))

    def forward_hidden(self, x, r, y):
        """Return the spatial hidden tokens before the output head (for representation
        heads / predict-then-condition). Shape: (B, num_patches, hidden_size)."""
        seq = self._build_sequence(x, r, y)
        for block in self.blocks:
            seq = block(seq, self.rope_angles)
        return seq[:, self.prefix_tokens :]

    def forward_with_hidden(self, x, r, y):
        """Return (image, spatial_hidden) in one pass."""
        seq = self._build_sequence(x, r, y)
        for block in self.blocks:
            seq = block(seq, self.rope_angles)
        tokens = seq[:, self.prefix_tokens :]
        image = self.unpatchify(self.final_layer(tokens))
        return image, tokens


# Official size presets (B/L/XL).
CROSSFLOWDIT_B_2 = partial(
    CrossFlowDiT,
    latent_size=32,
    patch_size=2,
    out_patch_size=16,
    depth=12,
    hidden_size=768,
    num_heads=12,
)
CROSSFLOWDIT_L_2 = partial(
    CrossFlowDiT,
    latent_size=32,
    patch_size=2,
    out_patch_size=16,
    depth=24,
    hidden_size=1024,
    num_heads=16,
)
CROSSFLOWDIT_XL_2 = partial(
    CrossFlowDiT,
    latent_size=32,
    patch_size=2,
    out_patch_size=16,
    depth=28,
    hidden_size=1152,
    num_heads=16,
)

CROSSFLOWDIT_PRESETS = {
    "B": CROSSFLOWDIT_B_2,
    "L": CROSSFLOWDIT_L_2,
    "XL": CROSSFLOWDIT_XL_2,
}
