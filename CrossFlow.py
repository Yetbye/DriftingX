"""X-Drifting generator: latent noise -> pixel, using the official CrossFlow backbone.

This wrapper binds the **official** CrossFlow backbone (`models/crossflow_official.py`,
vendored from `https://github.com/C-Fun/cross-flow.git`) to the Drift training
framework. The backbone is kept byte-for-byte faithful to the official
`crossflowDiT` so our comparisons against CrossFlow are fair — same parameters,
same modules (RMSNorm + SwiGLU + 2D RoPE + QK-norm + zero-init vector gates),
same in-context prefix tokens (class + target time r), same pixel unpatchify head.

Two modes, selected by the training objective (see docs/EXPERIMENT-DESIGN.md):

  * **L-P / X-Drifting**  (noise=latent, supervision=pixel) — the drift-loss model.
  * **CrossFlow L_CF**    (same backbone, CrossFlow's differential objective).

The Drift framework drives the model through the interface
``model(c=labels, cfg_scale=..., train=False)["samples"] -> NHWC pixels in [-1,1]``,
matching `models/generator.py:DitGen`. Because Drift is single-step (noise -> data),
the destination time ``r`` is fixed to 0 — the architecture is unchanged, only the
``r``-embedding degenerates to a learnable constant bias.

Sizes match the official presets exactly:

    B  depth=12 hidden=768  heads=12 ->  87.4M
    L  depth=24 hidden=1024 heads=16 -> 305.6M
    XL depth=28 hidden=1152 heads=16 -> 450.3M

Reference: `docs/BACKBONE-COMPARISON.md`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.crossflow_official import CROSSFLOWDIT_PRESETS, CrossFlowDiT


# Official per-size backbone config (must match C-Fun/cross-flow `crossflowDiT_*_2`).
_OFFICIAL_SIZE = {
    "B": dict(depth=12, hidden_size=768, num_heads=12),
    "L": dict(depth=24, hidden_size=1024, num_heads=16),
    "XL": dict(depth=28, hidden_size=1152, num_heads=16),
}


def resolve_size(size, hidden_size=None, depth=None, num_heads=None):
    """Resolve (hidden_size, depth, num_heads); explicit values win over the preset."""
    preset = {}
    if size is not None:
        key = str(size).strip().upper()
        if key not in _OFFICIAL_SIZE:
            raise ValueError(f"Unknown CrossFlow size {size!r}; expected one of {sorted(_OFFICIAL_SIZE)}.")
        preset = _OFFICIAL_SIZE[key]
    hidden_size = hidden_size if hidden_size is not None else preset.get("hidden_size")
    depth = depth if depth is not None else preset.get("depth")
    num_heads = num_heads if num_heads is not None else preset.get("num_heads")
    if hidden_size is None or depth is None or num_heads is None:
        raise ValueError("Provide `size` (B/L/XL) or explicit hidden_size/depth/num_heads.")
    return int(hidden_size), int(depth), int(num_heads)


class CrossFlow(nn.Module):
    """Drift-framework generator wrapping the official CrossFlow backbone.

    Interface (same as `models/generator.py:DitGen`):
        model(c=labels, cfg_scale=..., train=False)["samples"] -> NHWC pixels in [-1,1]

    The output is produced by the official pixel unpatchify head (tanh-bounded to
    [-1, 1] to match the Drift pipeline's image convention).
    """

    def __init__(
        self,
        num_classes=1000,
        size: Optional[str] = None,
        img_size=256,
        patch_size=16,          # input latent patch size (official: 2 on 32x32 latent)
        latent_channels=32,
        latent_size=32,
        latent_patch_size=2,    # alias for patch_size (kept for config compatibility)
        hidden_size=None,
        depth=None,
        num_heads=None,
        mlp_ratio: float = 8 / 3,
        num_context_tokens=16,  # split into class/time tokens below (official: 8+4)
        num_class_tokens: Optional[int] = None,
        num_time_tokens: Optional[int] = None,
        output_activation="none",
        grad_checkpoint=False,
        out_patch_size=None,    # derived from img_size/grid when absent; round-trips config
        arch=None,              # accepted so config dicts work as module kwargs; unused
    ):
        super().__init__()
        del arch

        hidden_size, depth, num_heads = resolve_size(size, hidden_size, depth, num_heads)

        # `latent_patch_size` is the authoritative name in our configs; `patch_size`
        # is kept as a legacy alias. The official name is `patch_size`.
        in_patch = int(latent_patch_size)

        # Output pixel patch: derived from img_size and the token grid so the
        # latent->pixel mapping matches the official (grid = latent_size/patch).
        grid_size = int(latent_size) // in_patch
        if int(img_size) % grid_size != 0:
            raise ValueError(
                f"img_size {img_size} not divisible by token grid {grid_size} "
                f"(= latent_size {latent_size} // patch_size {in_patch})"
            )
        derived_out_patch = int(img_size) // grid_size
        out_patch_size = int(out_patch_size) if out_patch_size is not None else derived_out_patch

        # Official prefix-token split (class=8 + time=4). `num_context_tokens` is the
        # legacy total and is only used to derive the split when the explicit counts
        # are absent; default to the official 8/4 to match parameter counts exactly.
        if num_class_tokens is None and num_time_tokens is None:
            num_class_tokens, num_time_tokens = 8, 4
        else:
            num_class_tokens = int(num_class_tokens) if num_class_tokens is not None else 8
            num_time_tokens = int(num_time_tokens) if num_time_tokens is not None else 4

        self.num_classes = num_classes
        self.size = str(size).strip().upper() if size is not None else None
        self.img_size = int(img_size)
        self.latent_channels = int(latent_channels)
        self.latent_size = int(latent_size)
        self.latent_patch_size = in_patch
        self.out_patch_size = out_patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = float(mlp_ratio)
        self.num_class_tokens = num_class_tokens
        self.num_time_tokens = num_time_tokens
        self.output_activation = output_activation
        self.grad_checkpoint = bool(grad_checkpoint)

        self.backbone = CrossFlowDiT(
            latent_size=int(latent_size),
            patch_size=in_patch,
            latent_channels=int(latent_channels),
            out_patch_size=out_patch_size,
            out_channels=3,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=float(mlp_ratio),
            num_classes=num_classes,
            num_class_tokens=num_class_tokens,
            num_time_tokens=num_time_tokens,
        )

        # Persisted for artifact(metadata.json) round-trip; keeps size + all dims
        # explicit so rebuilds are exact.
        self.model_config = {
            "arch": "crossflow",
            "size": self.size,
            "num_classes": num_classes,
            "img_size": int(img_size),
            "patch_size": in_patch,
            "latent_channels": int(latent_channels),
            "latent_size": int(latent_size),
            "latent_patch_size": in_patch,
            "out_patch_size": out_patch_size,
            "hidden_size": hidden_size,
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": float(mlp_ratio),
            "num_class_tokens": num_class_tokens,
            "num_time_tokens": num_time_tokens,
            "output_activation": output_activation,
            "grad_checkpoint": self.grad_checkpoint,
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def dummy_input(self):
        return {
            "c": torch.ones(1, dtype=torch.long),
            "cfg_scale": 1.0,
            "temp": 1.0,
            "deterministic": True,
        }

    def rng_keys(self):
        return ["noise"]

    def forward(self, c, cfg_scale=1.0, temp=1.0, deterministic=True, train=False):
        del deterministic, train
        bsz = c.shape[0]
        device = c.device

        # Latent prior noise (NCHW) — the model's cross-space input.
        z = torch.randn(
            (bsz, self.latent_channels, self.latent_size, self.latent_size),
            device=device,
            dtype=torch.float32,
        ) * float(temp)

        # Drift is single-step: destination time r = 0 (architecture unchanged).
        r = torch.zeros((bsz,), device=device, dtype=torch.float32)

        # cfg_scale is unused by the raw crossflowDiT (no CFG term in the head);
        # we accept and ignore it so the Drift framework can pass it through.
        _ = cfg_scale

        imgs = self.backbone(z, r, c)  # [B, 3, H, W], linear output (official has no tanh)
        # The official crossflowDiT head is a zero-init linear (no activation); the
        # output is unconstrained. We keep it byte-faithful. `output_activation` is
        # retained for config compatibility but only "none" is valid against the
        # official head; "tanh" is applied only if explicitly requested.
        if self.output_activation == "tanh":
            imgs = torch.tanh(imgs)
        elif self.output_activation not in ("none", None):
            raise ValueError(f"Unknown output_activation: {self.output_activation}")

        samples = imgs.permute(0, 2, 3, 1).contiguous()  # -> NHWC, Drift convention
        return {"samples": samples, "noise": {"z": z, "r": r}}

    def get_hidden(self, c, temp: float = 1.0):
        """Return the spatial hidden tokens (B, num_patches, hidden) for the
        representation head (mechanism B: predict-then-condition). Samples the same
        latent prior as forward so the hidden state matches a generation pass."""
        bsz = c.shape[0]
        device = c.device
        z = torch.randn(
            (bsz, self.latent_channels, self.latent_size, self.latent_size),
            device=device, dtype=torch.float32,
        ) * float(temp)
        r = torch.zeros((bsz,), device=device, dtype=torch.float32)
        return self.backbone.forward_hidden(z, r, c)


def build_crossflow(size: str, **overrides) -> CrossFlow:
    """Build a CrossFlow generator by size name (B/L/XL)."""
    return CrossFlow(size=size, **overrides)
