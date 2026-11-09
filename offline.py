"""Offline resource resolution for air-gapped clusters.

The training cluster has no network and no interactive terminal. Everything the
code needs from the outside world (VAE weights, MAE feature extractors, ConvNeXt,
FID/IS inception weights, reference statistics) is gathered on a networked
machine with ``scripts/fetch_offline_assets.py`` into a single bundle directory,
copied to the cluster, and pointed at with one variable::

    export DRIFT_OFFLINE_BUNDLE=/scratch/$USER/offline_bundle

Setting that variable wires every individual path variable (HF_ROOT, SDVAE_PATH,
IMAGENET_FID_NPZ, ...) to the bundle layout, so no other configuration is needed.
Explicitly-set individual variables always win, which keeps local dev unchanged.

Bundle layout produced by the fetch script::

    offline_bundle/
      hf_root/models/mae/pytorch/<name>/{metadata.json,ema_params.pt}
      hf_root/convnext/convnextv2-base-22k-224/
      sdvae/{config.json,diffusion_pytorch_model.safetensors}
      torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth
      fid/imagenet_256_fid_stats.npz
      imnet_stats/imagenet_val_prc_arr0.npz
"""

from __future__ import annotations

import os
from pathlib import Path

BUNDLE_ENV = "DRIFT_OFFLINE_BUNDLE"


def offline_bundle() -> Path | None:
    value = os.environ.get(BUNDLE_ENV)
    if not value:
        return None
    return Path(value).expanduser().resolve()


def is_offline() -> bool:
    return offline_bundle() is not None


def bundle_path(*parts: str) -> Path | None:
    root = offline_bundle()
    if root is None:
        return None
    return root.joinpath(*parts)


def _set_default(key: str, value: Path) -> None:
    """Set env[key] only if unset, so explicit overrides keep working."""
    if not os.environ.get(key):
        os.environ[key] = str(value)


def apply_offline_env() -> dict[str, str]:
    """Wire all path variables from the bundle. Returns what was set (for logging).

    Also pins HuggingFace/transformers to offline mode so a missing cache fails
    fast with a clear error instead of hanging on a network timeout.
    """
    root = offline_bundle()
    if root is None:
        return {}
    applied: dict[str, str] = {}

    def _apply(key: str, value: Path) -> None:
        if not os.environ.get(key):
            os.environ[key] = str(value)
            applied[key] = str(value)

    _apply("HF_ROOT", root / "hf_root")
    _apply("HF_HOME", root / "hf_root")
    _apply("HF_HUB_CACHE", root / "hf_root" / "hub")
    _apply("SDVAE_PATH", root / "sdvae")
    _apply("DRIFT_CONVNEXT_PATH", root / "hf_root" / "convnext" / "convnextv2-base-22k-224")
    _apply("IMAGENET_FID_NPZ", root / "fid" / "imagenet_256_fid_stats.npz")
    _apply("IMAGENET_PR_NPZ", root / "imnet_stats" / "imagenet_val_prc_arr0.npz")
    # torch.hub cache: where pytorch-fid looks for pt_inception-*.pth
    _apply("TORCH_HOME", root / "torch")

    # Force offline behaviour for the hub clients so errors are immediate.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return applied


def describe() -> str:
    root = offline_bundle()
    if root is None:
        return "offline bundle: not configured (DRIFT_OFFLINE_BUNDLE unset)"
    lines = [f"offline bundle: {root}"]
    for key in ("HF_ROOT", "SDVAE_PATH", "DRIFT_CONVNEXT_PATH", "IMAGENET_FID_NPZ", "TORCH_HOME"):
        lines.append(f"  {key}={os.environ.get(key, '<unset>')}")
    return "\n".join(lines)
