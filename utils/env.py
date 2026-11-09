"""Global paths for the Drift PyTorch port."""

from __future__ import annotations

import os
from pathlib import Path

from utils.offline import apply_offline_env

# Resolve an offline bundle (if DRIFT_OFFLINE_BUNDLE is set) before reading any
# path below, so a single variable configures HF_ROOT / SDVAE_PATH / FID stats.
apply_offline_env()

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _PROJECT_ROOT.parents[2]

IMAGENET_PATH = os.environ.get("IMAGENET_PATH", str(_WORKSPACE_ROOT / "imagenet"))
IMAGENET_CACHE_PATH = os.environ.get("IMAGENET_CACHE_PATH", str(_WORKSPACE_ROOT / "imagenet_latent_cache"))
IMAGENET_FID_NPZ = os.environ.get("IMAGENET_FID_NPZ", str(_WORKSPACE_ROOT / "imagenet_stats" / "imagenet_256_fid_stats.npz"))
IMAGENET_PR_NPZ = os.environ.get("IMAGENET_PR_NPZ", str(_WORKSPACE_ROOT / "imagenet_stats" / "imagenet_val_prc_arr0.npz"))

HF_REPO_ID = os.environ.get("HF_REPO_ID", "Goodeat/drifting")
HF_ROOT = os.environ.get("HF_ROOT", str(_PROJECT_ROOT / "hf_cache"))

# MNIST test dataset root (torchvision layout: <root>/MNIST/raw/...)
MNIST_PATH = os.environ.get("MNIST_PATH", str(_PROJECT_ROOT / "data" / "mnist"))
CIFAR10_PATH = os.environ.get("CIFAR10_PATH", str(_PROJECT_ROOT / "data" / "cifar10"))
CIFAR10_FID_NPZ = os.environ.get("CIFAR10_FID_NPZ", str(_PROJECT_ROOT / "data" / "cifar10_stats.npz"))
