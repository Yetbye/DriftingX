"""Fetch SD-VAE-ft-mse (used for latent-space drift tests) via an HF mirror.

Lands in <repo>/hf_cache/sdvae so that SDVAE_PATH can point straight at it
and dataset/vae.py loads it with local_files_only=True (no hub access at runtime).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download

repo_root = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else repo_root / "hf_cache" / "sdvae"

path = snapshot_download(
    "stabilityai/sd-vae-ft-mse",
    local_dir=str(out),
    allow_patterns=["config.json", "configuration.json", "diffusion_pytorch_model.safetensors"],
)
print("Downloaded to:", path)
assert (out / "config.json").is_file() and (out / "diffusion_pytorch_model.safetensors").is_file()
