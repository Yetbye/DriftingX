#!/usr/bin/env python
"""Compute CIFAR-10 reference statistics for FID (cached to .npz).

FID needs the real-data feature mean/covariance in Inception space. This
computes them once from the CIFAR-10 test set and caches them, so training-time
eval can call `_load_ref_stats("cifar10")` offline.

    python scripts/compute_fid_stats.py --dataset cifar10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.env import CIFAR10_FID_NPZ, CIFAR10_PATH  # noqa: E402


def compute_cifar10_stats(num=10000, out=Path(CIFAR10_FID_NPZ)) -> dict:
    import torch
    from torchvision import datasets, transforms

    from utils.fid_util import _compute_stats

    tf = transforms.Compose([transforms.ToTensor()])  # [0,1] RGB
    ds = datasets.CIFAR10(str(CIFAR10_PATH), train=False, download=True, transform=tf)
    imgs = []
    n = min(num, len(ds))
    for i in range(n):
        x, _ = ds[i]
        imgs.append((x.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    arr = np.stack(imgs)
    stats = _compute_stats(arr, n, compute_logits=False, compute_features=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, mu=stats["mu"], sigma=stats["sigma"])
    print(f"wrote {out}  ({n} samples, mu{stats['mu'].shape})")
    return {"mu": stats["mu"], "sigma": stats["sigma"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    ap.add_argument("--num", type=int, default=10000)
    args = ap.parse_args()
    if args.dataset == "cifar10":
        compute_cifar10_stats(args.num)


if __name__ == "__main__":
    main()
