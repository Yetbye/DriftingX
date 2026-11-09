#!/usr/bin/env python
"""Compute FID for trained generator runs against a dataset reference.

Generates class-conditional samples from each run's EMA weights and computes FID
against the reference stats cached by `scripts/compute_fid_stats.py`.

    python scripts/eval_fid_all.py --runs runs/C10_P_pixel runs/C10_X_LP \
        --dataset cifar10 --num 10000

Writes runs/c10_eval_summary.json and a sample grid per run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.fid_util import _canonical_dataset_name, _compute_stats, _load_ref_stats  # noqa: E402


def _fid(mu1, s1, mu2, s2):
    from scipy import linalg

    covmean = linalg.sqrtm(s1.dot(s2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def load_model(run: Path, device):
    from models.generator import build_generator_from_config
    from utils.misc import load_config

    cfg = load_config(str(run / "log" / "config.yaml"))
    model = build_generator_from_config(dict(cfg.model), num_classes=int(cfg.dataset.num_classes))
    ema = run / "params_ema" / "ema_params.pt"
    if ema.is_file():
        sd = torch.load(ema, map_location="cpu")
    else:
        ckpts = sorted((run / "checkpoints").glob("step_*.pt"))
        payload = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        sd = payload.get("ema_model") or payload.get("model")
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def generate(model, num, device, batch=128, seed=0):
    torch.manual_seed(seed)
    n_cls = int(getattr(model, "num_classes", 10))
    out, done = [], 0
    while done < num:
        b = min(batch, num - done)
        labels = torch.arange(done, done + b, device=device) % n_cls
        with torch.inference_mode():
            s = model(c=labels, cfg_scale=1.0)["samples"]  # NHWC
        s = ((s.permute(0, 3, 1, 2).contiguous() + 1) / 2).clamp(0, 1)
        out.append(s.cpu())
        done += b
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--num", type=int, default=10000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    canon = _canonical_dataset_name(args.dataset)
    ref = _load_ref_stats(canon)

    results = []
    for r in args.runs:
        run = Path(r).resolve()
        print(f"\n=== {run.name} ===")
        model = load_model(run, device)
        samples = generate(model, args.num, device)
        s_uint8 = (samples.permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)
        stats = _compute_stats(s_uint8, args.num, compute_logits=False, compute_features=False)
        fid = _fid(ref["mu"], ref["sigma"], stats["mu"], stats["sigma"])

        from PIL import Image
        from torchvision.utils import make_grid

        grid = (make_grid(samples[:64], nrow=8, padding=2) * 255).byte().permute(1, 2, 0).numpy()
        img_out = run / "fid_samples.jpg"
        Image.fromarray(grid).save(img_out)

        results.append({"run": str(run), "fid": fid, "num_samples": args.num, "grid": str(img_out)})
        print(f"  FID: {fid:.2f}   grid: {img_out}")

    out = Path("runs/c10_eval_summary.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
