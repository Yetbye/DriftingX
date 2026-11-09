#!/usr/bin/env python
"""Evaluate MNIST generation QUALITY for a trained generator (drift loss).

Drift loss is an internal fixed-point residual — its numeric value has no
cross-model meaning. The thing worth reporting is **whether the generator
produces clear, class-correct digits**. This script measures exactly that:

  1. Class-conditional accuracy: a reference MNIST classifier judges each
     generated sample; acc = fraction where predicted == conditioning label.
     This is the headline number — it says "when asked for a 5, does it draw a 5".
  2. Classifier confidence (mean softmax prob of the predicted class): how crisp
     the digits are, independent of being right.
  3. FID against real MNIST (needs MNIST reference stats; computed once and cached).

Usage:
    python scripts/eval_mnist_gen.py --run runs/X_LP_drift [--num 5000]
    python scripts/eval_mnist_gen.py --run runs/E1_P_pixel runs/X_LP_drift

It loads the EMA weights from <run>/params_ema/ (or the latest checkpoint), the
model config from <run>/log/config.yaml, generates class-conditional samples, and
scores them. Writes a JSON summary + a sample grid per run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.misc import load_config  # noqa: E402


# ---------------------------------------------------------------- reference classifier
class MNISTClassifier(nn.Module):
    """Small CNN; ~99% test accuracy in a few epochs."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 16 * 16, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.fc(self.conv(x))


def _to_gray1(x_nchw3: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) in [0,1] -> (B,1,H,W) grayscale for the classifier."""
    if x_nchw3.shape[1] == 3:
        return (0.299 * x_nchw3[:, 0:1] + 0.587 * x_nchw3[:, 1:2] + 0.114 * x_nchw3[:, 2:3])
    return x_nchw3


def train_reference_classifier(device, epochs=3, ckpt=Path("runs/.mnist_classifier.pt")) -> MNISTClassifier:
    from torchvision import datasets, transforms

    if ckpt.is_file():
        m = MNISTClassifier().to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        return m

    root = str(_REPO / "data" / "mnist")
    tf = transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = datasets.MNIST(root, train=True, download=False, transform=tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, num_workers=2)
    m = MNISTClassifier().to(device)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    m.train()
    for ep in range(epochs):
        tot = 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(m(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"  [classifier] epoch {ep+1}/{epochs} loss {tot/len(dl):.3f}")
    m.eval()
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), ckpt)
    return m


# ---------------------------------------------------------------- reference stats for FID
def compute_mnist_ref_stats(device, cache=Path("runs/.mnist_ref_stats.npz")) -> dict:
    if cache.is_file():
        d = np.load(cache)
        return {"mu": d["mu"], "sigma": d["sigma"]}
    from torchvision import datasets, transforms
    from utils.fid_util import _compute_stats

    root = str(_REPO / "data" / "mnist")
    tf = transforms.Compose([transforms.Resize(64), transforms.Grayscale(3), transforms.ToTensor()])
    ds = datasets.MNIST(root, train=False, download=False, transform=tf)
    imgs = []
    for i in range(min(10000, len(ds))):
        x, _ = ds[i]
        imgs.append((x.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    arr = np.stack(imgs)
    stats = _compute_stats(arr, len(arr), compute_logits=False, compute_features=False)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, mu=stats["mu"], sigma=stats["sigma"])
    return {"mu": stats["mu"], "sigma": stats["sigma"]}


# ---------------------------------------------------------------- sample generation
def load_model(run: Path, device):
    from models.generator import build_generator_from_config

    cfg = load_config(str(run / "log" / "config.yaml"))
    model_cfg = dict(cfg.model)
    model = build_generator_from_config(model_cfg, num_classes=int(cfg.dataset.num_classes))
    # prefer EMA weights
    ema = run / "params_ema" / "ema_params.pt"
    if ema.is_file():
        sd = torch.load(ema, map_location="cpu")
        model.load_state_dict(sd, strict=False)
    else:
        ckpts = sorted((run / "checkpoints").glob("step_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"no EMA or checkpoint under {run}")
        payload = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        sd = payload.get("ema_model") or payload.get("model")
        model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, cfg


def generate_samples(model, num: int, device, batch=256, cfg_scale=1.0, seed=0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (samples (N,3,H,W) in [0,1], labels (N,))."""
    torch.manual_seed(seed)
    n_cls = int(getattr(model, "num_classes", 10))
    out, labs = [], []
    done = 0
    while done < num:
        b = min(batch, num - done)
        labels = torch.arange(done, done + b, device=device) % n_cls
        with torch.inference_mode():
            s = model(c=labels, cfg_scale=cfg_scale)["samples"]  # NHWC
        s = s.permute(0, 3, 1, 2).contiguous()
        s = ((s + 1) / 2).clamp(0, 1)  # [-1,1] -> [0,1]
        out.append(s.cpu())
        labs.append(labels.cpu())
        done += b
    return torch.cat(out), torch.cat(labs)


# ---------------------------------------------------------------- scoring
@torch.no_grad()
def score(samples: torch.Tensor, labels: torch.Tensor, clf: MNISTClassifier, device) -> dict:
    preds, confs = [], []
    for i in range(0, len(samples), 256):
        x = _to_gray1(samples[i:i + 256]).to(device)
        logits = clf(x)
        prob = F.softmax(logits, dim=1)
        preds.append(prob.argmax(1).cpu())
        confs.append(prob.max(1).values.cpu())
    preds = torch.cat(preds)
    confs = torch.cat(confs)
    acc = (preds == labels).float().mean().item()
    return {
        "class_acc": float(acc),
        "mean_conf": float(confs.mean()),
        "per_class_acc": {
            int(c): float((preds[labels == c] == c).float().mean()) for c in range(10) if (labels == c).any()
        },
    }


def evaluate_run(run: Path, num: int, device, clf, ref, cfg_scale=1.0) -> dict:
    from utils.fid_util import _compute_stats
    from scipy import linalg

    def _fid(mu1, s1, mu2, s2):
        covmean = linalg.sqrtm(s1.dot(s2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(np.sum((mu1 - mu2) ** 2) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))

    model, cfg = load_model(run, device)
    samples, labels = generate_samples(model, num, device, cfg_scale=cfg_scale)
    res = score(samples, labels, clf, device)

    # FID vs real MNIST
    s_uint8 = (samples.permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)
    stats = _compute_stats(s_uint8, num, compute_logits=False, compute_features=False)
    res["fid"] = _fid(ref["mu"], ref["sigma"], stats["mu"], stats["sigma"])
    res["run"] = str(run)
    res["num_samples"] = num
    res["cfg_scale"] = cfg_scale

    # sample grid
    from PIL import Image

    grid = (torchvision_utils_make_grid(samples[:64]) * 255).byte().permute(1, 2, 0).numpy()
    out_img = run / "eval_samples.jpg"
    Image.fromarray(grid).save(out_img)
    res["sample_grid"] = str(out_img)
    return res


def torchvision_utils_make_grid(t: torch.Tensor, nrow=8) -> torch.Tensor:
    from torchvision.utils import make_grid

    return make_grid(t, nrow=nrow, padding=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--num", type=int, default=5000)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    print("training/loading reference MNIST classifier...")
    clf = train_reference_classifier(device)
    print("computing MNIST reference stats for FID...")
    ref = compute_mnist_ref_stats(device)

    all_res = []
    for r in args.run:
        print(f"\n=== {r} ===")
        res = evaluate_run(Path(r).resolve(), args.num, device, clf, ref, cfg_scale=args.cfg_scale)
        all_res.append(res)
        print(f"  class-acc: {res['class_acc']*100:.1f}%   FID: {res['fid']:.2f}   mean-conf: {res['mean_conf']:.3f}")
        print(f"  sample grid: {res['sample_grid']}")

    out = Path("runs/mnist_eval_summary.json")
    out.write_text(json.dumps(all_res, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
