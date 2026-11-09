#!/usr/bin/env python
"""Download the VA-VAE encoder used by CrossFlow (f16, 32-ch, DINOv2-trained).

Source: hustvl/vavae-imagenet256-f16d32-dinov2 on the HuggingFace hub
(same repo the official C-Fun/cross-flow code uses; see utils/vae_util.py there).

Writes <dest>/vavae/vavae-imagenet256-f16d32-dinov2.pt so the offline bundle can
be wired via DRIFT_VAVAE_PATH.
"""
import argparse, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = "hustvl/vavae-imagenet256-f16d32-dinov2"
FILE = "vavae-imagenet256-f16d32-dinov2.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="/mnt/afs/shawn/offline_bundle")
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download

    out_dir = Path(args.dest) / "vavae"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / FILE
    if target.is_file() and target.stat().st_size > 100_000_000:
        print(f"[vavae] already present: {target} ({target.stat().st_size/1e6:.0f} MB)")
        return 0
    print(f"[vavae] {REPO}/{FILE} -> {target}", flush=True)
    t0 = time.time()
    p = hf_hub_download(REPO, FILE, local_dir=str(out_dir))
    tmp = Path(p)
    if tmp != target:
        tmp.replace(target)
    print(f"[vavae] done {target.stat().st_size/1e6:.0f} MB in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
