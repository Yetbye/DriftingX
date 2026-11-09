"""Visualize a Drift training run: loss curves + collected sample grids.

Reads the local tracking files a run writes (log/metrics.jsonl, log/images/*.jpg)
and produces a single overview PNG. No WandB needed.

Usage:
    python scripts/visualize_run.py [run_dir] [--out out.png]

run_dir defaults to the newest runs/* directory. The overview is written to
<run_dir>/log/overview.png (or --out).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from PIL import Image  # noqa: E402


def _load_metrics(metrics_path: Path) -> list[dict]:
    records = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _metric_name(key: str) -> str:
    # drop per-feature scale suffixes for grouping; keep them in the curve legend though
    return key


def _is_numeric_series(records: list[dict], key: str) -> bool:
    vals = [r[key] for r in records if isinstance(r.get(key), (int, float))]
    return len(vals) >= 2


def _loss_keys(records: list[dict]) -> list[str]:
    """Prefer scalar 'loss'/'recon_loss', then per-feature drift losses aggregated."""
    for candidate in ("loss", "recon_loss", "cls_loss"):
        if _is_numeric_series(records, candidate):
            return [candidate]
    keys = []
    for k in records[0]:
        if k.endswith("/global") and _is_numeric_series(records, k):
            keys.append(k)
    if not keys:
        keys = [k for k in records[0] if _is_numeric_series(records, k) and k != "step"]
    return keys[:12]


def _plot_curves(ax, records: list[dict], keys: list[str], title: str) -> None:
    x = [r.get("step", i) for i, r in enumerate(records)]
    for key in keys:
        y = [r.get(key) for r in records]
        # drop None for plotting (log_every_k may skip some keys)
        xs = [xv for xv, yv in zip(x, y) if isinstance(yv, (int, float))]
        ys = [yv for yv in y if isinstance(yv, (int, float))]
        ax.plot(xs, ys, label=key, linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="best", ncol=1)
    ax.grid(True, alpha=0.3)


def _load_images(images_dir: Path) -> list[tuple[str, Image.Image]]:
    out = []
    for p in sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png")):
        try:
            out.append((p.name, Image.open(p).convert("RGB")))
        except Exception:
            continue
    return out


def _plot_image_grids(ax, images: list[tuple[str, Image.Image]], title: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    if not images:
        ax.text(0.5, 0.5, "no sample grids yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        return
    # show up to 4 grids side by side, each scaled to a fixed height
    n = min(len(images), 4)
    ims = [img for _, img in images[-n:]]
    heights = [im.height for im in ims]
    widths = [im.width for im in ims]
    total_w = sum(widths)
    max_h = max(heights)
    canvas = Image.new("RGB", (total_w, max_h), "white")
    x = 0
    for im in ims:
        canvas.paste(im, (x, 0))
        x += im.width
    ax.imshow(canvas)
    ax.set_title(title, fontsize=10)
    # annotate with the step of the last grid
    ax.text(0.01, 0.02, images[-1][0], transform=ax.transAxes, fontsize=7,
            color="white", bbox=dict(facecolor="black", alpha=0.5))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", default=None, help="Run directory (default: newest runs/*)")
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()

    root = Path(args.run_dir).resolve() if args.run_dir else None
    if root is None:
        runs = sorted(Path("runs").glob("*_*/"), key=lambda p: p.stat().st_mtime)
        runs = [p for p in runs if p.is_dir() and not p.is_symlink()]
        if not runs:
            print("no runs found", file=sys.stderr)
            return 1
        root = runs[-1]
    if not root.is_dir():
        print(f"run dir not found: {root}", file=sys.stderr)
        return 1

    metrics_path = root / "log" / "metrics.jsonl"
    images_dir = root / "log" / "images"
    records = _load_metrics(metrics_path) if metrics_path.is_file() else []
    images = _load_images(images_dir) if images_dir.is_dir() else []

    has_metrics = bool(records)
    has_images = bool(images)

    nrows = int(has_metrics) + int(has_images)
    if nrows == 0:
        print(f"nothing to visualize in {root}", file=sys.stderr)
        return 1

    fig = plt.figure(figsize=(12, 4.2 * nrows))
    gs = GridSpec(nrows, 1, figure=fig, hspace=0.45)
    row = 0

    if has_metrics:
        ax = fig.add_subplot(gs[row]); row += 1
        keys = _loss_keys(records)
        _plot_curves(ax, records, keys, f"metrics — {root.name}")

    if has_images:
        ax = fig.add_subplot(gs[row]); row += 1
        _plot_image_grids(ax, images, f"samples — {root.name}")

    out = Path(args.out) if args.out else root / "log" / "overview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
