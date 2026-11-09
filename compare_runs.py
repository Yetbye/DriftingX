"""Side-by-side comparison of several Drift/CrossFlow training runs.

Produces one figure with a shared loss panel plus one sample-grid panel per run,
so the three MNIST paths (latent / pixel / crossflow) can be compared directly.

Usage:
    python scripts/compare_runs.py runs/a_latest runs/b_latest runs/c_latest \
        --labels latent pixel crossflow --out runs/compare.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from PIL import Image  # noqa: E402


def _load_metrics(run: Path) -> list[dict]:
    p = run / "log" / "metrics.jsonl"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _series(records: list[dict], key: str):
    xs, ys = [], []
    for i, r in enumerate(records):
        v = r.get(key)
        if isinstance(v, (int, float)):
            xs.append(r.get("step", i))
            ys.append(v)
    return xs, ys


def _last_grid(run: Path):
    d = run / "log" / "images"
    if not d.is_dir():
        return None, None
    files = sorted(list(d.glob("*.jpg")) + list(d.glob("*.png")))
    if not files:
        return None, None
    p = files[-1]
    return p.name, Image.open(p).convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="Run directories (or *_latest symlinks).")
    ap.add_argument("--labels", nargs="*", default=None, help="Labels for each run.")
    ap.add_argument("--out", default="runs/compare.png")
    ap.add_argument("--loss-key", default="loss")
    args = ap.parse_args()

    runs = [Path(r).resolve() for r in args.runs]
    labels = args.labels or [r.name for r in runs]
    if len(labels) != len(runs):
        raise SystemExit("--labels must match the number of runs")

    metrics = [_load_metrics(r) for r in runs]
    grids = [_last_grid(r) for r in runs]
    has_loss = any(m for m in metrics)
    has_grid = any(g[1] is not None for g in grids)

    nrows = int(has_loss) + int(has_grid)
    fig = plt.figure(figsize=(4.2 * len(runs), 4.4 * max(nrows, 1)))
    gs = GridSpec(nrows, len(runs), figure=fig, hspace=0.35, wspace=0.25)

    row = 0
    if has_loss:
        ax = fig.add_subplot(gs[row, :]); row += 1
        for recs, lab in zip(metrics, labels):
            xs, ys = _series(recs, args.loss_key)
            if xs:
                ax.plot(xs, ys, label=f"{lab} (final {ys[-1]:.1f})", linewidth=1.6)
        ax.set_xlabel("step")
        ax.set_ylabel(args.loss_key)
        ax.set_title("drift loss comparison")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    if has_grid:
        for j, ((name, img), lab, recs) in enumerate(zip(grids, labels, metrics)):
            ax = fig.add_subplot(gs[row, j])
            ax.set_xticks([]); ax.set_yticks([])
            if img is None:
                ax.text(0.5, 0.5, "no samples", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(lab, fontsize=10)
                continue
            ax.imshow(img)
            step = recs[-1].get("step") if recs else "?"
            ax.set_title(f"{lab} — samples @ step {step}", fontsize=10)
            ax.text(0.01, 0.02, name, transform=ax.transAxes, fontsize=6,
                    color="white", bbox=dict(facecolor="black", alpha=0.5))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
