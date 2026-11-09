#!/usr/bin/env python
"""Full offline analysis of one training run -> a single PNG + a markdown report.

The cluster has no interactive terminal and runs are not watched live, so every
piece of information must be recoverable afterwards from the run directory.
This script reads what training wrote and renders it:

  <run>/log/metrics.jsonl   scalar curves (drift loss, per-temperature, FID, IS, throughput)
  <run>/log/config.yaml     resolved hyperparameters
  <run>/log/run.json        host/slurm/env snapshot
  <run>/log/images/*.jpg    sample grids written every train_sample_per_step

Outputs (into <run>/log/):
  report.png        multi-panel figure: loss, FID/IS, throughput, sample evolution
  report.md         human-readable summary with the final/best numbers
  samples_evolution.png   montage of sample grids across training

Usage:
    python scripts/analyze_run.py runs/<stamp>_<name> [--out-dir DIR]
    python scripts/analyze_run.py --latest
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from PIL import Image  # noqa: E402


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def load_metrics(run: Path) -> list[dict]:
    p = run / "log" / "metrics.jsonl"
    if not p.is_file():
        return []
    recs = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return recs


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        return {}
    if not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}


def series(recs: list[dict], key: str) -> tuple[list, list]:
    xs, ys = [], []
    for i, r in enumerate(recs):
        v = r.get(key)
        if isinstance(v, (int, float)) and v == v:  # skip NaN
            xs.append(r.get("step", i))
            ys.append(v)
    return xs, ys


def find_keys(recs: list[dict], pred) -> list[str]:
    keys: list[str] = []
    for r in recs:
        for k in r:
            if k not in keys and pred(k, r):
                keys.append(k)
    return keys


# ----------------------------------------------------------------------------
# panels
# ----------------------------------------------------------------------------
def panel_loss(ax, recs: list[dict], title: str) -> None:
    # total loss
    xs, ys = series(recs, "loss")
    if not xs and not ys:
        # drift runs log per-feature losses instead of a scalar 'loss'
        xs, ys = series(recs, "recon_loss")
    if xs:
        ax.plot(xs, ys, label="loss", lw=1.6, color="C0")
    # per-temperature drift losses (loss_0.02/..., loss_0.2/global)
    temp_keys = find_keys(recs, lambda k, r: k.startswith("loss_") and isinstance(r.get(k), (int, float)))
    # aggregate to the '/global' variant when present to avoid clutter
    glob = [k for k in temp_keys if k.endswith("/global")]
    chosen = glob or [k for k in temp_keys if "/" not in k][:6]
    for k in chosen:
        xs, ys = series(recs, k)
        if xs:
            ax.plot(xs, ys, lw=0.9, alpha=0.8, label=k)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, ncol=2)


def panel_fid(ax, recs: list[dict], title: str) -> dict:
    """Plot FID/IS curves; return the best-FID summary."""
    fid_keys = find_keys(recs, lambda k, r: k.endswith("_fid") and isinstance(r.get(k), (int, float)))
    isc_keys = find_keys(recs, lambda k, r: k.endswith("_isc_mean") and isinstance(r.get(k), (int, float)))
    best = {}
    for k in fid_keys:
        xs, ys = series(recs, k)
        if xs:
            ax.plot(xs, ys, marker="o", ms=3, lw=1.2, label=k)
    for k in isc_keys:
        xs, ys = series(recs, k)
        if xs:
            ax2 = ax.twinx()
            ax2.plot(xs, ys, ls="--", lw=1.0, alpha=0.7, label=k)
            ax2.set_ylabel("IS")
            break
    # best_fid / best_cfg recorded by the trainer
    bf_x, bf_y = series(recs, "best_fid")
    if bf_x:
        ax.plot(bf_x, bf_y, marker="*", ms=8, lw=0, color="crimson", label="best_fid")
        best["best_fid"] = bf_y[-1]
        xs_c, ys_c = series(recs, "best_cfg")
        if ys_c:
            best["best_cfg"] = ys_c[-1]
    if not fid_keys:
        ax.text(0.5, 0.5, "no FID logged (run_eval off or not reached)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("step")
    ax.set_ylabel("FID")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, loc="upper right")
    return best


def panel_throughput(ax, recs: list[dict], title: str) -> dict:
    info = {}
    xs, ys = series(recs, "total_time")
    if xs:
        ax.plot(xs, ys, lw=1.1, color="C3", label="sec/step")
        ax.set_ylabel("sec / step")
        info["sec_per_step_last"] = ys[-1]
        info["sec_per_step_mean"] = sum(ys) / len(ys)
        if len(xs) > 1 and xs[-1] > xs[0]:
            per = (xs[-1] - xs[0]) / (len(xs) - 1)
            info["steps_per_sec"] = 1.0 / per if per else float("nan")
    kx, ky = series(recs, "kimg")
    if kx:
        info["kimg_last"] = ky[-1]
    ax.set_xlabel("step")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    return info


def _load_grids(images_dir: Path) -> list[tuple[int, str, Image.Image]]:
    out = []
    for p in sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))):
        m = re.search(r"step(\d+)", p.stem)
        step = int(m.group(1)) if m else -1
        try:
            out.append((step, p.name, Image.open(p).convert("RGB")))
        except Exception:  # noqa: BLE001
            continue
    return sorted(out, key=lambda t: t[0])


def build_evolution(run: Path, out_path: Path, max_cols: int = 6) -> bool:
    """Montage of sample grids sampled across training (the 'did it learn' view)."""
    grids = _load_grids(run / "log" / "images")
    if not grids:
        return False
    # subsample to at most max_cols columns
    if len(grids) > max_cols:
        idx = [round(i * (len(grids) - 1) / (max_cols - 1)) for i in range(max_cols)]
        grids = [grids[i] for i in sorted(set(idx))]

    base_h = 180
    thumbs = []
    for step, name, img in grids:
        w = max(1, int(img.width * base_h / img.height))
        thumbs.append((step, name, img.resize((w, base_h))))
    gap = 8
    total_w = sum(t[2].width for t in thumbs) + gap * (len(thumbs) - 1)
    canvas = Image.new("RGB", (total_w, base_h + 26), "white")
    x = 0
    for step, name, im in thumbs:
        canvas.paste(im, (x, 22))
        x += im.width + gap
    from PIL import ImageDraw

    d = ImageDraw.Draw(canvas)
    x = 0
    for step, name, im in thumbs:
        d.text((x + 4, 4), f"step {step}", fill="black")
        x += im.width + gap
    canvas.save(out_path)
    return True


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def write_markdown(run: Path, recs: list[dict], cfg: dict, run_meta: dict, best: dict,
                   perf: dict, out_path: Path, evolution: Path | None) -> None:
    lines: list[str] = []
    lines.append(f"# Run report: `{run.name}`\n")
    lines.append(f"- path: `{run}`")
    if run_meta:
        lines.append(f"- created: {run_meta.get('created_at', '?')}  host: {run_meta.get('hostname', '?')}")
        env = run_meta.get("env", {})
        if env:
            lines.append(f"- slurm/env: `{env}`")
    if recs:
        steps = [r.get("step") for r in recs if isinstance(r.get("step"), (int, float))]
        if steps:
            lines.append(f"- steps logged: {len(recs)} (step {int(min(steps))} .. {int(max(steps))})")
    lines.append("")

    lines.append("## Final metrics\n")
    last = recs[-1] if recs else {}
    for k in ("loss", "recon_loss", "cls_loss", "g_norm", "lr", "kimg"):
        if isinstance(last.get(k), (int, float)):
            lines.append(f"- {k}: {_fmt(last[k])}")
    for k, v in best.items():
        lines.append(f"- {k}: {_fmt(v)}")
    for k in ("sec_per_step_mean", "sec_per_step_last", "steps_per_sec"):
        if isinstance(perf.get(k), (int, float)):
            lines.append(f"- {k}: {_fmt(perf[k])}")
    lines.append("")

    if cfg:
        lines.append("## Configuration\n")
        for section in ("model", "dataset", "optimizer", "train", "feature"):
            sec = cfg.get(section)
            if isinstance(sec, dict):
                lines.append(f"### {section}\n")
                for k, v in sec.items():
                    lines.append(f"- {k}: {v}")
                lines.append("")

    # FID table
    fid_keys = find_keys(recs, lambda k, r: k.endswith("_fid") and isinstance(r.get(k), (int, float)))
    if fid_keys:
        lines.append("## FID / IS history\n")
        lines.append("| step | " + " | ".join(fid_keys) + " |")
        lines.append("|---" * (len(fid_keys) + 1) + "|")
        for r in recs:
            vals = [r.get(k) for k in fid_keys]
            if any(isinstance(v, (int, float)) for v in vals):
                lines.append("| " + str(int(r.get("step", -1))) + " | "
                             + " | ".join(_fmt(v) if isinstance(v, (int, float)) else "" for v in vals) + " |")
        lines.append("")

    lines.append("## Figures\n")
    lines.append("- `report.png` — loss / FID / throughput panels")
    if evolution:
        lines.append(f"- `{evolution.name}` — sample grids across training")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_run(run: Path, out_dir: Path | None = None) -> int:
    if not run.is_dir():
        print(f"run dir not found: {run}", file=sys.stderr)
        return 1
    log_dir = run / "log"
    out_dir = out_dir or log_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = load_metrics(run)
    cfg = load_yaml(log_dir / "config.yaml")
    run_meta = {}
    rj = log_dir / "run.json"
    if rj.is_file():
        try:
            run_meta = json.loads(rj.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            run_meta = {}

    # sample evolution
    evo_path = out_dir / "samples_evolution.png"
    has_evo = build_evolution(run, evo_path)

    fig = plt.figure(figsize=(13, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)
    best = panel_fid(fig.add_subplot(gs[0, 0]), recs, "FID / IS vs step")
    panel_loss(fig.add_subplot(gs[0, 1]), recs, "loss vs step")
    perf = panel_throughput(fig.add_subplot(gs[1, 0]), recs, "throughput")
    ax = fig.add_subplot(gs[1, 1])
    if has_evo:
        ax.imshow(Image.open(evo_path))
    else:
        ax.text(0.5, 0.5, "no sample grids saved", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("samples across training", fontsize=10)
    fig.suptitle(f"{run.name}", fontsize=12)
    fig.savefig(out_dir / "report.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    write_markdown(run, recs, cfg, run_meta, best, perf, out_dir / "report.md",
                   evo_path if has_evo else None)
    print(f"wrote {out_dir / 'report.png'}")
    print(f"wrote {out_dir / 'report.md'}")
    if has_evo:
        print(f"wrote {evo_path}")
    return 0


def find_latest(root: Path) -> Path | None:
    cands = [p for p in root.glob("*") if p.is_dir() and not p.is_symlink()]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", help="Run directory.")
    ap.add_argument("--latest", action="store_true", help="Analyze the newest run under --root.")
    ap.add_argument("--root", default="runs", help="Root searched by --latest.")
    ap.add_argument("--out-dir", default=None, help="Where to write report.png/.md (default: <run>/log).")
    args = ap.parse_args()

    if args.latest or not args.run_dir:
        run = find_latest(Path(args.root))
        if run is None:
            print(f"no runs under {args.root}", file=sys.stderr)
            return 1
    else:
        run = Path(args.run_dir).expanduser().resolve()
    return analyze_run(run, Path(args.out_dir).resolve() if args.out_dir else None)


if __name__ == "__main__":
    raise SystemExit(main())
