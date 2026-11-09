#!/usr/bin/env python
"""Roll up every run under a root into one comparison dashboard + index.

Use after a batch of jobs finishes, to compare experiments without a live UI.

    python scripts/analyze_all.py --root runs
    python scripts/analyze_all.py --root runs --pattern 'mnist_real_gen_*'

Writes:
    <root>/comparison.png   loss curves (overlaid) + best-FID bar chart
    <root>/index.md         table of all runs: steps, final loss, best FID/cfg
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

from analyze_run import load_metrics, load_yaml, series  # type: ignore  # noqa: E402


def collect(root: Path, pattern: str, include_symlinks: bool, include_smoke: bool) -> list[dict]:
    runs = []
    for p in sorted(root.glob(pattern)):
        if not p.is_dir():
            continue
        if p.is_symlink() and not include_symlinks:
            continue
        recs = load_metrics(p)
        if not recs:
            continue
        name = p.name
        if not include_smoke and ("smoke" in name or "_cf" in name):
            continue
        cfg = load_yaml(p / "log" / "config.yaml")
        model = (cfg.get("model") or {})
        last = recs[-1]
        loss_key = "loss" if any(isinstance(r.get("loss"), (int, float)) for r in recs) else "recon_loss"
        _, losses = series(recs, loss_key)
        best_fid, best_cfg = None, None
        _, bf = series(recs, "best_fid")
        if bf:
            best_fid = min(bf)
        _, bc = series(recs, "best_cfg")
        if bc:
            best_cfg = bc[-1]
        runs.append({
            "path": p, "name": name, "recs": recs, "cfg": cfg,
            "arch": model.get("arch", "dit"),
            "size": model.get("size") or f"h{model.get('hidden_size','?')}d{model.get('depth','?')}",
            "spatial": "latent" if model.get("in_channels") == 4 else "pixel",
            "steps": int(last.get("step", -1)) + 1,
            "final_loss": losses[-1] if losses else None,
            "loss_key": loss_key,
            "best_fid": best_fid, "best_cfg": best_cfg,
        })
    return runs


def render_comparison(runs: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for r in runs:
        xs, ys = series(r["recs"], r["loss_key"])
        if xs:
            ax.plot(xs, ys, lw=1.3, label=f"{r['name']} ({r['loss_key']})")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("drift loss comparison")
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)

    ax = axes[1]
    with_fid = [r for r in runs if r["best_fid"] is not None]
    if with_fid:
        names = [r["name"] for r in with_fid]
        vals = [r["best_fid"] for r in with_fid]
        ax.barh(range(len(vals)), vals, color="C2")
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("best FID (lower is better)")
        ax.set_title("best FID per run")
    else:
        ax.text(0.5, 0.5, "no run logged FID", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def render_index(runs: list[dict], out_path: Path) -> None:
    lines = ["# Run index\n",
             f"({len(runs)} runs)\n",
             "| run | arch | size | space | steps | final loss | best FID | best cfg |",
             "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        fl = f"{r['final_loss']:.2f}" if r["final_loss"] is not None else ""
        bf = f"{r['best_fid']:.3f}" if r["best_fid"] is not None else ""
        bc = f"{r['best_cfg']:.1f}" if r["best_cfg"] is not None else ""
        lines.append(f"| `{r['name']}` | {r['arch']} | {r['size']} | {r['spatial']} | {r['steps']} | {fl} | {bf} | {bc} |")
    lines.append("")
    lines.append("Per-run reports: `python scripts/analyze_run.py <run_dir>`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--include-symlinks", action="store_true")
    ap.add_argument("--include-smoke", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"root not found: {root}", file=sys.stderr)
        return 1
    runs = collect(root, args.pattern, args.include_symlinks, args.include_smoke)
    if not runs:
        print(f"no runs with metrics under {root}", file=sys.stderr)
        return 1

    render_comparison(runs, root / "comparison.png")
    render_index(runs, root / "index.md")
    print(f"wrote {root / 'comparison.png'}")
    print(f"wrote {root / 'index.md'}")
    print(f"{len(runs)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
