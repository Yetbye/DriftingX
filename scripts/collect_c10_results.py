#!/usr/bin/env python
"""Collect final FID results from the CIFAR-10 experiment runs and print
paper-ready table rows.

Usage:
    python scripts/collect_c10_results.py
Reads runs/*/log/metrics.jsonl for the final (best) FID and prints markdown rows
for Tables 1 and 3 of paper/draft.md.
"""

from __future__ import annotations

import json
from pathlib import Path

RUNS = {
    "T1  ground=SD-VAE":   "C10_L1_gsdvae",
    "T1  ground=RAE":      "C10_L1_grae",
    "T1  ground=PAE":      "C10_L1_gpae",
    "T2  +f-loss":         "C10_L2_floss",
    "T2  +predict-cond":   "C10_L2_repcond",
    "T2  +both":           "C10_L2_both",
}


def final_fid(run: Path):
    mj = run / "log" / "metrics.jsonl"
    if not mj.is_file():
        return None
    best, last = None, None
    for line in mj.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "best_fid" in r and isinstance(r["best_fid"], (int, float)):
            best = r["best_fid"]
        if "fid" in r and isinstance(r["fid"], (int, float)):
            last = r["fid"]
        if "eval/CFG1.0_fid" in r:
            last = r["eval/CFG1.0_fid"]
    return {"best_fid": best, "last_fid": last}


def main():
    for label, name in RUNS.items():
        run = Path("runs") / name
        res = final_fid(run)
        if res is None:
            print(f"{label:28s} {name:22s} (no metrics)")
        else:
            bf = f"{res['best_fid']:.2f}" if res["best_fid"] is not None else "—"
            lf = f"{res['last_fid']:.2f}" if res["last_fid"] is not None else "—"
            print(f"{label:28s} {name:22s} best_FID={bf:>7s}  last_FID={lf:>7s}")


if __name__ == "__main__":
    main()
