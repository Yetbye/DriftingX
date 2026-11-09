#!/usr/bin/env python
"""Compare our CrossFlow backbone against the official C-Fun/cross-flow one.

    python scripts/compare_backbone.py [--official /path/to/cross-flow]

Prints parameter counts per size and a module-stack diff, so the paper can cite
an objective number. This is what motivates docs/BACKBONE-COMPARISON.md.
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def ours_report():
    from CrossFlow import CrossFlow

    rows = {}
    for size in ("B", "L", "XL"):
        m = CrossFlow(size=size, num_classes=1000)
        rows[size] = {
            "params": m.num_parameters(),
            "block": type(m.backbone.blocks[0]).__name__,
            "attn": type(m.backbone.blocks[0].attn).__name__,
            "mlp": type(m.backbone.blocks[0].mlp).__name__,
            "norm1": type(m.backbone.blocks[0].norm1).__name__,
            "has_rope": any("rope" in n.lower() for n, _ in m.named_parameters()),
            "has_swiglu": any("swiglu" in type(s).__name__.lower() for s in m.modules()),
            "has_vec_gate": any(hasattr(s, "attn_scale") for s in m.modules()),
            "has_qknorm": any(hasattr(s, "q_norm") for s in m.modules()),
            "pos_embed": "sincos(fixed)" if not isinstance(m.backbone.pos_embed, __import__("torch").nn.Parameter) else "learnable",
        }
    return rows


def official_report(official_dir: str):
    """Run the official repo in an isolated subprocess (our `models/` package
    shadows theirs, so the import must happen with only the official path on
    sys.path). Returns rows via JSON on stdout."""
    import json
    import subprocess

    probe = r"""
import json, sys
# Isolate from the caller's cwd: our repo has a `models/` package that shadows
# the official one. Only the official dir may be on sys.path.
sys.path = [p for p in sys.path if p not in ("", os.getcwd())] if False else sys.path
sys.path.insert(0, %r)
from models.crossflowDiT import crossflowDiT_B_2, crossflowDiT_L_2, crossflowDiT_XL_2
import torch
rows = {}
for name, fn in (("B", crossflowDiT_B_2), ("L", crossflowDiT_L_2), ("XL", crossflowDiT_XL_2)):
    m = fn()
    rows[name] = {
        "params": sum(p.numel() for p in m.parameters()),
        "block": type(m.blocks[0]).__name__,
        "attn": type(m.blocks[0].attn).__name__,
        "mlp": type(m.blocks[0].mlp).__name__,
        "norm1": type(m.blocks[0].norm1).__name__,
        "has_rope": any("rope" in n.lower() for n, _ in m.named_parameters()),
        "has_swiglu": any("swiglu" in type(s).__name__.lower() for s in m.modules()),
        "has_vec_gate": any(hasattr(s, "attn_scale") for s in m.modules()),
        "has_qknorm": any(hasattr(s, "q_norm") for s in m.modules()),
        "pos_embed": "learnable" if isinstance(m.pos_embed, torch.nn.Parameter) else "buffer",
    }
print(json.dumps(rows))
""" % os.path.abspath(official_dir)

    out = subprocess.run([sys.executable, "-P", "-c", probe], capture_output=True, text=True, timeout=300,
                         cwd=os.path.abspath(official_dir))
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[-800:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", default="/tmp/crossflow_official",
                    help="Path to a clone of C-Fun/cross-flow.")
    args = ap.parse_args()

    ours = ours_report()
    print("\n=== OURS (CrossFlow.py) ===")
    for k, v in ours.items():
        print(f"  {k:3s} {v['params']/1e6:7.1f}M  block={v['block']} attn={v['attn']} "
              f"mlp={v['mlp']} norm={v['norm1']} rope={v['has_rope']} swiglu={v['has_swiglu']} "
              f"gate={v['has_vec_gate']} qknorm={v['has_qknorm']} pe={v['pos_embed']}")

    if not os.path.isdir(args.official):
        print(f"\n[skip] official repo not found at {args.official}", file=sys.stderr)
        return 0

    try:
        off = official_report(args.official)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[skip] could not load official backbone: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 0

    print("\n=== OFFICIAL (crossflowDiT) ===")
    for k, v in off.items():
        print(f"  {k:3s} {v['params']/1e6:7.1f}M  block={v['block']} attn={v['attn']} "
              f"mlp={v['mlp']} norm={v['norm1']} rope={v['has_rope']} swiglu={v['has_swiglu']} "
              f"gate={v['has_vec_gate']} qknorm={v['has_qknorm']} pe={v['pos_embed']}")

    print("\n=== DELTA (ours - official) ===")
    for k in ours:
        o, f = ours[k], off[k]
        d = (o["params"] - f["params"]) / f["params"] * 100
        print(f"  {k:3s} {o['params']/1e6:7.1f}M vs {f['params']/1e6:7.1f}M  ({d:+.1f}%)")
    print("\nSee docs/BACKBONE-COMPARISON.md for the structural diff and the fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
