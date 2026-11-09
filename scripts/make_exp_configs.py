#!/usr/bin/env python
"""Generate all CIFAR-10 experiment configs (the full paper matrix).

Produces configs/exp/*.yaml covering the three experimental layers described in
paper/draft.md:

  Layer 1  noise prior:       P-P vs L-P (X-Drifting)
  Layer 2  grounding encoder: SD-VAE / RAE / PAE (representation ladder)
  Layer 3  mechanisms:        f-loss / predict-then-condition ablations

Run:  python scripts/make_exp_configs.py
Then: bash scripts/run_all_experiments.sh
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

OUT = Path("configs/exp")


def base_dataset():
    return {
        "name": "cifar10", "resolution": 64, "use_aug": True,
        "use_latent": False, "use_cache": False, "num_classes": 10,
        "batch_size": 32, "eval_batch_size": 32,
        "kwargs": {"num_workers": 2, "pin_memory": True, "prefetch_factor": 2},
    }


def optimizer_cfg():
    return {
        "lr_schedule": {"learning_rate": 0.0004, "warmup_steps": 500,
                         "lr_schedule": "const", "total_steps": 20000},
        "weight_decay": 0.0, "adam_b1": 0.9, "adam_b2": 0.95,
    }


def train_cfg(*, freq=0.0, rep_align=0.0, rep_condition=False):
    t = {
        "objective": "drift", "train_batch_size": 0, "seed": 42,
        "total_steps": 20000, "save_per_step": 2000, "keep_last": 2,
        "eval_per_step": 2500, "eval_samples": 10000,
        "pos_per_sample": 8, "neg_per_sample": 4,
        "positive_bank_size": 32, "negative_bank_size": 128,
        "forward_dict": {"gen_per_label": 8, "cfg_min": 1.0, "cfg_max": 4.0,
                          "neg_cfg_pw": 5.0, "no_cfg_frac": 0.0},
        "activation_kwargs": {"patch_mean_size": [2, 4], "patch_std_size": [2, 4],
                              "use_std": True, "use_mean": True, "every_k_block": 2},
        "loss_kwargs": {"R_list": [0.2, 0.05, 0.02]},
        "cfg_list": [1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5],
        "ema_decay": 0.999, "push_per_step": 64, "push_at_resume": 1,
        "eval_on_step_one": False, "run_eval": True,
        "train_sample_per_step": 2000, "train_sample_grid_size": 36,
        "train_sample_cfg": 2.0, "train_sample_seed": 1234,
    }
    if freq and freq > 0:
        t.update({"freq_weight": freq, "freq_weight_start": 0.8,
                  "freq_weight_end": 0.0, "freq_transition_frac": 0.4,
                  "freq_weight_high": 1.0})
    if rep_align and rep_align > 0:
        t.update({"rep_align_weight": rep_align, "rep_condition": rep_condition})
    return t


def ditgen_pixel():
    return {"cond_dim": 512, "input_size": 64, "in_channels": 3, "patch_size": 8,
            "hidden_size": 512, "depth": 8, "num_heads": 8, "mlp_ratio": 4.0,
            "out_channels": 3, "use_qknorm": True, "use_swiglu": True,
            "use_rope": True, "use_rmsnorm": True, "n_cls_tokens": 8,
            "noise_classes": 64, "noise_coords": 32}


def crossflow_latent_pixel():
    return {"arch": "crossflow", "img_size": 64, "patch_size": 8,
            "latent_channels": 4, "latent_size": 8, "latent_patch_size": 1,
            "hidden_size": 512, "depth": 8, "num_heads": 8, "mlp_ratio": 8 / 3,
            "num_class_tokens": 8, "num_time_tokens": 4, "output_activation": "none"}


def feature_spec(kind="mae", mae="runs/cifar_mae_pixel_latest", grounding=None):
    """kind: 'mae' (pixel-MAE baseline) or 'grounding' (frozen encoder)."""
    if kind == "grounding":
        return {"mae_path": mae, "use_mae": False, "use_convnext": False,
                "grounding": grounding or {"kind": "sdvae"}}
    return {"mae_path": mae, "use_mae": True, "use_convnext": False}


def write(name, model, train, feature):
    cfg = {"hsdp_dim": 1, "compile": 0,
           "logging": {"use_wandb": False, "log_every_k": 20},
           "dataset": base_dataset(), "model": model,
           "optimizer": optimizer_cfg(), "train": train, "feature": feature}
    p = OUT / f"{name}.yaml"
    yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
    print(f"wrote {p}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- Layer 1 (grounding encoder ladder) ----
    # NOTE: Layer "noise prior" (P-P vs L-P) is dropped — the input is just
    # different-sized noise; the representation prior (grounding) is what matters.
    for gk in ["sdvae", "rae", "pae"]:
        write(f"C10_L1_g{gk}", crossflow_latent_pixel(), train_cfg(),
              feature_spec("grounding", grounding={"kind": gk}))

    # ---- mechanism ablations ----
    write("C10_L2_floss", crossflow_latent_pixel(), train_cfg(freq=0.5),
          feature_spec("mae"))
    write("C10_L2_repcond", crossflow_latent_pixel(),
          train_cfg(rep_align=0.5, rep_condition=True),
          feature_spec("grounding", grounding={"kind": "sdvae"}))
    write("C10_L2_both", crossflow_latent_pixel(),
          train_cfg(freq=0.5, rep_align=0.5, rep_condition=True),
          feature_spec("grounding", grounding={"kind": "sdvae"}))

    print("\nAll experiment configs written to", OUT)


if __name__ == "__main__":
    main()
