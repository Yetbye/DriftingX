#!/usr/bin/env bash
# Entry point used inside the container (and usable directly on a host).
#
# Responsibilities:
#   1. wire the offline asset bundle (one variable -> all paths)
#   2. sanity-check the dataset and bundle are actually visible
#   3. launch training with the right distributed env, forwarding all args
#
# Usage inside the container:
#   drift-train --gen --config configs/gen/crossflow_sota_B.yaml --workdir runs/cf_B
#
# Env (set by docker run / sbatch):
#   DRIFT_OFFLINE_BUNDLE  offline bundle root (see docs/CLUSTER-GUIDE.md)
#   IMAGENET_PATH         ImageFolder root with train/ and val/
#   IMAGENET_CACHE_PATH   cached latents root (only for use_cache configs)
#   NPROC_PER_NODE        GPUs per node (default: visible device count)
#   DRIFT_DRY_CHECK=1     print torch/paths and exit before training
#
# Resume: rerun with the same --workdir and pass --resume (add --no-stamp on the
# first run so the directory is not timestamped). The trainer then restores the
# latest checkpoint from <workdir>/checkpoints automatically.
set -euo pipefail

# --- 0. locate the repo ------------------------------------------------------
REPO_ROOT="${DRIFT_REPO_ROOT:-/workspace/drifting}"
cd "$REPO_ROOT"

PY="${DRIFT_PYTHON:-python}"

# --- 1. offline asset bundle -------------------------------------------------
if [[ -n "${DRIFT_OFFLINE_BUNDLE:-}" ]]; then
  export HF_ROOT="${HF_ROOT:-${DRIFT_OFFLINE_BUNDLE}/hf_root}"
  export HF_HOME="${HF_HOME:-${DRIFT_OFFLINE_BUNDLE}/hf_root}"
  export SDVAE_PATH="${SDVAE_PATH:-${DRIFT_OFFLINE_BUNDLE}/sdvae}"
  export DRIFT_CONVNEXT_PATH="${DRIFT_CONVNEXT_PATH:-${DRIFT_OFFLINE_BUNDLE}/hf_root/convnext/convnextv2-base-22k-224}"
  export IMAGENET_FID_NPZ="${IMAGENET_FID_NPZ:-${DRIFT_OFFLINE_BUNDLE}/fid/imagenet_256_fid_stats.npz}"
  export IMAGENET_PR_NPZ="${IMAGENET_PR_NPZ:-${DRIFT_OFFLINE_BUNDLE}/imnet_stats/imagenet_val_prc_arr0.npz}"
  # pytorch-fid / torch-fidelity read Inception weights from the torch hub cache
  export TORCH_HOME="${TORCH_HOME:-${DRIFT_OFFLINE_BUNDLE}/torch}"
fi
# Never let a stray download hang an offline job.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${REPO_ROOT}/.cache/torchinductor}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

if [[ "${DRIFT_DRY_CHECK:-0}" == "1" ]]; then
  echo "=== environment check ==="
  "$PY" - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
import torch
print("torch", torch.__version__, "cuda_build", torch.version.cuda,
      "cuda_available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
try:
    import utils.env as env
    for k in ("IMAGENET_PATH", "IMAGENET_CACHE_PATH", "HF_ROOT", "SDVAE_PATH",
              "DRIFT_CONVNEXT_PATH", "IMAGENET_FID_NPZ", "TORCH_HOME"):
        print(f"  {k} = {os.environ.get(k)}")
except Exception as e:
    print("env import failed:", e)
PY
fi

# --- 2. dataset sanity -------------------------------------------------------
if [[ -n "${IMAGENET_PATH:-}" ]]; then
  for split in train val; do
    if [[ ! -d "${IMAGENET_PATH}/${split}" ]]; then
      echo "ERROR: ${IMAGENET_PATH}/${split} not found. Mount ImageNet at IMAGENET_PATH." >&2
      exit 3
    fi
  done
  echo "[data] IMAGENET_PATH=${IMAGENET_PATH}"
else
  echo "WARNING: IMAGENET_PATH is unset; ImageNet configs will fail." >&2
fi
if [[ -n "${DRIFT_OFFLINE_BUNDLE:-}" && ! -f "${SDVAE_PATH:-}/config.json" ]]; then
  echo "WARNING: SD-VAE not found at ${SDVAE_PATH:-<unset>} (needed for latent configs)." >&2
fi

# --- 3. launch ---------------------------------------------------------------
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)}"
  export MASTER_PORT="${MASTER_PORT:-54123}"
fi

NPROC="${NPROC_PER_NODE:-}"
if [[ -z "$NPROC" ]]; then
  NPROC="$("$PY" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)"
  [[ "$NPROC" == "0" ]] && NPROC=1
fi
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-0}}"

echo "[launch] nproc_per_node=$NPROC nodes=$NNODES node_rank=$NODE_RANK"
echo "[launch] args: $*"

if [[ "$NPROC" == "1" && "$NNODES" == "1" ]]; then
  exec "$PY" -u "$@"
fi

# `torchrun_jsc` is the cluster-friendly launcher shipped in the env; fall back
# to plain torchrun if it is unavailable.
if "$PY" -c "import torchrun_jsc" 2>/dev/null; then
  LAUNCHER=(-m torchrun_jsc)
else
  LAUNCHER=(-m torch.distributed.run)
fi
exec "$PY" -u "${LAUNCHER[@]}" \
  --nnodes="$NNODES" \
  --nproc-per-node="$NPROC" \
  --node-rank="$NODE_RANK" \
  --master-addr="${MASTER_ADDR:-127.0.0.1}" \
  --master-port="${MASTER_PORT:-54123}" \
  "$@"
