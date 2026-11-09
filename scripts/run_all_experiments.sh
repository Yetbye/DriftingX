#!/usr/bin/env bash
# One-shot: run ALL CIFAR-10 paper experiments on the offline cluster.
#
#   bash scripts/run_all_experiments.sh [--gpu-start N] [--only LAYER]
#
# Layers (see paper/draft.md §4):
#   L1  grounding encoder: SD-VAE / RAE / PAE (representation ladder)
#   L2  mechanisms:        f-loss / predict-then-condition ablations
# (Layer "noise prior" P-P vs L-P is dropped: input is just different-sized noise.)
#
# Prereqs (offline bundle + CIFAR-10 data must be in place):
#   DRIFT_OFFLINE_BUNDLE  offline asset bundle (SD-VAE, MAE, ConvNeXt, FID)
#   data/cifar10          CIFAR-10 (already downloaded)
#
# Each run writes runs/<name>/ (checkpoints, log/metrics.jsonl, log/images,
# params_ema). Reports are generated automatically at exit. FID is evaluated at
# eval_per_step and at the end.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/home/lxw/miniconda3/envs/drifting/bin/python}
GPU_START=${GPU_START:-3}
ONLY=${ONLY:-all}

export DRIFT_OFFLINE_BUNDLE=${DRIFT_OFFLINE_BUNDLE:-/mnt/afs/shawn/offline_bundle}
export HF_HUB_OFFLINE=1
export SDVAE_PATH=${SDVAE_PATH:-$DRIFT_OFFLINE_BUNDLE/sdvae}
export TORCH_HOME=${TORCH_HOME:-$DRIFT_OFFLINE_BUNDLE/torch}
export HF_ROOT=${HF_ROOT:-$DRIFT_OFFLINE_BUNDLE/hf_root}

mkdir -p runs logs
echo "offline bundle: $DRIFT_OFFLINE_BUNDLE"

# ---- 0. MAE feature extractor (pixel) for the non-grounded baselines ----
if [[ ! -e runs/cifar_mae_pixel_latest/params_ema/ema_params.pt ]]; then
  echo "===== [0] CIFAR-10 MAE feature extractor ====="
  CUDA_VISIBLE_DEVICES=$GPU_START $PY main.py --config configs/mae/cifar_pixel_64.yaml \
    --workdir runs/cifar_mae_pixel --no-stamp 2>&1 | tail -3
  ln -sfn cifar_mae_pixel runs/cifar_mae_pixel_latest
fi

run_one () {  # $1=name $2=gpu $3=config
  local name=$1 gpu=$2 cfg=$3
  echo "===== [$name] on GPU $gpu ====="
  CUDA_VISIBLE_DEVICES=$gpu $PY main.py --gen --config "configs/exp/${cfg}.yaml" \
    --workdir "runs/${name}" --no-stamp 2>&1 | tail -3
}

# Layer "noise prior" (P-P vs L-P) is dropped: input is just different-sized noise.
declare -a L1=(C10_L1_gsdvae C10_L1_grae C10_L1_gpae)
declare -a L2=(C10_L2_floss C10_L2_repcond C10_L2_both)

run_layer () {  # $1=layer-name $2=space-separated jobs
  local layer=$1; shift
  local jobs=("$@")
  local g=$GPU_START pids=()
  for name in "${jobs[@]}"; do
    run_one "$name" "$g" "$name" &
    pids+=($!)
    g=$(( (g+1) % 8 ))
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  echo "===== $layer done ====="
}

if [[ "$ONLY" == "all" || "$ONLY" == "L1" ]]; then run_layer "L1 (grounding)" "${L1[@]}"; fi
if [[ "$ONLY" == "all" || "$ONLY" == "L2" ]]; then run_layer "L2 (mechanisms)" "${L2[@]}"; fi


# ---- collect results + visualize ----
echo "===== collect + visualize ====="
$PY scripts/collect_c10_results.py 2>/dev/null || true
for name in "${L1[@]}" "${L2[@]}"; do
  [[ -d "runs/$name" ]] && $PY scripts/analyze_run.py "runs/$name" 2>/dev/null || true
done

echo "ALL EXPERIMENTS DONE. See runs/index.md and runs/*/log/report.md"
