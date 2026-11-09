#!/usr/bin/env bash
# Run ALL quality experiments in one shot: MNIST (class-acc + samples) and
# CIFAR-10 (FID). Assumes CIFAR-10 is at data/cifar10/ (downloaded) and the
# offline bundle is at $DRIFT_OFFLINE_BUNDLE.
#
#   bash scripts/run_all_quality.sh
set -uo pipefail
cd "$(dirname "$0")/.."

PY=/home/lxw/miniconda3/envs/drifting/bin/python
export DRIFT_OFFLINE_BUNDLE=${DRIFT_OFFLINE_BUNDLE:-/mnt/afs/shawn/offline_bundle}
export HF_HUB_OFFLINE=1
export TORCH_HOME=${TORCH_HOME:-$DRIFT_OFFLINE_BUNDLE/torch}
GPU0=${GPU0:-3}; GPU1=${GPU1:-4}

mkdir -p runs logs

echo "===== [1/6] CIFAR-10 FID reference stats ====="
$PY scripts/compute_fid_stats.py --dataset cifar10 --num 10000 2>&1 | tail -2

echo "===== [2/6] CIFAR-10 MAE feature extractor ====="
CUDA_VISIBLE_DEVICES=$GPU0 $PY main.py --config configs/mae/cifar_pixel_64.yaml \
  --workdir runs/cifar_mae_pixel --no-stamp 2>&1 | tail -3
ln -sfn "$(basename "$(ls -dt runs/*_cifar_mae_pixel | head -1)")" runs/cifar_mae_pixel_latest

echo "===== [3/6] CIFAR-10 generators: P-P baseline vs X-Drifting ====="
CUDA_VISIBLE_DEVICES=$GPU0 $PY main.py --gen --config configs/exp/C10_P_pixel.yaml \
  --workdir runs/C10_P_pixel --no-stamp 2>&1 | tail -3 &
PID1=$!
CUDA_VISIBLE_DEVICES=$GPU1 $PY main.py --gen --config configs/exp/C10_X_LP.yaml \
  --workdir runs/C10_X_LP --no-stamp 2>&1 | tail -3 &
PID2=$!
wait $PID1 $PID2

echo "===== [4/6] CIFAR-10 FID evaluation ====="
$PY scripts/eval_fid_all.py --runs runs/C10_P_pixel runs/C10_X_LP --dataset cifar10 --num 10000 2>&1 | tail -20

echo "===== [5/6] MNIST quality (class-acc) ====="
$PY scripts/eval_mnist_gen.py --run runs/E1_P_pixel runs/X_LP_drift --num 5000 2>&1 | grep -vE "B/s|it/s" | tail -12

echo "===== [6/6] summary ====="
$PY - <<'PY'
import json, os
for f in ["runs/mnist_eval_summary.json", "runs/c10_eval_summary.json"]:
    if os.path.exists(f):
        print(f"\n=== {f} ===")
        for r in json.load(open(f)):
            print({k: (round(v,3) if isinstance(v,float) else v) for k,v in r.items() if k in ("run","fid","class_acc","num_samples")})
PY
echo "ALL QUALITY EXPERIMENTS DONE"
