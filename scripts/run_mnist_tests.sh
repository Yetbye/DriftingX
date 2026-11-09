#!/usr/bin/env bash
# MNIST experiment runner for the Drifting / CrossFlow models.
# Trains two MAE feature extractors, then the three generators, then runs pytest.
# Usage: bash scripts/run_mnist_tests.sh [--gpu N]
set -euo pipefail

cd "$(dirname "$0")/.."
GPU=0
[[ "${1:-}" == "--gpu" ]] && GPU="${2:-0}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drifting

export CUDA_VISIBLE_DEVICES=$GPU
export HF_ENDPOINT=https://hf-mirror.com
export SDVAE_PATH="$PWD/hf_cache/sdvae"
export HF_ROOT="$PWD/hf_cache"

link_latest() {  # $1 = run name suffix, $2 = symlink name
  local dir
  dir="$(ls -dt runs/*_"$1" 2>/dev/null | head -1 || true)"
  if [[ -z "$dir" ]]; then echo "no run found for $1" >&2; exit 1; fi
  ln -sfn "$(basename "$dir")" "runs/$2"
  echo "linked runs/$2 -> $dir"
}

mkdir -p logs

echo "===== [0/6] weights ====="
[[ -f "$SDVAE_PATH/diffusion_pytorch_model.safetensors" ]] || python scripts/download_sdvae.py

echo "===== [1/6] MAE feature extractors (latent + pixel) ====="
python main.py --config configs/mae/mnist_real_latent_64.yaml --workdir runs/mnist_real_mae_latent 2>&1 | tail -3
python main.py --config configs/mae/mnist_real_pixel_64.yaml  --workdir runs/mnist_real_mae_pixel  2>&1 | tail -3
link_latest mnist_real_mae_latent mnist_real_mae_latent_latest
link_latest mnist_real_mae_pixel  mnist_real_mae_pixel_latest

echo "===== [2/6] generator — latent (SD-VAE) space ====="
python main.py --gen --config configs/gen/mnist_real_latent_64.yaml --workdir runs/mnist_real_gen_latent 2>&1 | tail -3

echo "===== [3/6] generator — pixel space ====="
python main.py --gen --config configs/gen/mnist_real_pixel_64.yaml --workdir runs/mnist_real_gen_pixel 2>&1 | tail -3

echo "===== [4/6] generator — CrossFlow (latent noise -> pixels) ====="
python main.py --gen --config configs/gen/mnist_real_crossflow_64.yaml --workdir runs/mnist_real_gen_crossflow 2>&1 | tail -3

link_latest mnist_real_gen_latent    mnist_real_gen_latent_latest
link_latest mnist_real_gen_pixel     mnist_real_gen_pixel_latest
link_latest mnist_real_gen_crossflow mnist_real_gen_crossflow_latest

echo "===== [5/6] pytest unit tests ====="
pytest -q

echo "===== [6/6] visualize + summarize ====="
for r in mnist_real_gen_latent mnist_real_gen_pixel mnist_real_gen_crossflow; do
  python scripts/visualize_run.py "runs/${r}_latest" || true
done
python - <<'PY'
import json, os, glob
for link in sorted(glob.glob("runs/mnist_real_gen_*_latest")):
    run = os.path.realpath(link)
    recs = [json.loads(l) for l in open(os.path.join(run, "log/metrics.jsonl"))]
    f, l = recs[0], recs[-1]
    print(f"{os.path.basename(link):40s} loss {f.get('loss'):.1f} -> {l.get('loss'):.1f}  (step {l.get('step')})")
    print(f"    {os.path.join(run, 'log/overview.png')}")
PY

echo "ALL MNIST EXPERIMENTS DONE"
