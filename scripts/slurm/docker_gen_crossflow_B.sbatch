#!/usr/bin/env bash
# ImageNet drift/CrossFlow training inside the packaged container.
# Adjust the SBATCH block and the mounts for your cluster, then:
#     sbatch scripts/slurm/docker_gen_crossflow_B.sbatch
#
#SBATCH --job-name=cf_B
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

# ---- site-specific paths ----------------------------------------------------
IMAGE="${DRIFT_IMAGE:-drift:latest}"
BUNDLE="${DRIFT_OFFLINE_BUNDLE:?set DRIFT_OFFLINE_BUNDLE (/scratch/\$USER/offline_bundle)}"
IMAGENET="${IMAGENET_PATH:?set IMAGENET_PATH (ImageFolder root with train/ val/)}"
LATCACHE="${IMAGENET_CACHE_PATH:-}"
RUNS="${DRIFT_RUNS_DIR:-$PWD/runs}"
CONFIG="${CONFIG:-configs/gen/crossflow_sota_B.yaml}"
RUN_NAME="${RUN_NAME:-crossflow_B}"
EXTRA_ARGS="${EXTRA_ARGS:---gen}"

mkdir -p "$RUNS"

# ---- launch -----------------------------------------------------------------
# Resume behaviour: RESUME=1 adds --resume (requires an existing checkpoint);
# otherwise the run starts/continues under --no-stamp so the directory is stable
# across submissions. Set RESUME=1 when resubmitting after a preemption.
RESUME_FLAG="${RESUME:+--resume}"
STAMP_FLAG=""
[[ -z "${RESUME:-}" ]] && STAMP_FLAG="--no-stamp"

# --gpus all exposes every GPU on the node; the entrypoint picks the distributed
# launcher and size from NPROC_PER_NODE / SLURM_*.
# Every writable path (runs, script logs, torch inductor cache) is mounted from
# the host so nothing is lost when the container exits.
srun --ntasks=1 --gpus-per-task="${SLURM_GPUS_ON_NODE:-8}" \
  docker run --rm \
    --gpus all \
    --ipc=host \
    --shm-size=64g \
    --user "$(id -u):$(id -g)" \
    -e NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-8}" \
    -e NNODES=1 \
    -e DRIFT_OFFLINE_BUNDLE=/bundle \
    -e IMAGENET_PATH=/data/imagenet \
    ${LATCACHE:+-e IMAGENET_CACHE_PATH=/data/latent_cache} \
    -v "$BUNDLE":/bundle:ro \
    -v "$IMAGENET":/data/imagenet:ro \
    ${LATCACHE:+-v "$LATCACHE":/data/latent_cache:ro} \
    -v "$RUNS":/workspace/drifting/runs \
    -v "$PWD/logs":/workspace/drifting/logs \
    "$IMAGE" \
    drift-train ${EXTRA_ARGS} --config "$CONFIG" \
      --workdir "/workspace/drifting/runs/$RUN_NAME" ${STAMP_FLAG} ${RESUME_FLAG}
