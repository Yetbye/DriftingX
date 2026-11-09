#!/usr/bin/env bash
# Rebuild the `drifting` env preferring conda packages (torch/torchvision/numpy/scipy/matplotlib...),
# pip only for packages conda lacks. Uses the tuna mirrors configured in ~/.condarc and pip.conf.
set -euo pipefail
source /home/lxw/miniconda3/etc/profile.d/conda.sh

echo "[1/4] removing old drifting env..."
conda env remove -n drifting -y

echo "[2/4] create base env (python 3.12)..."
conda create -n drifting -y python=3.12

echo "[3/4] conda install torch stack + scientific/viz libs (tuna mirror)..."
conda install -n drifting -y -c pytorch \
  pytorch=2.11.0 torchvision=0.26.0 \
  numpy scipy pyyaml pillow tqdm msgpack-python matplotlib

echo "[4/4] pip install remaining pure-python deps (tuna mirror)..."
/home/lxw/miniconda3/envs/drifting/bin/pip install --no-input \
  absl-py einops diffusers "transformers>=4.44,<5" huggingface_hub \
  torchmetrics "pytorch-fid>=0.3.0" wandb opt-einsum accelerate pytest

echo "=== verify ==="
/home/lxw/miniconda3/envs/drifting/bin/python -c \
  "import torch, torchvision, diffusers, transformers, matplotlib; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count()); print('torchvision', torchvision.__version__); print('matplotlib', matplotlib.__version__)"
echo "DONE"
