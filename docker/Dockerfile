# Reproducible image for the drifting / CrossFlow ImageNet experiments.
#
# Build (needs network + a working docker daemon):
#     bash scripts/build_docker.sh            # builds + `docker save`s a tar
#
# The image contains the code + the conda env only. Datasets, the offline asset
# bundle and run outputs are mounted at runtime (see docs/CLUSTER-GUIDE.md), so
# the image stays small and can be reused across experiments.
#
# Base: CUDA 13 devel runtime matching the cu130 torch build in environment.yml.
# We keep the devel variant so `torch.compile` can find nvcc/ptxas; switch to
# `-runtime-` if you never compile kernels.
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:$PATH

# Minimal system deps: git for the code, libgl for PIL/torchvision, and the
# utils needed by the entrypoint scripts.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        bzip2 \
        git \
        libgl1 \
        libglib2.0-0 \
        tini \
    && rm -rf /var/lib/apt/lists/*

# --- conda env -------------------------------------------------------------
COPY docker/environment.yml /tmp/environment.yml
# `-p /opt/env` keeps the env at a stable path so it can be activated with an
# absolute path (no shell init needed in non-interactive `docker run`).
RUN curl -fsSL -o /tmp/miniconda.sh \
        https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && conda config --system --set channel_priority flexible \
    && conda tos accept --override-channels --channel defaults || true \
    && conda create -y -p /opt/env --file /tmp/environment.yml \
    && conda clean -afy \
    && rm -rf /opt/conda/pkgs

# --- code ------------------------------------------------------------------
WORKDIR /workspace/drifting
COPY . /workspace/drifting
# Drop anything that must not be baked into the image.
RUN rm -rf /workspace/drifting/runs /workspace/drifting/data /workspace/drifting/.git \
           /workspace/drifting/offline_bundle /workspace/drifting/logs

# Make the env the default python without requiring conda activation.
ENV PATH=/opt/env/bin:$PATH \
    CONDA_PREFIX=/opt/env \
    DRIFT_REPO_ROOT=/workspace/drifting \
    DRIFT_IN_CONTAINER=1

# `drift-train` wires the offline bundle + distributed launcher, then runs main.py.
RUN install -m 0755 /workspace/drifting/scripts/docker/entrypoint.sh /usr/local/bin/drift-train

# Fail fast if the env is broken.
RUN python -c "import torch, torchvision, diffusers, transformers, matplotlib, pytorch_fid, torch_fidelity; \
print('torch', torch.__version__, 'cuda_build', torch.version.cuda); \
print('ok')"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["drift-train", "--help"]
