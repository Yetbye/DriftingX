#!/usr/bin/env bash
# Fallback packaging that needs NO docker daemon and NO root: conda-pack the
# conda env into a relocatable tarball and ship it together with the code.
#
# On the cluster the tarball is extracted and activated with an absolute path, so
# no `conda init` / interactive shell is required:
#
#     tar -xzf drift_env.tar.gz -C /scratch/$USER/envs/drifting
#     source /scratch/$USER/envs/drifting/bin/activate
#
# Produces dist/drift_env.tar.gz and dist/drift_env.manifest.txt
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-drifting}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/dist}"
OUT="${OUT:-${OUT_DIR}/drift_env.tar.gz}"

# conda-pack must run with the target env's own conda available.
if ! command -v conda >/dev/null 2>&1; then
  source /home/lxw/miniconda3/etc/profile.d/conda.sh
fi

mkdir -p "$OUT_DIR"
echo "=== conda-pack '$ENV_NAME' -> $OUT ==="
# --ignore-missing-files tolerates conda-metadata gaps; --n-threads speeds it up.
/home/lxw/miniconda3/envs/drifting/bin/conda-pack -n "$ENV_NAME" -o "$OUT" --ignore-missing-files --n-threads -1 --force

{
  echo "env_name: $ENV_NAME"
  echo "created: $(date -Is)"
  echo "python: $(conda run -n "$ENV_NAME" python -V 2>/dev/null | tail -1)"
  echo "torch: $(conda run -n "$ENV_NAME" python -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)' 2>/dev/null | tail -1)"
  echo "size_bytes: $(stat -c %s "$OUT")"
} > "${OUT_DIR}/drift_env.manifest.txt"

ls -lh "$OUT"
cat "${OUT_DIR}/drift_env.manifest.txt"
echo
echo "Upload '$OUT' + the repo to the cluster, then follow docs/CLUSTER-GUIDE.md."
