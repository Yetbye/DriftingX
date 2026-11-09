#!/usr/bin/env bash
# Build the drifting training image and export it as a tar for upload.
#
# Requires a machine with a WORKING docker daemon and network access (the image
# is built from conda + pip mirrors). The cluster is offline, so we `docker save`
# the finished image and load it there.
#
#   bash scripts/build_docker.sh                       # tag drift:latest
#   TAG=drift:v1 OUT=/scratch/drift_v1.tar bash scripts/build_docker.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TAG="${TAG:-drift:latest}"
OUT="${OUT:-${REPO_ROOT}/dist/$(echo "$TAG" | tr '/:' '__').tar}"
DOCKERFILE="${DOCKERFILE:-docker/Dockerfile}"

DOCKER="${DOCKER:-docker}"
if ! command -v "$DOCKER" >/dev/null 2>&1; then
  echo "ERROR: '$DOCKER' not found on PATH." >&2
  exit 1
fi

if ! "$DOCKER" info >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: docker daemon not reachable.

If you are on a rootless setup and see an error about `newuidmap`, install the
`uidmap` package (needs root) and start the daemon:

    sudo apt-get install -y uidmap
    systemctl --user enable --now docker.service
    export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock

No docker at all? Use the conda-pack fallback instead (no daemon required):

    bash scripts/pack_env.sh
EOF
  exit 1
fi

echo "=== building $TAG from $DOCKERFILE ==="
"$DOCKER" build -f "$DOCKERFILE" -t "$TAG" .

echo "=== saving image to $OUT ==="
mkdir -p "$(dirname "$OUT")"
"$DOCKER" save "$TAG" -o "$OUT"
ls -lh "$OUT"
echo
echo "Upload '$OUT' to the cluster, then:"
echo "    docker load -i $(basename "$OUT")"
