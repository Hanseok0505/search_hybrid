#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUNDLE_DIR="$ROOT_DIR"
IMAGE_DIR="$ROOT_DIR/images"

if [[ ! -d "$BUNDLE_DIR" ]]; then
  echo "[airgap] ERROR: bundle directory not found: $BUNDLE_DIR"
  exit 1
fi

if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "[airgap] ERROR: image directory not found: $IMAGE_DIR"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[airgap] ERROR: docker is not installed or not in PATH"
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "[airgap] ERROR: docker daemon is not running. Start Docker Desktop (or daemon) and retry."
  exit 1
fi

# load all container images
for f in "$IMAGE_DIR"/*.tar; do
  if [[ -f "$f" ]]; then
    echo "[airgap] docker load: $f"
    docker load -i "$f"
  fi
done

echo "[airgap] image load complete"

echo "[airgap] loading compose file: docker-compose.airgap.yml"

echo "[airgap] to run services in air-gapped network:"
echo "  cd $ROOT_DIR"
echo "  cp .env.airgap.example .env.airgap"
echo "  # edit .env.airgap (ollama + model endpoints)"
echo "  docker compose -f docker-compose.airgap.yml --env-file .env.airgap up -d"
