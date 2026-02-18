#!/usr/bin/env bash
set -euo pipefail

BUNDLE_PATH="${1:-${AIRGAP_BUNDLE_PATH:-airgap-bundle.tar.gz}}"
TARGET_DIR="${2:-/opt/hybrid-search}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -z "$BUNDLE_PATH" ]]; then
  echo "[airgap-deploy] ERROR: BUNDLE_PATH is empty"
  exit 1
fi

if [[ ! -f "$BUNDLE_PATH" ]]; then
  if [[ -f "$SCRIPT_DIR/$BUNDLE_PATH" ]]; then
    BUNDLE_PATH="$SCRIPT_DIR/$BUNDLE_PATH"
  else
    echo "[airgap-deploy] ERROR: bundle archive not found: $BUNDLE_PATH"
    exit 1
  fi
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[airgap-deploy] ERROR: docker is not installed or not in PATH"
  exit 1
fi

mkdir -p "$TARGET_DIR"

echo "[airgap-deploy] extracting: $BUNDLE_PATH -> $TARGET_DIR"
tar -xzf "$BUNDLE_PATH" -C "$TARGET_DIR"

BUNDLE_DIR="$TARGET_DIR/airgap-bundle"
if [[ ! -d "$BUNDLE_DIR" ]]; then
  echo "[airgap-deploy] ERROR: expected directory $BUNDLE_DIR not found after extraction"
  exit 1
fi

cd "$BUNDLE_DIR"

if [[ ! -f .env.airgap ]]; then
  echo "[airgap-deploy] create .env.airgap from template"
  cp .env.airgap.example .env.airgap
fi

if [[ ! -f .env.airgap ]]; then
  echo "[airgap-deploy] ERROR: .env.airgap missing"
  exit 1
fi

echo "[airgap-deploy] load images"
bash load_bundle.sh

if [[ "${AIRGAP_AUTOSTART:-1}" == "1" ]]; then
  echo "[airgap-deploy] start services"
  docker compose -f docker-compose.airgap.yml --env-file .env.airgap up -d

  if [[ "${AIRGAP_WAIT_HEALTH:-1}" == "1" ]]; then
    for i in $(seq 1 90); do
      if curl -sf http://127.0.0.1:8080/v1/health >/dev/null 2>&1; then
        echo "[airgap-deploy] health check passed"
        exit 0
      fi
      sleep 2
    done
    echo "[airgap-deploy] WARN: services started but health check timed out"
    exit 1
  fi
fi

echo "[airgap-deploy] done"