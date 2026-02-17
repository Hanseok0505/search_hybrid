#!/bin/sh
set -eu

OLLAMA_HOST="${OLLAMA_HOST:-${OLLAMA_BASE_URL:-http://host.docker.internal:11434}}"
export OLLAMA_HOST

MODELS="${OLLAMA_PRELOAD_MODELS:-${OLLAMA_DEFAULT_MODEL:-}}"
TIMEOUT_SECONDS="${OLLAMA_PRELOAD_TIMEOUT_SEC:-300}"

if [ -z "$MODELS" ]; then
  echo "[ollama-bootstrap] No model names provided. set OLLAMA_PRELOAD_MODELS or OLLAMA_DEFAULT_MODEL."
  exit 0
fi

for i in $(seq 1 "$TIMEOUT_SECONDS"); do
  if OLLAMA_HOST="$OLLAMA_HOST" ollama list >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! OLLAMA_HOST="$OLLAMA_HOST" ollama list >/dev/null 2>&1; then
  echo "[ollama-bootstrap] ERROR: Ollama server not reachable at $OLLAMA_HOST within timeout"
  exit 1
fi

for raw in $(printf '%s' "$MODELS" | tr ',' '\n'); do
  model="$(printf '%s' "$raw" | tr -d '\r' | xargs)"
  if [ -z "$model" ]; then
    continue
  fi

  if ollama list | awk '{print $1}' | grep -Fxq -- "$model"; then
    echo "[ollama-bootstrap] Already present: $model"
    continue
  fi

  echo "[ollama-bootstrap] Pulling: $model"
  if ollama pull "$model"; then
    echo "[ollama-bootstrap] Pulled successfully: $model"
  else
    echo "[ollama-bootstrap] WARN: Failed to pull $model"
  fi
done

echo "[ollama-bootstrap] Completed"
