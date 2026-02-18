#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
BUNDLE_DIR="${BUNDLE_DIR:-$ROOT_DIR/airgap-bundle}"
BUNDLE_NAME="${BUNDLE_NAME:-airgap-bundle.tar.gz}"
APP_IMAGE_NAME="${APP_IMAGE_NAME:-hybrid-search:latest}"
COMPOSE_FILE="${COMPOSE_FILE:-$ROOT_DIR/docker-compose.airgap.yml}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[airgap] ERROR: docker is not installed or not in PATH"
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "[airgap] ERROR: docker daemon is not running. Start Docker Desktop (or daemon) before building the bundle."
  exit 1
fi

mkdir -p "$ROOT_DIR/scripts/airgap"
cd "$ROOT_DIR"

# 1) build local app image

echo "[airgap] building app image: $APP_IMAGE_NAME"
docker build -t "$APP_IMAGE_NAME" .

# 2) collect required images
IMAGES=(
  "docker.elastic.co/elasticsearch/elasticsearch:8.15.2"
  "redis:7.4-alpine"
  "redis/redisinsight:2.58"
  "neo4j:5.25"
  "quay.io/coreos/etcd:v3.5.5"
  "minio/minio:RELEASE.2025-01-20T14-49-07Z"
  "milvusdb/milvus:v2.4.14"
  "nginx:1.27-alpine"
  "$APP_IMAGE_NAME"
)

# pull if missing/older tags
for image in "${IMAGES[@]}"; do
  if [[ "$image" == "$APP_IMAGE_NAME" ]]; then
    continue
  fi
  echo "[airgap] pulling image: $image"
  docker pull "$image"
done

# 3) save all image tarballs
mkdir -p "$BUNDLE_DIR/images"
for image in "${IMAGES[@]}"; do
  file_name=$(echo "$image" | sed 's#[/:]#_#g').tar
  echo "[airgap] saving image: $image -> $file_name"
  docker save "$image" -o "$BUNDLE_DIR/images/$file_name"
done

# 4) package runbook + compose and env template
mkdir -p "$BUNDLE_DIR"
cp "$COMPOSE_FILE" "$BUNDLE_DIR/docker-compose.airgap.yml"
mkdir -p "$BUNDLE_DIR/infra/nginx" "$BUNDLE_DIR/infra/redis"
cp "$ROOT_DIR/infra/nginx/nginx.conf" "$BUNDLE_DIR/infra/nginx/nginx.conf"
cp "$ROOT_DIR/infra/redis/redis.conf" "$BUNDLE_DIR/infra/redis/redis.conf"
cp "$ROOT_DIR/.env.example" "$BUNDLE_DIR/.env.airgap.example"
cp "$ROOT_DIR/.env.example" "$BUNDLE_DIR/.env.airgap"
cp "$ROOT_DIR/docs/AIR_GAP_DEPLOYMENT.md" "$BUNDLE_DIR/AIR_GAP_DEPLOYMENT.md"
cp "$ROOT_DIR/scripts/airgap/load_bundle.sh" "$BUNDLE_DIR/load_bundle.sh"
cp "$ROOT_DIR/scripts/airgap/deploy_airgap.sh" "$BUNDLE_DIR/deploy_airgap.sh"

chmod +x "$BUNDLE_DIR/load_bundle.sh"
chmod +x "$BUNDLE_DIR/deploy_airgap.sh"

# 5) compress
cd "$ROOT_DIR"
if [[ -f "$BUNDLE_NAME" ]]; then
  rm -f "$BUNDLE_NAME"
fi
tar -czf "$BUNDLE_NAME" -C "$ROOT_DIR" airgap-bundle

echo "[airgap] bundle created: $ROOT_DIR/$BUNDLE_NAME"
echo "[airgap] image list:"
for image in "${IMAGES[@]}"; do
  echo "  - $image"
done
