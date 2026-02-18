# Air-Gap Deployment Guide (Ubuntu 22.04)

This project provides an offline deployment package for an Ubuntu 22.04 host without direct internet access.

## 1) Package on online machine

Requirements:
- Docker + Docker Compose v2
- Ollama running locally (optional) if you want on-prem LLMs

From repository root:

```bash
python -m pip install --upgrade pip
scripts/airgap/build_offline_bundle.sh
```

This creates:
- `airgap-bundle.tar.gz`
  - `airgap-bundle/images/*.tar` : all required images
  - `docker-compose.airgap.yml` : offline compose manifest
  - `.env.airgap.example`
  - `AIR_GAP_DEPLOYMENT.md`
  - `load_bundle.sh`
  - `deploy_airgap.sh`
  - `infra/nginx/nginx.conf`, `infra/redis/redis.conf`

If you use a private app image tag, set:

```bash
APP_IMAGE_NAME=registry.local/hybrid-search:airgap \
scripts/airgap/build_offline_bundle.sh
```

Then update `.env.airgap` values in the target server and use the same tag in compose if changed.

## 2) Transfer to air-gapped Ubuntu host

Copy `airgap-bundle.tar.gz` using removable media or approved internal channel.

```bash
scp /path/airgap-bundle.tar.gz <offline-host>:/tmp/
```

## 3) Prepare offline host

```bash
mkdir -p /opt/hybrid-search
tar -xzf /tmp/airgap-bundle.tar.gz -C /opt/hybrid-search
chmod +x /opt/hybrid-search/airgap-bundle/deploy_airgap.sh
cd /opt/hybrid-search/airgap-bundle
AIRGAP_AUTOSTART=1 AIRGAP_WAIT_HEALTH=1 bash deploy_airgap.sh /tmp/airgap-bundle.tar.gz /opt/hybrid-search
```

## 4) Configure runtime

From `/opt/hybrid-search/airgap-bundle`:

```bash
cp .env.airgap.example .env.airgap
```

Edit `.env.airgap`:
- `LLM_PROVIDER=ollama`
- `OLLAMA_BASE_URL=http://host.docker.internal:11434`
- `OLLAMA_BASE_URLS=http://host.docker.internal:11434,http://localhost:11434`
- any auth/storage/embedding settings as needed

> On Linux, compose uses `host-gateway` mapping so `host.docker.internal` can resolve to host.

## 5) Start services

```bash
cd /opt/hybrid-search/airgap-bundle
docker compose -f docker-compose.airgap.yml --env-file .env.airgap up -d
```

Check readiness:

```bash
docker compose -f docker-compose.airgap.yml ps
curl -s http://localhost:8080/v1/health
```

## 6) Smoke check

```bash
python scripts/check_runtime.py --api-base http://localhost:8080 --skip-compose
```

If backend services are healthy, expected health output should include `status=ok`/`degraded`.

## 7) Notes

- This package excludes external dependency downloads. Ensure required artifacts are available before packaging.
- For air-gap environments that need updates, rebuild and regenerate a fresh bundle, then replace on target host.
- If you use a different local LLM provider, set `LLM_PROVIDER` and model IDs in `.env.airgap`.
