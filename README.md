# Hybrid Search Platform (Elasticsearch + Milvus + Graph + LLM)

Production-oriented MVP for large-scale retrieval with keyword, vector, graph, cache, and LLM integration.

## 1. Components
- API: FastAPI (`app/main.py`)
- Keyword retrieval: Elasticsearch
- Vector retrieval: Milvus
- Graph retrieval: Neo4j Fulltext + graph relation signal
- Cache: Redis
- Fusion: Weighted RRF
- Optional rerank: LLM reranker
- LLM providers: `openai`, `openai_compatible`, `bedrock`, `ollama`
- UI: Notebook-style 3-pane page (`GET /ui`)
- Parser service: Apache Tika (`tika`) for DOC/XLS/PPT/HWP-like formats.
- LLM transport for Ollama goes through `litellm` (`litellm.acompletion` / `litellm.aembedding`) with default model `gpt-oss-120b-cloud`.

## 2. Quick Start
```bash
cp .env.example .env
docker compose up -d --build
```

### Ollama Local Model Setup
`gpt-oss-120b-cloud` is the platform default, but model availability can differ by host.

Run order for runtime endpoint is:
- `OLLAMA_BASE_URL` (primary): `http://host.docker.internal:11434`
- `OLLAMA_BASE_URLS`: `http://host.docker.internal:11434,http://localhost:11434,http://ollama:11434`
- `OLLAMA_HOST` for bootstrap: from `${OLLAMA_BASE_URL}` by default

```bash
docker compose up -d --build
docker compose logs -f ollama-bootstrap
```

If `tika` is unavailable or you need to restart only the parser:
```bash
docker compose up -d --build tika
``` 

Recommended runtime tuning:
- Set `OLLAMA_TIMEOUT_SEC` in `.env` when using slow models (for example: `OLLAMA_TIMEOUT_SEC=120.0`).

Main endpoints:
- `GET /`
- `GET /ui`
- `GET /v1/health`
- `POST /v1/search`
- `POST /v1/ask` (integrated search + answer)
- `POST /v1/upload` (supports `already_embedded`)
- `GET /v1/sources`
- `GET /v1/models`
- `POST /v1/free/answer`
- `POST /v1/llm/invoke`
- `GET /v1/cache/stats`
- `POST /v1/cache/invalidate`
- `POST /v1/search` accepts optional `selected_source_ids` and `embedded_only` filters (same structure as UI source checks).

## 3. Notebook-Style UX
- Left pane:
  - file upload
  - `already embedded` checkbox
  - source selection checkboxes
  - `embedded only` filter
- Center pane:
  - provider/model selector (`ollama` supported)
  - integrated ask (`/v1/ask`)
  - search-only mode (`/v1/search`)
  - `/v1/search` now fuses local source search (sample + uploaded docs) together with ES/Milvus/Graph results.
- Right pane:
  - debug cards for hits/upload/free-answer
  - note: `Free API` is non-RAG utility, while `RAG Answer` uses `/v1/ask`

## 3-1. Ollama Model Discovery
- `/v1/models?provider=ollama` now probes real Ollama endpoints in order:
  - `OLLAMA_BASE_URL`
  - `OLLAMA_BASE_URLS` (comma-separated fallbacks, default includes `http://host.docker.internal:11434`)
- If all probes fail, response marks model as `available=false`.

## 4. Upload Example
```bash
curl -X POST "http://localhost:8080/v1/upload" \
  -F "file=@C:/path/to/site_note.txt" \
  -F "top_down_context={\"project_id\":\"PROJ-SMART-CAMPUS\",\"building\":\"TOWER-A\"}" \
  -F "already_embedded=true"
```
If the native extension parser cannot extract text, the service now falls back to Tika extraction.

## 5. Ask Example
```bash
curl -X POST "http://localhost:8080/v1/ask" \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"Tower A B2 slab quality checklist summary\",
    \"top_k\": 8,
    \"selected_source_ids\": [\"upl-123\"],
    \"embedded_only\": false,
    \"provider\": \"ollama\",
    \"model\": \"gpt-oss-120b-cloud\",
    \"top_down_context\": {
      \"project_id\": \"PROJ-SMART-CAMPUS\",
      \"building\": \"TOWER-A\",
      \"level\": \"B2\"
    }
  }"
```

## 6. Conda + Local Run
```bash
conda create -y -n hybrid-search python=3.11
conda run -n hybrid-search python -m pip install --upgrade pip
conda run -n hybrid-search pip install -e . pytest
conda run -n hybrid-search pytest -q
conda run -n hybrid-search uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## 7. Redis for 11,000+ Concurrent Users
- connection pool: `REDIS_MAX_CONNECTIONS`
- lock for stampede: `REDIS_LOCK_TTL_SEC`, `REDIS_LOCK_WAIT_MS`
- namespace isolation: `REDIS_NAMESPACE`
- config: `infra/redis/redis.conf`

## 8. Integrated Verification (Production-Style Smoke Check)
Run this after services are up (docker compose or direct app run):

```bash
conda run -n hybrid-search python scripts/run_integrated_smoke_check.py --base-url http://localhost:8080
```

Tip:

- `--local` mode is available for in-process checks when using the same Python environment as the service:

```bash
conda run -n hybrid-search python scripts/run_integrated_smoke_check.py --local
```

- If external services are not up during smoke check, run without `--strict` so degraded backends are tolerated.

Optional flags:

- `--strict`: require `elastic`, `milvus`, `graph`, `redis` to be `up`
- `--provider openai|openai_compatible|bedrock|ollama`
- `--model <model_name>`
- `--request-timeout <seconds>`: per-request HTTP timeout for smoke check (default 120.0)
- `--skip-upload`: skip upload + selected-source RAG check

Example:

```bash
conda run -n hybrid-search python scripts/run_integrated_smoke_check.py \
  --base-url http://localhost:8080 \
  --provider ollama \
  --model gpt-oss-120b-cloud
```

## 9. Ask Synthesis Status

`/v1/ask` now returns additional fields:

- `rag_synthesized` (bool): `true` when LLM answer was generated; `false` when fallback source summarization was used.
- `fallback_reason` (string | null): `null` on LLM success, otherwise reason text.

```json
{
  "answer": "...",
  "provider": "ollama",
  "model": "gpt-oss-120b-cloud",
  "hits": [...],
  "rag_synthesized": false,
  "fallback_reason": "llm_generation_failed: ..."
}
```

## 9. Upload Type Matrix Check
Validates file upload → parse → search flow across common file types.

```bash
conda run -n hybrid-search python scripts/tmp_upload_matrix.py
```

The script uploads text/csv/json/html/docx/xlsx/pptx/hwp/xls sample cases, prints parser/extracted status, and verifies each uploaded doc appears in `/v1/search`.

## 10. Runtime Readiness Check (Compose + API)
Run this before smoke checks to confirm the local infrastructure is actually up.

```bash
conda run -n hybrid-search python scripts/check_runtime.py
```

Options:
- `--project-root .` (path where `docker-compose.yml` exists)
- `--api-base http://localhost:8080`
- `--strict` (fail on any issue)
- `--skip-compose` (skip docker compose checks)
- `--skip-ports` (skip raw socket checks)

## 11. Air-Gap Deployment (Ubuntu 22.04)

This project includes an offline packaging flow for environments without internet access.

```bash
scripts/airgap/build_offline_bundle.sh
```

- Generates `airgap-bundle.tar.gz`
- Includes `docker-compose.airgap.yml`, `.env.airgap`, startup scripts, and all required Docker images
- Loads and runs on an offline Ubuntu 22.04 node

See the full guide:

```bash
docs/AIR_GAP_DEPLOYMENT.md
```

Quick one-shot on an offline Ubuntu host:

```bash
AIRGAP_BUNDLE_PATH=/tmp/airgap-bundle.tar.gz \\
AIRGAP_AUTOSTART=1 \\
AIRGAP_WAIT_HEALTH=1 \\
bash airgap-bundle/deploy_airgap.sh /tmp/airgap-bundle.tar.gz /opt/hybrid-search
```
