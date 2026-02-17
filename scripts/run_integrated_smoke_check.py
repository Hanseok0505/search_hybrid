from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import httpx
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.utils.compat import to_thread


class _AsyncResponseFacade:
    """Lightweight async HTTP facade used by both remote and in-process smoke checks."""

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        raise NotImplementedError

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        raise NotImplementedError


class _AsyncHTTPXClient(_AsyncResponseFacade):
    def __init__(self, base_url: str, request_timeout: float = 20.0) -> None:
        self._client = httpx.AsyncClient(timeout=request_timeout, base_url=base_url.rstrip("/"))

    async def __aenter__(self) -> "_AsyncHTTPXClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self._client.get(url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self._client.post(url, **kwargs)

    async def close(self) -> None:
        await self._client.aclose()


class _TestClientAdapter(_AsyncResponseFacade):
    def __init__(self, client: object) -> None:
        self._client = client

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return await to_thread(self._client.get, url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return await to_thread(self._client.post, url, **kwargs)


def _print(msg: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}")


def _fail(message: str) -> None:
    _print(f"FAIL: {message}")
    raise SystemExit(1)


async def _post_upload(
    client: _AsyncResponseFacade,
    project_id: str,
) -> dict[str, Any]:
    top_down_context = {
        "project_id": project_id,
        "building": "TOWER-A",
        "level": "B2",
        "work_type": "concrete",
    }
    content = "Smoke test upload content. Concrete pour quality checklist evidence note."
    files = {"file": ("smoke_check.txt", io.BytesIO(content.encode("utf-8")), "text/plain")}
    r = await client.post(
        "/v1/upload",
        data={"top_down_context": json.dumps(top_down_context), "already_embedded": "true"},
        files=files,
    )
    if r.status_code >= 400:
        _fail(f"upload failed: {r.status_code} {r.text}")
    data = r.json()
    _print(f"uploaded: {data.get('doc_id')}")
    return data


async def _check_health(client: _AsyncResponseFacade, strict: bool) -> dict[str, Any]:
    r = await client.get("/v1/health")
    if r.status_code != 200:
        _fail(f"health status code {r.status_code}")

    payload = r.json()
    if strict:
        for key in ("elastic", "milvus", "graph", "redis"):
            if payload.get(key) != "up":
                _fail(f"strict mode: {key}={payload.get(key)}")
    elif payload.get("status") == "down":
        _fail("health is down")

    _print(f"health ok: {payload}")
    return payload


async def _check_models(client: _AsyncResponseFacade, provider: str) -> dict[str, Any]:
    r = await client.get("/v1/models", params={"provider": provider})
    if r.status_code != 200:
        _fail(f"models failed: {r.status_code} {r.text}")
    payload = r.json()
    _print(f"models[{provider}] = {payload}")
    return payload


async def _check_sources(client: _AsyncResponseFacade) -> dict[str, Any]:
    r = await client.get("/v1/sources")
    if r.status_code != 200:
        _fail(f"sources failed: {r.status_code} {r.text}")
    payload = r.json()
    _print(f"sources count={len(payload.get('items', []))}")
    return payload


async def _check_search(
    client: _AsyncResponseFacade,
    provider: str,
    model: Optional[str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": "Tower A B2 slab pour quality checklist",
        "top_k": 5,
        "top_down_context": {
            "project_id": "PROJ-SMART-CAMPUS",
            "building": "TOWER-A",
            "level": "B2",
        },
        "provider": provider,
        "use_cache": False,
    }
    if model:
        payload["model"] = model

    r = await client.post("/v1/search", json=payload)
    if r.status_code != 200:
        _fail(f"search failed: {r.status_code} {r.text}")

    body = r.json()
    if not isinstance(body.get("hits"), list):
        _fail(f"search invalid response: {body}")
    _print(
        "search hits="
        f"{len(body['hits'])}, took_ms={body.get('took_ms')}, cache_hit={body.get('cache_hit')}"
    )
    return body


async def _check_ask(
    client: _AsyncResponseFacade,
    provider: str,
    model: Optional[str],
    selected_source: Optional[str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": "Summarize quality checklist and concrete pour readiness",
        "top_k": 5,
        "top_down_context": {
            "project_id": "PROJ-SMART-CAMPUS",
            "building": "TOWER-A",
            "level": "B2",
        },
        "provider": provider,
        "use_cache": False,
    }
    if model:
        payload["model"] = model
    if selected_source:
        payload["selected_source_ids"] = [selected_source]

    r = await client.post("/v1/ask", json=payload)
    if r.status_code != 200:
        _fail(f"ask failed: {r.status_code} {r.text}")

    body = r.json()
    if not body.get("answer"):
        _fail("ask returned empty answer")
    if not isinstance(body.get("hits"), list):
        _fail(f"ask invalid response: {body}")

    if selected_source:
        hit_ids = [x.get("id") for x in body.get("hits", [])]
        if selected_source not in hit_ids:
            _print("WARN: selected source was not included in final hits")

    answer = body.get("answer", "")
    if "Key Evidence" not in answer and "Risks/Gaps" not in answer:
        _print("WARN: answer format is unusual, but non-empty")

    _print(f"ask hits={len(body['hits'])}, provider={body.get('provider')}, model={body.get('model')}")
    return body


async def _check_free_answer(client: _AsyncResponseFacade) -> dict[str, Any]:
    r = await client.post("/v1/free/answer", json={"query": "Tower A B2 slab safety checklist"})
    if r.status_code != 200:
        _fail(f"free answer failed: {r.status_code} {r.text}")
    body = r.json()
    if not body.get("answer"):
        _fail(f"free answer empty: {body}")
    _print(f"free answer provider={body.get('provider')}")
    return body


async def _check_cache_stats_if_needed(client: _AsyncResponseFacade) -> None:
    r = await client.get("/v1/cache/stats")
    if r.status_code == 404:
        _print("WARN: cache stats endpoint unavailable")
        return
    if r.status_code != 200:
        _print(f"WARN: cache stats failed with {r.status_code}")
        return
    _print(f"cache stats={r.json()}")


async def _run_checks(
    client: _AsyncResponseFacade,
    provider: str,
    model: Optional[str],
    strict: bool,
    do_upload: bool,
) -> None:
    await _check_health(client, strict)
    await _check_models(client, provider)
    sources_payload = await _check_sources(client)

    uploaded_id = None
    if do_upload:
        upload = await _post_upload(client, project_id="PROJ-SMART-CAMPUS")
        uploaded_id = upload.get("doc_id")

    await _check_search(client, provider, model)
    await _check_ask(client, provider, model, uploaded_id)
    await _check_free_answer(client)

    _print(f"source_count={len(sources_payload.get('items', []))}")
    await _check_cache_stats_if_needed(client)


async def run(
    base_url: str,
    provider: str,
    model: Optional[str],
    request_timeout: float,
    strict: bool,
    do_upload: bool,
    local: bool,
) -> None:
    if local:
        from app import main as application
        from fastapi.testclient import TestClient

        with TestClient(application.app) as tc:
            client: _AsyncResponseFacade = _TestClientAdapter(tc)
            await _run_checks(
                client=client,
                provider=provider,
                model=model,
                strict=strict,
                do_upload=do_upload,
            )
    else:
        async with _AsyncHTTPXClient(base_url=base_url, request_timeout=request_timeout) as client:
            await _run_checks(
                client=client,
                provider=provider,
                model=model,
                strict=strict,
                do_upload=do_upload,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run integrated smoke checks for hybrid search service")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="Base URL for API server (ex: http://localhost:8080)",
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["ollama", "openai", "openai_compatible", "bedrock"],
        help="Provider used for /v1/search and /v1/ask",
    )
    parser.add_argument("--model", default=None, help="Provider model override")
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="HTTP timeout for each request in seconds (use longer timeout when model response is slow).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run smoke checks against in-process FastAPI app",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require all backends to be up (elastic/milvus/graph/redis)",
    )
    parser.add_argument("--skip-upload", action="store_true", help="Skip upload step")

    args = parser.parse_args()

    asyncio.run(
        run(
            base_url=args.base_url,
            provider=args.provider,
            model=args.model,
            request_timeout=args.request_timeout,
            strict=args.strict,
            do_upload=not args.skip_upload,
            local=args.local,
        )
    )


if __name__ == "__main__":
    main()

