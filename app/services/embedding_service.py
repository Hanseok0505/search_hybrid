from __future__ import annotations
import hashlib
import json
import struct
from app.utils.compat import to_thread

import httpx
try:
    import boto3
except Exception:  # pragma: no cover
    boto3 = None
try:
    import litellm
except Exception:  # pragma: no cover
    litellm = None

from app.core.config import settings


class EmbeddingService:
    def __init__(self) -> None:
        self._provider = settings.llm_provider.lower()
        self._api_key = settings.llm_api_key
        self._enabled = bool(self._api_key) if self._provider == "openai" else self._provider == "bedrock"
        self._base_url = settings.llm_base_url.rstrip("/")
        self._model = settings.llm_embedding_model
        self._bedrock_model_id = settings.bedrock_embedding_model_id
        self._bedrock_region = settings.bedrock_region
        self._timeout = settings.llm_timeout_sec
        self._ollama_timeout = max(settings.llm_timeout_sec, settings.ollama_timeout_sec)
        self._dim = settings.vector_dim
        self._ollama_model = settings.ollama_default_model
        self._ollama_urls = [u.strip().rstrip("/") for u in [settings.ollama_base_url, settings.ollama_base_urls] if u.strip()]
        self._client = httpx.AsyncClient(timeout=self._timeout)
        self._bedrock = (
            boto3.client("bedrock-runtime", region_name=self._bedrock_region)
            if self._provider == "bedrock" and boto3 is not None
            else None
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def embed(self, text: str) -> list[float]:
        if self._provider == "openai" and self._enabled:
            headers = {"Authorization": f"Bearer {self._api_key}"}
            body = {"model": self._model, "input": text}
            resp = await self._client.post(f"{self._base_url}/embeddings", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()["data"][0]["embedding"]
            if len(data) >= self._dim:
                return data[: self._dim]
            return data + [0.0] * (self._dim - len(data))
        if self._provider == "bedrock" and self._enabled and self._bedrock:
            try:
                data = await to_thread(self._embed_bedrock, text)
                if len(data) >= self._dim:
                    return data[: self._dim]
                return data + [0.0] * (self._dim - len(data))
            except Exception:
                pass
        if self._provider == "ollama":
            data = await self._embed_ollama(text)
            if data:
                if len(data) >= self._dim:
                    return data[: self._dim]
                return data + [0.0] * (self._dim - len(data))

        # Deterministic local fallback to keep MVP callable without external API.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = []
        while len(values) < self._dim:
            for i in range(0, len(digest), 4):
                chunk = digest[i : i + 4]
                if len(chunk) < 4:
                    continue
                num = struct.unpack(">I", chunk)[0] / 4294967295.0
                values.append((num * 2) - 1)
                if len(values) == self._dim:
                    break
            digest = hashlib.sha256(digest).digest()
        return values

    def _embed_bedrock(self, text: str) -> list[float]:
        assert self._bedrock is not None
        payload = {"inputText": text}
        response = self._bedrock.invoke_model(
            modelId=self._bedrock_model_id,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
        parsed = json.loads(response["body"].read().decode("utf-8"))
        embedding = parsed.get("embedding") or parsed.get("embeddings", [[]])[0]
        return [float(x) for x in embedding]

    async def _embed_ollama(self, text: str) -> list[float]:
        if litellm is None:
            return []

        urls: list[str] = []
        for base in self._ollama_urls:
            for item in base.split(","):
                candidate = item.strip().rstrip("/")
                if candidate and candidate not in urls:
                    urls.append(candidate)

        if not urls:
            return []

        candidates = await self._ollama_model_candidates()
        for base in urls:
            for model in candidates:
                try:
                    raw = await litellm.aembedding(
                        model=f"ollama/{model}",
                        input=[text],
                        api_base=base,
                        timeout=self._ollama_timeout,
                    )
                    data = raw.get("data", [])
                    if not data:
                        continue
                    first = data[0] if isinstance(data[0], dict) else {}
                    return [float(x) for x in first.get("embedding", [])]
                except Exception:
                    continue
        return []

    async def _ollama_model_candidates(self) -> list[str]:
        requested = self._ollama_model
        urls: list[str] = []
        for base in self._ollama_urls:
            for item in base.split(","):
                candidate = item.strip().rstrip("/")
                if candidate and candidate not in urls:
                    urls.append(candidate)

        available: list[str] = []
        for base in urls:
            try:
                resp = await self._client.get(f"{base}/api/tags")
                if resp.status_code >= 400:
                    continue
                payload = resp.json()
                names = [m.get("name") for m in payload.get("models", []) if isinstance(m, dict)]
                available = [str(name) for name in names if isinstance(name, str) and name.strip()]
                if available:
                    break
            except Exception:
                continue

        ordered: list[str] = []
        if requested:
            ordered.append(requested)
        for name in available:
            if name not in ordered:
                ordered.append(name)

        if not ordered:
            return []
        if len(ordered) == 1:
            return ordered

        seen: set[str] = set()
        deduped: list[str] = []
        for name in ordered:
            if name in seen:
                continue
            seen.add(name)
            deduped.append(name)
        return deduped




