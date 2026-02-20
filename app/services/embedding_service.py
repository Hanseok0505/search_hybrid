from __future__ import annotations

import hashlib
import json
import struct
from app.domain.construction import _ollama_model_aliases
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
        self._ollama_urls = self._collect_ollama_urls()
        self._ollama_model_cache: dict[str, list[str]] = {}
        self._client = httpx.AsyncClient(timeout=self._timeout)
        self._bedrock = (
            boto3.client("bedrock-runtime", region_name=self._bedrock_region)
            if self._provider == "bedrock" and boto3 is not None
            else None
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def embed(self, text: str) -> list[float]:
        if not text:
            return self._fallback_vector("")

        if self._provider == "openai" and self._enabled:
            headers = {"Authorization": f"Bearer {self._api_key}"}
            body = {"model": self._model, "input": text}
            resp = await self._client.post(f"{self._base_url}/embeddings", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()["data"][0]["embedding"]
            if len(data) >= self._dim:
                return data[: self._dim]
            return data + [0.0] * (self._dim - len(data))

        if self._provider == "openai_compatible":
            headers = self._build_headers(self._api_key)
            body = {"model": self._model, "input": text}
            resp = await self._client.post(f"{self._base_url}/embeddings", headers=headers, json=body)
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data", [])
            if data and isinstance(data[0], dict):
                values = data[0].get("embedding", [])
                if isinstance(values, list) and len(values) >= self._dim:
                    return values[: self._dim]
                if isinstance(values, list):
                    return values + [0.0] * (self._dim - len(values))

        if self._provider == "bedrock" and self._enabled and self._bedrock:
            try:
                data = await to_thread(self._embed_bedrock, text)
                if len(data) >= self._dim:
                    return data[: self._dim]
                return data + [0.0] * (self._dim - len(data))
            except Exception:
                pass

        if self._provider == "ollama":
            try:
                data = await self._embed_ollama(text)
                if data:
                    if len(data) >= self._dim:
                        return data[: self._dim]
                    return data + [0.0] * (self._dim - len(data))
            except Exception:
                pass

        return self._fallback_vector(text)

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
        if not text:
            return self._fallback_vector(text)

        urls = self._collect_ollama_urls()
        if not urls:
            return self._fallback_vector(text)

        candidates = await self._ollama_model_candidates()
        if not candidates:
            candidates = [self._ollama_model]

        for model in candidates:
            for base in urls:
                if litellm is not None:
                    try:
                        raw = await litellm.aembedding(
                            model=f"ollama/{model}",
                            input=[text],
                            api_base=base,
                            timeout=min(self._ollama_timeout, 25.0),
                        )
                        data = raw.get("data", [])
                        if not data:
                            continue
                        first = data[0] if isinstance(data[0], dict) else {}
                        values = first.get("embedding", [])
                        if isinstance(values, list) and values:
                            return [float(x) for x in values]
                    except Exception:
                        pass

                try:
                    direct = await self._client.post(
                        f"{base}/api/embeddings",
                        json={"model": model, "prompt": text},
                        timeout=min(self._ollama_timeout, 25.0),
                    )
                    if direct.status_code >= 400:
                        continue
                    direct_json = direct.json()
                    values = direct_json.get("embedding") or []
                    if not values and isinstance(direct_json.get("data"), list) and direct_json["data"]:
                        payload = direct_json["data"][0]
                        if isinstance(payload, dict):
                            values = payload.get("embedding", [])
                    if isinstance(values, list) and values:
                        return [float(x) for x in values]
                except Exception:
                    continue

        return self._fallback_vector(text)

    def _collect_ollama_urls(self) -> list[str]:
        raw_values: list[str] = []
        for value in (settings.ollama_base_url.strip(), settings.ollama_base_urls.strip()):
            if not value:
                continue
            for item in value.split(","):
                candidate = item.strip().rstrip("/")
                if candidate and candidate not in raw_values:
                    raw_values.append(candidate)

        # Local-first fallbacks for direct host run / container bridge usage.
        for candidate in (
            "http://localhost:11434",
            "http://127.0.0.1:11434",
            "http://host.docker.internal:11434",
            "http://ollama:11434",
            "http://host.docker.internal:11434",
        ):
            if candidate not in raw_values:
                raw_values.append(candidate)

        return raw_values

    @staticmethod
    def _build_headers(api_key: Optional[str]) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def _ollama_model_candidates(self) -> list[str]:
        requested = self._ollama_model
        urls = self._collect_ollama_urls()

        ordered: list[str] = []
        if requested:
            ordered.extend(_ollama_model_aliases(requested))

        # Prefer common embedding models when request model is unavailable.
        ordered.extend(
            [
                "nomic-embed-text",
                "mxbai-embed-large",
                "all-minilm",
                "bge-base-en-v1.5",
                "bge-small-en-v1.5",
            ]
        )

        for base in urls:
            names = self._ollama_model_cache.get(base)
            if names is None:
                try:
                    resp = await self._client.get(
                        f"{base}/api/tags",
                        timeout=min(self._ollama_timeout, 8.0),
                    )
                    if resp.status_code >= 400:
                        names = []
                    else:
                        payload = resp.json()
                        values = [m.get("name") for m in payload.get("models", []) if isinstance(m, dict)]
                        names = [str(name).strip() for name in values if isinstance(name, str) and str(name).strip()]
                except Exception:
                    names = []
                self._ollama_model_cache[base] = names

            for name in names:
                if name not in ordered:
                    ordered.append(name)
                for alias in _ollama_model_aliases(name):
                    if alias not in ordered:
                        ordered.append(alias)

        if requested and requested not in ordered:
            ordered.append(requested)

        if not ordered:
            return []

        # Bring known embedding-capable models forward when they are present.
        embed_preference = [name for name in ordered if self._looks_like_embedding_model(name)]
        if embed_preference:
            ordered = embed_preference + [name for name in ordered if name not in embed_preference]

        deduped: list[str] = []
        seen = set()
        for value in ordered:
            if not value or value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped

    @staticmethod
    def _looks_like_embedding_model(model_name: str) -> bool:
        lowered = model_name.lower()
        return any(token in lowered for token in ("embed", "nomic", "mxbai", "bge", "e5", "minilm"))

    def _fallback_vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
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
