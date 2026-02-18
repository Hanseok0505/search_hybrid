from __future__ import annotations

import json
from typing import Dict, List, Optional

import httpx
from app.utils.compat import to_thread
from app.core.config import settings
from app.models.schemas import LLMInvokeRequest, LLMMessage

try:
    import boto3
except Exception:  # pragma: no cover
    boto3 = None

try:
    import litellm
except Exception:  # pragma: no cover
    litellm = None


class LLMGateway:
    def __init__(self) -> None:
        self._default_provider = settings.llm_provider.lower()
        self._base_url = settings.llm_base_url.rstrip("/")
        self._api_key = settings.llm_api_key
        self._chat_path = settings.llm_chat_path
        self._embed_path = settings.llm_embeddings_path
        self._auth_header = settings.llm_auth_header
        self._auth_prefix = settings.llm_auth_prefix
        self._timeout = settings.llm_timeout_sec
        self._ollama_timeout = max(
            settings.llm_timeout_sec,
            settings.ollama_timeout_sec,
        )
        self._client = httpx.AsyncClient(timeout=self._timeout)

        self._ollama_urls = self._build_ollama_urls()
        self._active_ollama_url = self._ollama_urls[0]

    async def close(self) -> None:
        await self._client.aclose()

    async def invoke(self, payload: LLMInvokeRequest) -> Dict:
        provider = (payload.provider or self._default_provider).lower()
        task = payload.task.lower()
        if task not in {"chat", "embed"}:
            raise ValueError("task must be one of: chat, embed")

        if provider in {"openai", "openai_compatible"}:
            out = await (self._chat_openai_compatible(payload) if task == "chat" else self._embed_openai_compatible(payload))
        elif provider == "bedrock":
            out = await (self._chat_bedrock(payload) if task == "chat" else self._embed_bedrock(payload))
        elif provider == "ollama":
            out = await (self._chat_ollama(payload) if task == "chat" else self._embed_ollama(payload))
        else:
            raise ValueError("provider must be one of: openai, openai_compatible, bedrock, ollama")

        out["provider"] = provider
        out["task"] = task
        return out

    def _headers(self) -> Dict[str, str]:
        if not self._api_key:
            return {}
        return {self._auth_header: f"{self._auth_prefix}{self._api_key}"}

    async def _chat_openai_compatible(self, payload: LLMInvokeRequest) -> Dict:
        model = payload.model or settings.llm_rerank_model
        messages = payload.messages or [LLMMessage(role="user", content=payload.text or "")]
        body: Dict = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": payload.temperature,
        }
        if payload.max_tokens:
            body["max_tokens"] = payload.max_tokens
        r = await self._client.post(f"{self._base_url}{self._chat_path}", headers=self._headers(), json=body)
        r.raise_for_status()
        raw = r.json()
        text = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"model": model, "output_text": text, "raw": raw}

    async def _embed_openai_compatible(self, payload: LLMInvokeRequest) -> Dict:
        model = payload.model or settings.llm_embedding_model
        body = {"model": model, "input": payload.text or ""}
        r = await self._client.post(f"{self._base_url}{self._embed_path}", headers=self._headers(), json=body)
        r.raise_for_status()
        raw = r.json()
        emb = raw.get("data", [{}])[0].get("embedding", [])
        return {"model": model, "embedding": emb, "raw": raw}

    async def _chat_bedrock(self, payload: LLMInvokeRequest) -> Dict:
        if boto3 is None:
            raise RuntimeError("boto3 is not installed")
        model = payload.model or settings.bedrock_rerank_model_id
        region = payload.region or settings.bedrock_region
        messages = payload.messages or [LLMMessage(role="user", content=payload.text or "")]
        client = boto3.client("bedrock-runtime", region_name=region)
        response = await to_thread(
            client.converse,
            modelId=model,
            messages=[{"role": m.role, "content": [{"text": m.content}]} for m in messages],
            inferenceConfig={"temperature": payload.temperature},
        )
        parts = response.get("output", {}).get("message", {}).get("content", [])
        text = "".join(p.get("text", "") for p in parts)
        return {"model": model, "output_text": text, "raw": response}

    async def _embed_bedrock(self, payload: LLMInvokeRequest) -> Dict:
        if boto3 is None:
            raise RuntimeError("boto3 is not installed")
        model = payload.model or settings.bedrock_embedding_model_id
        region = payload.region or settings.bedrock_region
        client = boto3.client("bedrock-runtime", region_name=region)
        text = payload.text or ""
        req_body = {"texts": [text], "input_type": "search_query"} if "cohere" in model.lower() else {"inputText": text}
        response = await to_thread(
            client.invoke_model,
            modelId=model,
            body=json.dumps(req_body),
            contentType="application/json",
            accept="application/json",
        )
        parsed = json.loads(response["body"].read().decode("utf-8"))
        emb = parsed.get("embedding") or parsed.get("embeddings", [[]])[0]
        return {"model": model, "embedding": emb, "raw": parsed}

    async def _chat_ollama(self, payload: LLMInvokeRequest) -> Dict:
        requested_model = payload.model
        requested = requested_model or settings.ollama_default_model
        use_explicit = requested_model is not None
        messages = payload.messages or [LLMMessage(role="user", content=payload.text or "")]
        if litellm is None:
            raise RuntimeError("litellm is not installed. Add litellm to dependencies and retry.")

        payload_messages = [{"role": m.role, "content": m.content} for m in messages]
        last_error = "unknown"
        model_candidates = await self._build_ollama_model_candidates(requested, explicit=use_explicit)

        for model in model_candidates:
            for base in self._ollama_url_candidates():
                try:
                    raw = await litellm.acompletion(
                        model=f"ollama/{model}",
                        messages=payload_messages,
                        api_base=base,
                        temperature=payload.temperature,
                        stream=False,
                        max_tokens=payload.max_tokens or 1024,
                        timeout=self._ollama_timeout,
                    )
                    raw_dict = self._to_raw_dict(raw)
                    text = ""
                    choices = raw_dict.get("choices", []) or []
                    if choices:
                        text = choices[0].get("message", {}).get("content", "")
                    if not text:
                        last_error = f"{model}: empty completion at {base}"
                        continue
                    self._active_ollama_url = base
                    return {"model": model, "output_text": text, "raw": raw_dict}
                except Exception as exc:
                    last_error = f"{model}: {str(exc)}"
                    continue

        raise RuntimeError(f"Ollama chat failed on all endpoints/models: {last_error}")

    async def _embed_ollama(self, payload: LLMInvokeRequest) -> Dict:
        requested_model = payload.model
        requested = requested_model or settings.ollama_default_model
        use_explicit = requested_model is not None
        text = payload.text or ""
        if litellm is None:
            raise RuntimeError("litellm is not installed. Add litellm to dependencies and retry.")

        last_error = "unknown"
        model_candidates = await self._build_ollama_model_candidates(requested, explicit=use_explicit)

        for model in model_candidates:
            for base in self._ollama_url_candidates():
                try:
                    raw = await litellm.aembedding(
                        model=f"ollama/{model}",
                        input=[text],
                        api_base=base,
                        timeout=self._ollama_timeout,
                    )
                    self._active_ollama_url = base
                    raw_dict = self._to_raw_dict(raw)
                    data = raw_dict.get("data", []) or []
                    if data:
                        first = data[0] if isinstance(data[0], dict) else {}
                        emb = first.get("embedding", [])
                        if emb:
                            self._active_ollama_url = base
                            return {"model": model, "embedding": emb, "raw": raw_dict}
                        if "embedding" in raw_dict:
                            self._active_ollama_url = base
                            return {"model": model, "embedding": raw_dict.get("embedding", []), "raw": raw_dict}
                    if "data" in raw_dict and isinstance(raw_dict["data"], list) and raw_dict["data"] == []:
                        continue
                    if raw_dict.get("embedding") is not None:
                        self._active_ollama_url = base
                        return {"model": model, "embedding": raw_dict.get("embedding", []), "raw": raw_dict}
                except Exception as exc:
                    last_error = f"{model}: {str(exc)}"
                    continue

        raise RuntimeError(f"Ollama embedding failed on all endpoints/models: {last_error}")

    async def _build_ollama_model_candidates(
        self,
        requested_model: Optional[str],
        explicit: bool = False,
    ) -> list[str]:
        requested = requested_model or settings.ollama_default_model
        try:
            models = await self._list_ollama_models()
            available = [m.get("model") for m in models if m.get("available")]
            available = [str(x) for x in available if isinstance(x, str) and x.strip()]
            requested = str(requested).strip()
            if not requested:
                requested = settings.ollama_default_model

            ordered: list[str] = []
            if requested:
                ordered.append(requested)
            for model in available:
                if model not in ordered:
                    ordered.append(model)
            if not explicit and not ordered and requested:
                ordered.append(requested)
            if not ordered:
                ordered = [requested] if requested else []

            seen: set[str] = set()
            deduped = []
            for model in ordered:
                if model not in seen:
                    seen.add(model)
                    deduped.append(model)
            return deduped
        except Exception:
            return [requested]

    @staticmethod
    def _to_raw_dict(payload_obj: object) -> Dict:
        if isinstance(payload_obj, Dict):
            return payload_obj
        if hasattr(payload_obj, "dict") and callable(payload_obj.dict):
            try:
                return payload_obj.dict()
            except Exception:
                pass
        if hasattr(payload_obj, "model_dump") and callable(payload_obj.model_dump):
            try:
                return payload_obj.model_dump()
            except Exception:
                pass
        try:
            return json.loads(json.dumps(payload_obj))  # pragma: no cover - defensive
        except Exception:
            return {"raw": str(payload_obj)}

    async def list_models(self, provider: Optional[str] = None) -> List[Dict]:
        p = (provider or "").strip().lower()
        if p == "ollama":
            models = await self._list_ollama_models()
            default_model = settings.ollama_default_model
            if default_model:
                model_names = {m.get("model") for m in models if isinstance(m.get("model"), str)}
                if default_model in model_names:
                    default_entry = next(
                        (m for m in models if m.get("model") == default_model),
                        {"provider": "ollama", "model": default_model, "available": False},
                    )
                    models = [m for m in models if m.get("model") != default_model]
                    models.insert(0, default_entry)
                else:
                    models.insert(0, {"provider": "ollama", "model": default_model, "available": False})
            return models

        items: List[Dict] = [
            {"provider": "openai", "model": settings.llm_rerank_model, "available": bool(settings.llm_api_key)},
            {"provider": "openai", "model": settings.llm_embedding_model, "available": bool(settings.llm_api_key)},
            {"provider": "openai_compatible", "model": settings.llm_rerank_model, "available": True},
            {"provider": "bedrock", "model": settings.bedrock_rerank_model_id, "available": True},
            {"provider": "bedrock", "model": settings.bedrock_embedding_model_id, "available": True},
        ]
        if not p:
            items.extend(await self._list_ollama_models())
        else:
            items = [x for x in items if x["provider"] == p]
        return items

    async def _list_ollama_models(self) -> List[Dict]:
        for base in self._ollama_url_candidates():
            try:
                r = await self._client.get(f"{base}/api/tags")
                if r.status_code < 400:
                    payload = r.json()
                    out = []
                    for m in payload.get("models", []):
                        name = m.get("name")
                        if name:
                            out.append({"provider": "ollama", "model": str(name), "available": True})
                    if out:
                        self._active_ollama_url = base
                        return out
            except Exception:
                pass

            try:
                r = await self._client.get(f"{base}/v1/models")
                if r.status_code < 400:
                    payload = r.json()
                    out = []
                    for m in payload.get("data", []):
                        name = m.get("id")
                        if name:
                            out.append({"provider": "ollama", "model": str(name), "available": True})
                    if out:
                        self._active_ollama_url = base
                        return out
            except Exception:
                pass

        return [{"provider": "ollama", "model": settings.ollama_default_model, "available": False}]

    def _build_ollama_urls(self) -> List[str]:
        raw = [settings.ollama_base_url.strip(), settings.ollama_base_urls.strip()]
        values: List[str] = []
        for v in raw:
            if not v:
                continue
            for item in v.split(","):
                item = item.strip().rstrip("/")
                if item:
                    values.append(item)
        # de-duplicate preserving order
        seen = set()
        out = []
        for item in values:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        if not out:
            out = ["http://localhost:11434"]
        return out

    def _ollama_url_candidates(self) -> List[str]:
        if self._active_ollama_url in self._ollama_urls:
            return [self._active_ollama_url] + [x for x in self._ollama_urls if x != self._active_ollama_url]
        return self._ollama_urls
