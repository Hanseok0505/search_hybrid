from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import json

import httpx

from app.core.config import settings
from app.domain.construction import _ollama_model_aliases
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
    _OLLAMA_CHAT_FALLBACKS: tuple[str, ...] = (
        "gpt-oss-120b-cloud",
        "llama3.1:8b",
        "qwen2.5:7b",
        "gemma3:12b",
        "phi4",
    )
    _OLLAMA_EMBED_FALLBACKS: tuple[str, ...] = (
        "nomic-embed-text",
        "mxbai-embed-large",
        "all-minilm",
        "bge-base-en-v1.5",
        "bge-small-en-v1.5",
    )

    def __init__(self) -> None:
        self._default_provider = settings.llm_provider.lower()
        self._llm_base_url = settings.llm_base_url.rstrip("/")
        self._llm_api_key = settings.llm_api_key
        self._chat_path = settings.llm_chat_path
        self._embed_path = settings.llm_embeddings_path
        self._auth_header = settings.llm_auth_header
        self._auth_prefix = settings.llm_auth_prefix
        self._timeout = settings.llm_timeout_sec
        self._ollama_timeout = max(settings.llm_timeout_sec, settings.ollama_timeout_sec)
        self._ollama_fast_timeout = min(max(8.0, float(settings.ollama_timeout_sec)), 20.0)
        self._ollama_probe_timeout = min(max(2.0, float(settings.ollama_timeout_sec) * 0.2), 6.0)
        self._ollama_model_budget = 5
        self._client = httpx.AsyncClient(timeout=self._timeout)

        self._ollama_urls = self._collect_ollama_urls()
        self._active_ollama_url = self._ollama_urls[0] if self._ollama_urls else "http://localhost:11434"
        self._ollama_cache: dict[str, list[str]] = {}

    async def close(self) -> None:
        await self._client.aclose()

    async def invoke(self, payload: LLMInvokeRequest) -> Dict[str, Any]:
        provider = (payload.provider or self._default_provider).strip().lower()
        task = (payload.task or "").strip().lower()
        if task not in {"chat", "embed"}:
            raise ValueError("task must be one of: chat, embed")

        if provider == "openai":
            if task == "chat":
                out = await self._chat_openai(payload, base_url=self._llm_base_url, api_key=self._llm_api_key, require_key=True)
            else:
                out = await self._embed_openai(payload)
        elif provider == "openai_compatible":
            if task == "chat":
                out = await self._chat_openai(payload, base_url=self._llm_base_url, api_key=self._llm_api_key, require_key=False)
            else:
                out = await self._embed_openai(payload)
        elif provider == "ollama":
            if task == "chat":
                out = await self._chat_ollama(payload)
            else:
                out = await self._embed_ollama(payload)
        elif provider == "bedrock":
            if task == "chat":
                out = await self._chat_bedrock(payload)
            else:
                out = await self._embed_bedrock(payload)
        elif provider == "gemini":
            if task == "chat":
                out = await self._chat_gemini(payload)
            else:
                raise ValueError("gemini provider does not support embed task in this service path")
        else:
            raise ValueError("provider must be one of: openai, openai_compatible, ollama, bedrock, gemini")

        out["provider"] = provider
        out["task"] = task
        return out

    async def list_models(self, provider: Optional[str] = None) -> List[Dict[str, object]]:
        selected = (provider or "").strip().lower()

        if selected == "ollama":
            return await self._list_ollama_models()
        if selected == "openai":
            return await self._list_openai_models(self._llm_base_url, self._llm_api_key, "openai")
        if selected == "openai_compatible":
            return await self._list_openai_models(self._llm_base_url, self._llm_api_key, "openai_compatible")
        if selected == "bedrock":
            return self._list_bedrock_models()
        if selected == "gemini":
            return await self._list_gemini_models()

        items: List[Dict[str, object]] = []
        items.extend(await self._list_openai_models(self._llm_base_url, self._llm_api_key, "openai"))
        items.extend(await self._list_openai_models(self._llm_base_url, self._llm_api_key, "openai_compatible"))
        items.extend(self._list_bedrock_models())
        items.extend(await self._list_gemini_models())
        items.extend(await self._list_ollama_models())
        return items

    async def _chat_openai(
        self,
        payload: LLMInvokeRequest,
        *,
        base_url: str,
        api_key: Optional[str],
        require_key: bool,
    ) -> Dict[str, object]:
        if require_key and not api_key:
            raise RuntimeError("Missing LLM API key")
        model = payload.model or settings.llm_rerank_model
        messages = self._messages_from_payload(payload)
        body: Dict[str, object] = {
            "model": model,
            "messages": messages,
            "temperature": payload.temperature,
        }
        if payload.max_tokens:
            body["max_tokens"] = payload.max_tokens
        headers = self._build_headers(api_key)
        r = await self._client.post(f"{base_url}{self._chat_path}", headers=headers, json=body)
        r.raise_for_status()
        raw = r.json()
        text = self._extract_openai_text(raw)
        return {"model": model, "output_text": text, "raw": raw}

    async def _embed_openai(self, payload: LLMInvokeRequest) -> Dict[str, object]:
        model = payload.model or settings.llm_embedding_model
        body: Dict[str, object] = {
            "model": model,
            "input": payload.text or "",
        }
        headers = self._build_headers(self._llm_api_key)
        r = await self._client.post(f"{self._llm_base_url}{self._embed_path}", headers=headers, json=body)
        r.raise_for_status()
        raw = r.json()
        emb = []
        try:
            emb_data = raw.get("data", [])
            if emb_data:
                first = emb_data[0]
                if isinstance(first, dict):
                    emb = first.get("embedding", [])
        except Exception:
            emb = []
        return {"model": model, "embedding": emb, "raw": raw}

    async def _chat_ollama(self, payload: LLMInvokeRequest) -> Dict[str, object]:
        requested = payload.model or settings.ollama_default_model
        candidates = await self._ollama_model_candidates(requested, purpose="chat")
        messages = self._messages_from_payload(payload)

        last_error = None
        for base in self._iter_ollama_urls():
            if not base:
                continue

            available = await self._fetch_ollama_models(base)
            ordered_models = [m for m in candidates if m in available]
            if requested and requested in available and requested not in ordered_models:
                ordered_models.insert(0, requested)
            if not ordered_models:
                # If this endpoint is reachable but has no detected models, fall back to configured candidates.
                ordered_models = candidates[: self._ollama_model_budget]

            for model in ordered_models[: self._ollama_model_budget]:
                text: str | None = None
                if litellm is not None:
                    try:
                        raw = await litellm.acompletion(
                            model=f"ollama/{model}",
                            messages=messages,
                            api_base=base,
                            temperature=payload.temperature,
                            stream=False,
                            max_tokens=payload.max_tokens or 1024,
                            timeout=min(self._ollama_fast_timeout, 10.0),
                        )
                        raw_dict = self._to_raw_dict(raw)
                        text = self._extract_litellm_text(raw_dict)
                        if text:
                            self._active_ollama_url = base
                            return {"model": model, "output_text": text, "raw": raw_dict}
                    except Exception as exc:  # pragma: no cover
                        last_error = str(exc)

                for endpoint, body in self._ollama_payload_pairs(model, messages, payload):
                    try:
                        direct = await self._client.post(
                            f"{base}/{endpoint}",
                            json=body,
                            timeout=min(self._ollama_fast_timeout, 10.0),
                        )
                        if direct.status_code < 400:
                            direct_json = direct.json()
                            text = (
                                self._extract_ollama_chat_text(direct_json)
                                if endpoint == "api/chat"
                                else self._extract_ollama_generate_text(direct_json)
                            )
                            if text:
                                self._active_ollama_url = base
                                return {"model": model, "output_text": text, "raw": direct_json}
                    except Exception as exc:  # pragma: no cover
                        last_error = str(exc)

        raise RuntimeError(f"Ollama chat failed on all endpoints/models: {last_error}")

    async def _embed_ollama(self, payload: LLMInvokeRequest) -> Dict[str, object]:
        requested = payload.model or settings.ollama_default_model
        candidates = await self._ollama_model_candidates(requested, purpose="embed")
        text = payload.text or ""
        if not text:
            return {"model": requested, "embedding": [], "raw": {}}

        last_error = None
        for model in candidates[: self._ollama_model_budget]:
            for base in self._iter_ollama_urls():
                if not base:
                    continue

                available = await self._fetch_ollama_models(base)
                if available and model not in available:
                    continue

                if litellm is not None:
                    try:
                        raw = await litellm.aembedding(
                            model=f"ollama/{model}",
                            input=[text],
                            api_base=base,
                            timeout=min(self._ollama_fast_timeout, 10.0),
                        )
                        raw_dict = self._to_raw_dict(raw)
                        emb = self._extract_litellm_embedding(raw_dict)
                        if emb:
                            self._active_ollama_url = base
                            return {"model": model, "embedding": emb, "raw": raw_dict}
                    except Exception as exc:  # pragma: no cover
                        last_error = str(exc)

                # Direct API fallback.
                try:
                    direct = await self._client.post(
                        f"{base}/api/embeddings",
                        json={"model": model, "prompt": text},
                        timeout=min(self._ollama_fast_timeout, 10.0),
                    )
                    if direct.status_code < 400:
                        direct_json = direct.json()
                        emb = self._extract_ollama_embed_array(direct_json)
                        if emb:
                            self._active_ollama_url = base
                            return {"model": model, "embedding": emb, "raw": direct_json}
                    # fallback for older/alternate Ollama embedding route variants
                    legacy = await self._client.post(
                        f"{base}/api/embedding",
                        json={"model": model, "text": text},
                        timeout=min(self._ollama_fast_timeout, 10.0),
                    )
                    if legacy.status_code < 400:
                        legacy_json = legacy.json()
                        emb = self._extract_ollama_embed_array(legacy_json)
                        if emb:
                            self._active_ollama_url = base
                            return {"model": model, "embedding": emb, "raw": legacy_json}
                except Exception as exc:  # pragma: no cover
                    last_error = str(exc)
                    continue

        raise RuntimeError(f"Ollama embedding failed on all endpoints/models: {last_error}")

    async def _chat_bedrock(self, payload: LLMInvokeRequest) -> Dict[str, object]:
        if boto3 is None:
            raise RuntimeError("boto3 is not installed")
        client = boto3.client("bedrock-runtime", region_name=payload.region or settings.bedrock_region)
        model = payload.model or settings.bedrock_rerank_model_id
        messages = self._messages_from_payload(payload)

        response = await _to_thread_sync(
            client.converse,
            modelId=model,
            messages=[{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages],
            inferenceConfig={"temperature": payload.temperature},
        )
        text = self._extract_bedrock_text(response)
        return {"model": model, "output_text": text, "raw": response}

    async def _embed_bedrock(self, payload: LLMInvokeRequest) -> Dict[str, object]:
        if boto3 is None:
            raise RuntimeError("boto3 is not installed")
        client = boto3.client("bedrock-runtime", region_name=payload.region or settings.bedrock_region)
        model = payload.model or settings.bedrock_embedding_model_id
        req_body = {"inputText": payload.text or ""}
        response = await _to_thread_sync(
            client.invoke_model,
            modelId=model,
            body=json.dumps(req_body),
            contentType="application/json",
            accept="application/json",
        )
        parsed = json.loads(response["body"].read().decode("utf-8"))
        emb = parsed.get("embedding") or (parsed.get("embeddings", [[]])[0] if parsed.get("embeddings") else [])
        if not isinstance(emb, list):
            emb = []
        return {"model": model, "embedding": emb, "raw": parsed}

    async def _chat_gemini(self, payload: LLMInvokeRequest) -> Dict[str, object]:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is missing")
        model = payload.model or settings.gemini_default_model
        messages = self._messages_from_payload(payload)
        prompt = " ".join(part.get("content", "") for part in messages if part.get("content"))
        url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
        headers = {"x-goog-api-key": settings.gemini_api_key}
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": payload.temperature,
                "maxOutputTokens": payload.max_tokens or 1024,
            },
        }
        r = await self._client.post(url, headers=headers, json=body, timeout=self._timeout)
        r.raise_for_status()
        raw = r.json()
        text = self._extract_gemini_text(raw)
        return {"model": model, "output_text": text, "raw": raw}

    async def _list_openai_models(self, base_url: str, api_key: Optional[str], provider: str) -> List[Dict[str, object]]:
        fallback = [
            {"provider": provider, "model": settings.llm_rerank_model, "available": bool(api_key)},
            {"provider": provider, "model": settings.llm_embedding_model, "available": bool(api_key)},
        ]
        try:
            headers = self._build_headers(api_key)
            r = await self._client.get(f"{base_url}/models", headers=headers)
            if r.status_code >= 400:
                return fallback
            payload = r.json()
            items = payload.get("data", [])
            out: List[Dict[str, object]] = []
            for item in items:
                model = item.get("id")
                if not model:
                    continue
                out.append({"provider": provider, "model": str(model), "available": True})
            if out:
                return out
        except Exception:
            return fallback
        return fallback

    async def _list_ollama_models(self) -> List[Dict[str, object]]:
        for base in self._iter_ollama_urls():
            try:
                r = await self._client.get(f"{base}/api/tags", timeout=self._ollama_probe_timeout)
                if r.status_code < 400:
                    payload = r.json()
                    models = []
                    discovered_names: list[str] = []
                    for item in payload.get("models", []):
                        name = item.get("name")
                        if not name:
                            continue
                        model_name = str(name)
                        discovered_names.append(model_name)
                        models.append({"provider": "ollama", "model": model_name, "available": True})
                    if models:
                        default = settings.ollama_default_model
                        default_aliases = _ollama_model_aliases(default) if default else []
                        default_aliases_lower = {alias.lower() for alias in default_aliases}
                        has_default_alias = any(
                            discovered in default_aliases or discovered.lower() in default_aliases_lower
                            for discovered in discovered_names
                        )
                        if default and not has_default_alias and all(m["model"] != default for m in models):
                            models.insert(0, {"provider": "ollama", "model": default, "available": False})
                        return models
            except Exception:
                pass
        return [{"provider": "ollama", "model": settings.ollama_default_model, "available": False}]

    def _list_bedrock_models(self) -> List[Dict[str, object]]:
        if boto3 is None or not settings.bedrock_region:
            return [
                {"provider": "bedrock", "model": settings.bedrock_rerank_model_id, "available": False},
                {"provider": "bedrock", "model": settings.bedrock_embedding_model_id, "available": False},
            ]
        try:
            client = boto3.client("bedrock", region_name=settings.bedrock_region)
            response = client.list_foundation_models()
            seen = set()
            out: List[Dict[str, object]] = []
            for model in response.get("modelSummaries", []):
                model_id = model.get("modelId") or model.get("modelId")
                if not model_id:
                    continue
                model_id_s = str(model_id)
                if model_id_s in seen:
                    continue
                seen.add(model_id_s)
                out.append({"provider": "bedrock", "model": model_id_s, "available": True})
            if out:
                return out
        except Exception:
            pass
        return [
            {"provider": "bedrock", "model": settings.bedrock_rerank_model_id, "available": False},
            {"provider": "bedrock", "model": settings.bedrock_embedding_model_id, "available": False},
        ]

    async def _list_gemini_models(self) -> List[Dict[str, object]]:
        if not settings.gemini_api_key:
            return [{"provider": "gemini", "model": settings.gemini_default_model, "available": False}]
        try:
            r = await self._client.get(
                "https://generativelanguage.googleapis.com/v1/models",
                headers={"x-goog-api-key": settings.gemini_api_key},
            )
            if r.status_code >= 400:
                return [{"provider": "gemini", "model": settings.gemini_default_model, "available": False}]
            payload = r.json()
            items = []
            for item in payload.get("models", []):
                name = item.get("name")
                if not name:
                    continue
                items.append({"provider": "gemini", "model": str(name.split("/")[-1]), "available": True})
            if items:
                return items
        except Exception:
            pass
        return [{"provider": "gemini", "model": settings.gemini_default_model, "available": False}]

    def _collect_ollama_urls(self) -> List[str]:
        raw: List[str] = []
        for value in (settings.ollama_base_url.strip(), settings.ollama_base_urls.strip()):
            if not value:
                continue
            for item in value.split(","):
                candidate = item.strip().rstrip("/")
                if candidate and candidate not in raw:
                    raw.append(candidate)
        for default in (
            "http://host.docker.internal:11434",
            "http://ollama:11434",
            "http://127.0.0.1:11434",
            "http://localhost:11434",
        ):
            if default not in raw:
                raw.append(default)
        return raw

    async def _ollama_model_candidates(self, requested: str, purpose: str = "chat") -> List[str]:
        requested = (requested or "").strip()
        if not requested:
            requested = settings.ollama_default_model
        requested_variants = _ollama_model_aliases(requested)
        is_embed = purpose in {"embed", "embedding"}

        available: List[str] = []
        for base in self._iter_ollama_urls():
            names = await self._fetch_ollama_models(base)
            for name in names:
                if name not in available:
                    available.append(name)

        ordered: List[str] = []
        ordered.extend(requested_variants)
        ordered.extend(self._OLLAMA_EMBED_FALLBACKS if is_embed else self._OLLAMA_CHAT_FALLBACKS)
        for model in available:
            for alias in _ollama_model_aliases(model):
                if alias not in ordered:
                    ordered.append(alias)
        for model in available:
            if model not in ordered:
                ordered.append(model)

        if requested and requested not in ordered:
            ordered.append(requested)

        deduped: List[str] = []
        seen = set()
        for value in ordered:
            if not value or value in seen:
                continue
            seen.add(value)
            deduped.append(value)

        return deduped or [settings.ollama_default_model]

    async def _fetch_ollama_models(self, base: str) -> List[str]:
        cached = self._ollama_cache.get(base)
        if cached is not None:
            return list(cached)

        names: List[str] = []
        try:
            r = await self._client.get(f"{base}/api/tags", timeout=self._ollama_probe_timeout)
            if r.status_code < 400:
                payload = r.json()
                for item in payload.get("models", []):
                    value = item.get("name")
                    if isinstance(value, str) and value.strip():
                        names.append(value.strip())
        except Exception:
            names = []

        self._ollama_cache[base] = names
        return names

    def _iter_ollama_urls(self) -> List[str]:
        if self._active_ollama_url in self._ollama_urls:
            return [self._active_ollama_url] + [x for x in self._ollama_urls if x != self._active_ollama_url]
        return list(self._ollama_urls)

    @staticmethod
    def _ollama_payload_pairs(
        model: str,
        messages: List[Dict[str, str]],
        payload: LLMInvokeRequest,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        prompt = LLMGateway._compose_prompt_from_messages(messages)
        options: Dict[str, Any] = {
            "temperature": payload.temperature,
        }
        if payload.max_tokens:
            options["num_predict"] = payload.max_tokens
        return [
            (
                "api/chat",
                {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": options,
                },
            ),
            (
                "api/generate",
                {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": options,
                    "raw": False,
                },
            ),
        ]

    @staticmethod
    def _messages_from_payload(payload: LLMInvokeRequest) -> List[Dict[str, str]]:
        if payload.messages:
            return [{"role": m.role, "content": m.content} for m in payload.messages]
        return [{"role": "user", "content": payload.text or ""}]

    def _build_headers(self, api_key: Optional[str]) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers[self._auth_header] = f"{self._auth_prefix}{api_key}"
        return headers

    @staticmethod
    def _extract_openai_text(raw: Dict[str, Any]) -> str:
        choices = raw.get("choices", [])
        if not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        message = first.get("message")
        if isinstance(message, dict):
            return str(message.get("content", "") or "")
        return str(first.get("text", "") or "")

    @staticmethod
    def _extract_litellm_text(raw: Dict[str, Any]) -> str:
        choices = raw.get("choices", [])
        if not choices or not isinstance(choices[0], dict):
            return ""
        message = choices[0].get("message")
        if isinstance(message, dict):
            return str(message.get("content", "") or "")
        return str(choices[0].get("text", "") or "")

    @staticmethod
    def _extract_bedrock_text(raw: Dict[str, Any]) -> str:
        message = raw.get("output", {}).get("message", {}).get("content", [])
        if not isinstance(message, list):
            return ""
        return "".join(str(part.get("text", "")) for part in message if isinstance(part, dict))

    @staticmethod
    def _extract_gemini_text(raw: Dict[str, Any]) -> str:
        candidates = raw.get("candidates", [])
        if not candidates:
            return ""
        candidate0 = candidates[0]
        if not isinstance(candidate0, dict):
            return ""
        content = candidate0.get("content", {})
        parts = content.get("parts", [])
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        return text

    @staticmethod
    def _extract_litellm_embedding(raw: Dict[str, Any]) -> List[float]:
        data = raw.get("data", [])
        if not isinstance(data, list) or not data:
            return []
        first = data[0]
        if isinstance(first, dict):
            emb = first.get("embedding")
            if isinstance(emb, list):
                return emb
        if isinstance(raw.get("embedding"), list):
            return raw["embedding"]
        return []

    @staticmethod
    def _extract_ollama_chat_text(payload: Dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            return ""
        message = payload.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        response = payload.get("response")
        if isinstance(response, str) and response.strip():
            return response.strip()
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                text = first.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        return ""

    @staticmethod
    def _extract_ollama_embed_array(payload: Dict[str, Any]) -> List[float]:
        if not isinstance(payload, dict):
            return []
        embedding = payload.get("embedding")
        if isinstance(embedding, list):
            return [float(value) for value in embedding if isinstance(value, (int, float))]
        data = payload.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            embedding = data[0].get("embedding")
            if isinstance(embedding, list):
                return [float(value) for value in embedding if isinstance(value, (int, float))]
        return []

    @staticmethod
    def _extract_ollama_generate_text(payload: Dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            return ""
        response = payload.get("response")
        if isinstance(response, str) and response.strip():
            return response.strip()
        message = payload.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        return ""

    @staticmethod
    def _compose_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
        if not messages:
            return ""
        parts = []
        for item in messages:
            role = (item.get("role") or "").strip()
            content = (item.get("content") or "").strip()
            if not content:
                continue
            if role:
                parts.append(f"{role}: {content}")
            else:
                parts.append(content)
        return "\n".join(parts)

    @staticmethod
    def _to_raw_dict(payload_obj: object) -> Dict[str, Any]:
        if isinstance(payload_obj, dict):
            return payload_obj
        if hasattr(payload_obj, "dict") and callable(payload_obj.dict):
            try:
                return payload_obj.dict()  # type: ignore[attr-defined]
            except Exception:
                pass
        if hasattr(payload_obj, "model_dump") and callable(payload_obj.model_dump):
            try:
                return payload_obj.model_dump()  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            return json.loads(json.dumps(payload_obj))
        except Exception:
            return {"raw": str(payload_obj)}


def _to_thread_sync(func, *args, **kwargs):  # pragma: no cover - thin sync wrapper
    import asyncio

    return asyncio.get_running_loop().run_in_executor(None, lambda: func(*args, **kwargs))
