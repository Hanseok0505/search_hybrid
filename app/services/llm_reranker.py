from __future__ import annotations
import asyncio
import json
from app.utils.compat import to_thread

import httpx
try:
    import boto3
except Exception:  # pragma: no cover
    boto3 = None

from app.core.config import settings
from app.domain.construction import build_construction_rerank_prompt
from app.models.schemas import Candidate


class LLMReranker:
    def __init__(self) -> None:
        self._provider = settings.llm_provider.lower()
        self._enabled = settings.llm_rerank_enabled and (
            bool(settings.llm_api_key) if self._provider == "openai" else self._provider == "bedrock"
        )
        self._api_key = settings.llm_api_key
        self._base_url = settings.llm_base_url.rstrip("/")
        self._model = settings.llm_rerank_model
        self._bedrock_model_id = settings.bedrock_rerank_model_id
        self._bedrock_region = settings.bedrock_region
        self._timeout = settings.llm_timeout_sec
        self._client = httpx.AsyncClient(timeout=self._timeout)
        self._bedrock = (
            boto3.client("bedrock-runtime", region_name=self._bedrock_region)
            if self._provider == "bedrock" and boto3 is not None
            else None
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def rerank(self, query: str, candidates: list[Candidate], top_k: int) -> list[Candidate]:
        if not self._enabled or not candidates:
            return candidates[:top_k]

        docs = [
            {
                "id": c.id,
                "title": c.title,
                "content": c.content,
                "metadata": c.metadata,
            }
            for c in candidates[:50]
        ]
        prompt = build_construction_rerank_prompt(query=query, docs=docs)
        try:
            if self._provider == "bedrock" and self._bedrock:
                content = await to_thread(self._rerank_bedrock, prompt)
            else:
                headers = {"Authorization": f"Bearer {self._api_key}"}
                body = {
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                }
                response = await self._client.post(
                    f"{self._base_url}/chat/completions", headers=headers, json=body
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            ordered = parsed.get("ids", [])
            pos = {cid: i for i, cid in enumerate(ordered)}
            candidates.sort(key=lambda c: pos.get(c.id, 10_000))
            return candidates[:top_k]
        except Exception:
            return candidates[:top_k]

    def _rerank_bedrock(self, prompt: str) -> str:
        assert self._bedrock is not None
        response = self._bedrock.converse(
            modelId=self._bedrock_model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0.0},
        )
        parts = response.get("output", {}).get("message", {}).get("content", [])
        text = "".join(p.get("text", "") for p in parts)
        return text




