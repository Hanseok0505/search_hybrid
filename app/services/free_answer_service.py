from __future__ import annotations

from typing import Callable, Optional, Tuple

try:
    import boto3
except Exception:  # pragma: no cover
    boto3 = None

from app.core.config import settings
from app.models.schemas import (
    Candidate,
    FreeAnswerRequest,
    LLMInvokeRequest,
    SearchRequest,
)


class FreeAnswerService:
    def __init__(self) -> None:
        pass

    async def close(self) -> None:
        return None

    async def answer(
        self,
        payload: FreeAnswerRequest,
        search_fn: Callable[[SearchRequest], object],
        llm_invoke: Callable[[LLMInvokeRequest], object],
    ) -> dict:
        search_request = SearchRequest(
            query=payload.query,
            top_k=payload.top_k,
            use_cache=payload.use_cache,
            top_down_context=payload.top_down_context,
            selected_source_ids=payload.selected_source_ids,
            embedded_only=payload.embedded_only,
            search_backends=payload.search_backends or ["elastic", "vector", "graph", "local"],
        )

        result = await search_fn(search_request)
        hits: list[Candidate] = result.hits if hasattr(result, "hits") else []

        if not hits:
            prompt = (
                "You are a construction-domain assistant.\n"
                "If no source documents were returned, answer with a strict safety-first template.\n\n"
                "Question: {query}\n\n"
                "System note: grounded evidence unavailable.\n\n"
                "Return in this format:\n"
                "- answer: concise guidance\n"
                "- confidence: low\n"
                "- suggested_next_steps: [1-3]\n"
            ).format(query=payload.query)
        else:
            context_lines = self._build_context_lines(hits, payload.top_k)
            prompt = (
                "You are a construction-domain assistant. "
                "Use ONLY the retrieved sources below and DO NOT use external knowledge.\n\n"
                f"Question: {payload.query}\n\n"
                "Retrieved sources:\n"
                + "\n".join(context_lines)
                + "\n\n"
                "Return a concise answer with citation markers like [1], [2] when possible."
            )

        answer, provider_used, model_used, rag, fallback_reason = await self._answer_from_context(
            payload,
            prompt,
            llm_invoke,
            hits,
        )
        return {
            "provider": provider_used,
            "query": payload.query,
            "answer": answer,
            "model": model_used,
            "hits": [self._serialize_hit(h) for h in hits[:payload.top_k]],
            "rag_synthesized": rag,
            "fallback_reason": fallback_reason,
            "source_url": None,
        }

    async def _answer_from_context(
        self,
        payload: FreeAnswerRequest,
        prompt: str,
        llm_invoke: Callable[[LLMInvokeRequest], object],
        hits: list[Candidate],
    ) -> tuple[str, str, Optional[str], bool, Optional[str]]:
        provider_order = self._build_provider_order(payload.provider)
        last_error = None

        for provider in provider_order:
            model = payload.model
            request = LLMInvokeRequest(
                task="chat",
                provider=provider,
                model=model,
                text=prompt,
                temperature=0.1,
                max_tokens=payload.max_tokens or 1024,
            )
            try:
                out = await llm_invoke(request)
                provider_used = out.get("provider") or provider
                model_used = out.get("model") or payload.model or provider
                text = (out.get("output_text") or "").strip()
                if text:
                    return text, str(provider_used), str(model_used), True, None
            except Exception as exc:
                last_error = str(exc)
                continue

        return (
            self._build_fallback_answer(hits, payload.query),
            payload.provider or settings.llm_provider or "llm",
            payload.model,
            False,
            f"llm_generation_failed: {last_error}" if last_error else "llm_generation_failed",
        )

    @staticmethod
    def _serialize_hit(hit: Candidate) -> dict:
        snippet = hit.content or ""
        if len(snippet) > 500:
            snippet = snippet[:500] + "..."
        return {
            "id": hit.id,
            "title": hit.title,
            "content": snippet,
            "source": hit.source,
            "raw_score": hit.raw_score,
            "fused_score": hit.fused_score,
            "metadata": hit.metadata,
        }

    @staticmethod
    def _build_context_lines(hits: list[Candidate], top_k: int) -> list[str]:
        out: list[str] = []
        source_budget = max(1, min(top_k, 8))
        for i, h in enumerate(hits[:source_budget], start=1):
            snippet = (h.content or "").replace("\n", " ").strip()
            if len(snippet) > 240:
                snippet = snippet[:240] + "..."
            out.append(f"[{i}] ({h.source}) {h.title or h.id}: {snippet}")
        return out

    @staticmethod
    def _build_provider_order(requested_provider: Optional[str]) -> list[str]:
        seen = set()
        providers: list[str] = []
        for provider in [requested_provider, settings.llm_provider, "ollama", "openai_compatible", "openai", "bedrock", "gemini"]:
            if not provider or provider in seen:
                continue
            if not FreeAnswerService._is_provider_enabled(provider):
                continue
            seen.add(provider)
            providers.append(provider)
        return providers

    @staticmethod
    def _is_provider_enabled(provider: str) -> bool:
        provider = (provider or "").lower()
        if provider == "openai":
            return bool(settings.llm_api_key)
        if provider == "openai_compatible":
            base = (settings.llm_base_url or "").lower()
            if "api.openai.com" in base:
                return bool(settings.llm_api_key)
            return True
        if provider == "gemini":
            return bool(settings.gemini_api_key)
        if provider == "bedrock":
            if not settings.bedrock_region:
                return False
            if boto3 is None:
                return False
            try:
                session = boto3.session.Session()
                return session.get_credentials() is not None
            except Exception:
                return False
        return provider == "ollama"

    @staticmethod
    def _build_fallback_answer(hits: list[Candidate], query: str) -> str:
        if not hits:
            return (
                f"Answer: No grounded documents were found for '{query}'.\n"
                "- confidence: low\n"
                "- suggested_next_steps:\n"
                "  1) broaden query terms\n"
                "  2) check source/context filters\n"
                "  3) upload relevant execution docs"
            )

        evidence = []
        for i, hit in enumerate(hits[:3], start=1):
            body = (hit.content or "").strip().replace("\n", " ")
            if len(body) > 220:
                body = body[:220] + "..."
            evidence.append(f"[{i}] {hit.title or hit.id}: {body}")
        return (
            f"Answer: Model synthesis was unavailable for '{query}'. "
            "Returning fallback grounded summary.\n"
            "Evidence:\n"
            + "\n".join(evidence)
        )


