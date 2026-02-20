from __future__ import annotations

import asyncio
import json
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile

from app.dependencies import Container
from app.models.schemas import (
    AskRequest,
    AskResponse,
    Candidate,
    CacheInvalidateRequest,
    CacheStatsResponse,
    DeleteSourceResponse,
    DeleteSourcesRequest,
    SourceContentResponse,
    FreeAnswerRequest,
    FreeAnswerResponse,
    HealthResponse,
    LLMInvokeRequest,
    LLMInvokeResponse,
    ModelListResponse,
    SearchRequest,
    SearchResponse,
    SourceListResponse,
    UploadResponse,
)
from app.core.config import settings

router = APIRouter(prefix="/v1", tags=["search"])


def get_container(request: Request) -> Container:
    container = getattr(request.app.state, "container", None)
    if container is None:
        container = Container()
        request.app.state.container = container
    return container


@router.post("/search", response_model=SearchResponse)
async def search(payload: SearchRequest, container: Container = Depends(get_container)) -> SearchResponse:
    return await container.hybrid.search(payload)


@router.get("/health", response_model=HealthResponse)
async def health(container: Container = Depends(get_container)) -> HealthResponse:
    elastic, milvus, graph, redis = await asyncio.gather(
        container.elastic.health(),
        container.vector.health(),
        container.graph.health(),
        container.cache.health(),
    )
    overall = "ok" if all(x in {"up", "disabled"} for x in [elastic, milvus, graph, redis]) else "degraded"
    return HealthResponse(status=overall, elastic=elastic, milvus=milvus, graph=graph, redis=redis)


@router.post("/llm/invoke", response_model=LLMInvokeResponse)
async def llm_invoke(
    payload: LLMInvokeRequest,
    container: Container = Depends(get_container),
) -> LLMInvokeResponse:
    try:
        out = await container.llm_gateway.invoke(payload)
        return LLMInvokeResponse.model_validate(out)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats(container: Container = Depends(get_container)) -> CacheStatsResponse:
    stats = await container.cache.stats()
    return CacheStatsResponse.model_validate(stats)


@router.post("/cache/invalidate")
async def cache_invalidate(
    payload: CacheInvalidateRequest,
    container: Container = Depends(get_container),
) -> dict:
    deleted = await container.cache.invalidate_by_prefix(payload.prefix)
    return {"deleted": deleted, "prefix": payload.prefix}


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    top_down_context: str = Form(default="{}"),
    already_embedded: bool = Form(default=False),
    container: Container = Depends(get_container),
) -> UploadResponse:
    try:
        parsed = {}
        if top_down_context:
            try:
                parsed = json.loads(top_down_context)
                if not isinstance(parsed, dict):
                    parsed = {}
            except Exception:
                parsed = {}
        out = await container.uploads.ingest(
            file=file,
            top_down_context=parsed,
            already_embedded=already_embedded,
        )
        return UploadResponse.model_validate(out)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/sources", response_model=SourceListResponse)
async def list_sources(container: Container = Depends(get_container)) -> SourceListResponse:
    items = container.sources.list_sources()
    return SourceListResponse(items=items, count=len(items))


@router.get("/sources/{source_id}/content", response_model=SourceContentResponse)
async def source_content(
    source_id: str,
    container: Container = Depends(get_container),
) -> SourceContentResponse:
    row = container.sources.get_source(source_id)
    if not row:
        raise HTTPException(status_code=404, detail="source not found")

    metadata = row.get("metadata", {})
    title = str(row.get("title") or row.get("id") or source_id)
    content = container.sources.read_source_content(source_id=source_id, max_chars=16000)
    source_type = str(row.get("_source_type", "upload"))
    # Provide stable content response from stored extracted content.
    can_delete = bool(row.get("_source_type") == "upload" and str(source_id).startswith("upl-"))
    return SourceContentResponse(
        id=str(source_id),
        title=title,
        source_type=source_type,
        stored_file_name=metadata.get("stored_file_name"),
        original_file_name=metadata.get("original_file_name"),
        can_delete=can_delete,
        content=content,
        content_truncated=len(content) >= 16000,
        metadata=metadata,
    )


@router.delete("/sources/{source_id}", response_model=DeleteSourceResponse)
async def delete_source(
    source_id: str,
    container: Container = Depends(get_container),
) -> DeleteSourceResponse:
    return await _delete_sources([source_id], container)


@router.post("/sources/delete", response_model=DeleteSourceResponse)
async def delete_sources(
    payload: DeleteSourcesRequest,
    container: Container = Depends(get_container),
) -> DeleteSourceResponse:
    source_ids = payload.source_ids
    if payload.delete_all_uploads:
        source_ids = [
            item.id
            for item in container.sources.list_sources()
            if item.can_delete
        ]
    return await _delete_sources(
        source_ids,
        container,
        skip_backend_cleanup=payload.skip_backend_cleanup,
    )


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    provider: Optional[str] = Query(default=None),
    container: Container = Depends(get_container),
) -> ModelListResponse:
    items = await container.llm_gateway.list_models(provider=provider)
    return ModelListResponse(items=items)


@router.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest, container: Container = Depends(get_container)) -> AskResponse:
    top_down = payload.top_down_context.model_dump(exclude_none=True) if payload.top_down_context else {}
    base_req = SearchRequest(
        query=payload.query,
        top_k=min(payload.top_k, 20),
        use_cache=payload.use_cache,
        top_down_context=payload.top_down_context,
        selected_source_ids=payload.selected_source_ids,
        embedded_only=payload.embedded_only,
        search_backends=payload.search_backends,
        rerank_with_llm=False,
    )
    result = await container.hybrid.search(base_req)
    hits = _filter_hits(
        result.hits,
        selected_ids=payload.selected_source_ids,
        embedded_only=payload.embedded_only,
        top_down_context=top_down,
    )

    if len(hits) < payload.top_k:
        local = container.sources.local_search(
            query=payload.query,
            top_k=min(payload.top_k, 20),
            selected_ids=payload.selected_source_ids,
            embedded_only=payload.embedded_only,
            top_down_filters=top_down,
        )
        hits = _merge_hits(hits, local, payload.top_k)
    if len(hits) < payload.top_k:
        relaxed = container.sources.fallback_context_docs(
            top_k=min(payload.top_k, 20),
            selected_ids=payload.selected_source_ids,
            embedded_only=payload.embedded_only,
            top_down_filters=top_down,
        )
        hits = _merge_hits(hits, relaxed, payload.top_k)
    else:
        hits = hits[: payload.top_k]
    hits = hits[: payload.top_k]

    answer, provider_used, model_used, rag_synthesized, fallback_reason = await _answer_from_hits(payload, hits, container)
    return AskResponse(
        answer=answer,
        provider=provider_used,
        model=model_used,
        hits=hits,
        rag_synthesized=rag_synthesized,
        fallback_reason=fallback_reason,
    )


@router.post("/free/answer", response_model=FreeAnswerResponse)
async def free_answer(
    payload: FreeAnswerRequest,
    container: Container = Depends(get_container),
) -> FreeAnswerResponse:
    out = await container.free_answer.answer(payload, container.hybrid.search, container.llm_gateway.invoke)
    return FreeAnswerResponse.model_validate(out)


def _filter_hits(
    hits: list[Candidate],
    selected_ids: Optional[List[str]],
    embedded_only: bool,
    top_down_context: dict,
) -> list[Candidate]:
    selected = set(selected_ids or [])
    out: list[Candidate] = []
    for h in hits:
        selected_match = bool(selected and h.id in selected)
        if selected and not selected_match:
            continue
        if embedded_only and not bool(h.metadata.get("embedded", False)):
            continue
        if selected_match:
            out.append(h)
            continue
        ok = True
        for k, v in top_down_context.items():
            if v is None:
                continue
            actual = h.metadata.get(k)
            if actual is None:
                continue
            expected = str(v)
            if str(actual) == expected or str(actual).lower() == expected.lower():
                continue
            ok = False
            break
        if ok:
            out.append(h)
    return out


async def _delete_sources(
    source_ids: List[str],
    container: Container,
    *,
    skip_backend_cleanup: bool = False,
) -> DeleteSourceResponse:
    selected = [sid.strip() for sid in source_ids if sid and sid.strip()]
    if not selected:
        return DeleteSourceResponse(deleted=False, deleted_source_ids=[], warning="No source id provided")

    deleted_local, warning = container.sources.delete_sources(selected)

    warnings: list[str] = []
    if warning:
        warnings.append(warning)

    if deleted_local and not skip_backend_cleanup:
        backend_status_raw = await asyncio.gather(
            container.elastic.health(),
            container.vector.health(),
            container.graph.health(),
            return_exceptions=True,
        )
        backend_status = {
            "elastic": backend_status_raw[0] if isinstance(backend_status_raw[0], str) else "down",
            "vector": backend_status_raw[1] if isinstance(backend_status_raw[1], str) else "down",
            "graph": backend_status_raw[2] if isinstance(backend_status_raw[2], str) else "down",
        }
        backend_enabled = {
            "elastic": backend_status["elastic"] in {"up", "degraded"},
            "vector": backend_status["vector"] in {"up", "degraded"},
            "graph": backend_status["graph"] in {"up", "degraded"},
        }
        skipped = [name for name, enabled in backend_enabled.items() if not enabled]
        if skipped:
            warnings.append(f"backend cleanup skipped: {', '.join(skipped)}")

        async def _safe_delete(doc_id: str) -> None:
            e_ok = False
            v_ok = False
            g_ok = False
            if backend_enabled["elastic"]:
                try:
                    e_ok = bool(await asyncio.wait_for(container.elastic.delete_document(doc_id), timeout=3.0))
                except Exception:
                    e_ok = False
            if backend_enabled["vector"]:
                try:
                    v_ok = bool(await asyncio.wait_for(container.vector.delete_document(doc_id), timeout=3.0))
                except Exception:
                    v_ok = False
            if backend_enabled["graph"]:
                try:
                    g_ok = bool(await asyncio.wait_for(container.graph.delete_document(doc_id), timeout=3.0))
                except Exception:
                    g_ok = False
            if not any((e_ok, v_ok, g_ok)):
                warnings.append(f"{doc_id}: not found in active backends")

        semaphore = asyncio.Semaphore(16)

        async def _delete_with_limit(doc_id: str) -> None:
            async with semaphore:
                await _safe_delete(doc_id)

        await asyncio.gather(*[_delete_with_limit(doc_id) for doc_id in deleted_local])

    if not deleted_local:
        return DeleteSourceResponse(deleted=False, deleted_source_ids=[], warning="; ".join(warnings) or "No delete targets")

    return DeleteSourceResponse(
        deleted=True,
        deleted_source_ids=deleted_local,
        warning="; ".join(warnings) if warnings else None,
    )



def _merge_hits(primary: list[Candidate], secondary: list[Candidate], top_k: int) -> list[Candidate]:
    seen = set()
    out: list[Candidate] = []
    for h in primary + secondary:
        if h.id in seen:
            continue
        seen.add(h.id)
        out.append(h)
        if len(out) >= top_k:
            break
    return out


async def _answer_from_hits(
    payload: AskRequest,
    hits: list[Candidate],
    container: Container,
) -> tuple[str, str, str, bool, Optional[str]]:
    max_tokens = payload.max_tokens
    if max_tokens is not None:
        max_tokens = min(max_tokens, 2048)

    context_lines = _build_query_context_lines(hits, payload.query, payload.top_k)
    has_sources = len(hits) > 0

    if has_sources:
        prompt = (
            "You are a construction-domain assistant.\n"
            "Use ONLY the provided sources and do NOT use outside knowledge.\n"
            "If evidence is weak, say 'insufficient evidence'.\n"
            "Include short citations like [1], [2].\n\n"
            f"Question: {payload.query}\n\n"
            "Sources:\n"
            + "\n".join(context_lines)
            + "\n\nReturn in this format:\n"
            "- Answer\n- Key evidence with citations\n- Risks/Gaps"
        )
    else:
        prompt = (
            "You are a construction-domain assistant.\n"
            "No source documents were retrieved for this request, so do not use outside knowledge.\n"
            "Use safety-first assumptions only.\n"
            f"Question: {payload.query}\n\n"
            "Return in this format:\n"
            "- Answer\n- Confidence\n- Suggested next steps"
        )

    provider_order = []
    seen = set()
    for provider in [payload.provider, settings.llm_provider, "ollama", "openai_compatible", "openai", "bedrock", "gemini"]:
        if not provider or provider in seen:
            continue
        if not _is_provider_enabled_for_generation(provider):
            continue
        seen.add(provider)
        provider_order.append(provider)

    last_error = None
    for provider in provider_order:
        try:
            out = await container.llm_gateway.invoke(
                LLMInvokeRequest(
                    task="chat",
                    provider=provider,
                    model=payload.model,
                    text=prompt,
                    temperature=0.1,
                    max_tokens=max_tokens,
                )
            )
            text = (out.get("output_text") or "").strip()
            model_used = out.get("model") or payload.model or provider
            if text:
                return text, str(provider), str(model_used), True, None
        except Exception as exc:
            last_error = str(exc)
            continue

    fallback_reason = f"llm_generation_failed: {last_error}" if last_error else "llm_generation_failed"
    provider_used = payload.provider or settings.llm_provider or "llm"
    return (
        _build_extract_answer(payload.query, hits),
        provider_used,
        payload.model,
        False,
        fallback_reason,
    )


def _build_query_context_lines(hits: list[Candidate], query: str, top_k: int) -> list[str]:
    source_budget = min(len(hits), max(1, min(8, top_k)))
    if not hits:
        return [f"[0] no matches for query: {query}"]

    lines: list[str] = []
    for i, h in enumerate(hits[:source_budget], start=1):
        body = (h.content or "").strip().replace("\n", " ")
        if len(body) > 260:
            body = body[:260] + "..."
        source = h.source or "source"
        file_name = (
            h.metadata.get("original_file_name") or h.metadata.get("stored_file_name")
        )
        if file_name:
            file_name = str(file_name)
        label = f" | file={file_name}" if file_name else ""
        lines.append(f"[{i}] ({source}) {h.title or h.id}{label}: {body}")
    return lines


def _build_extract_answer(query: str, hits: list[Candidate]) -> str:
    top = hits[:3]
    if not top:
        return (
            f"Answer: No grounded documents were found for '{query}'.\n"
            "Confidence: low\n"
            "Suggested next steps:\n"
            "1) broaden query terms\n"
            "2) remove strict top-down filters\n"
            "3) upload relevant execution documents"
        )

    evidence = []
    for i, h in enumerate(top, start=1):
        text = (h.content or "").strip().replace("\n", " ")
        if len(text) > 220:
            text = text[:220] + "..."
        evidence.append(f"- [{i}] {h.title or h.id}: {text}")

    summary = " ; ".join((h.title or h.id) for h in top)
    return (
        f"Answer: Retrieved source-based response for '{query}'. "
        f"Most relevant documents: {summary}.\n"
        "Key Evidence:\n"
        + "\n".join(evidence)
        + "\nRisks/Gaps: Additional matching documents may be required for full coverage."
    )


def _is_provider_enabled_for_generation(provider: str) -> bool:
    provider = (provider or "").lower()
    if not provider:
        return False
    if provider == "openai":
        return bool(settings.llm_api_key)
    if provider == "openai_compatible":
        base = (settings.llm_base_url or "").lower()
        is_openai_public = "api.openai.com" in base
        if is_openai_public:
            return bool(settings.llm_api_key)
        return True
    if provider == "gemini":
        return bool(settings.gemini_api_key)
    if provider == "bedrock":
        if not settings.bedrock_region:
            return False
        try:
            import boto3  # type: ignore

            session = boto3.session.Session()
            return session.get_credentials() is not None
        except Exception:
            return False
    return provider == "ollama"
