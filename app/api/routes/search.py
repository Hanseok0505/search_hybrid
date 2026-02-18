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
    FreeAnswerRequest,
    FreeAnswerResponse,
    HealthResponse,
    ModelListResponse,
    SearchResponse,
    LLMInvokeRequest,
    LLMInvokeResponse,
    SearchRequest,
    SourceListResponse,
    UploadResponse,
)

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
                # Accept malformed context as empty for better UX on multipart form clients.
                parsed = {}
        out = await container.uploads.ingest(
            file=file,
            top_down_context=parsed,
            already_embedded=already_embedded,
        )
        return UploadResponse.model_validate(out)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/free/answer", response_model=FreeAnswerResponse)
async def free_answer(
    payload: FreeAnswerRequest,
    container: Container = Depends(get_container),
) -> FreeAnswerResponse:
    out = await container.free_answer.answer(payload.query)
    return FreeAnswerResponse.model_validate(out)


@router.get("/sources", response_model=SourceListResponse)
async def list_sources(container: Container = Depends(get_container)) -> SourceListResponse:
    items = container.sources.list_sources()
    return SourceListResponse(items=items, count=len(items))


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

    answer, model_used, rag_synthesized, fallback_reason = await _answer_from_hits(payload, hits, container)
    return AskResponse(
        answer=answer,
        provider=payload.provider,
        model=model_used,
        hits=hits,
        rag_synthesized=rag_synthesized,
        fallback_reason=fallback_reason,
    )


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
            # Soft match: if key is missing in candidate metadata, do not exclude it.
            if k in h.metadata and h.metadata.get(k) != v:
                ok = False
                break
        if ok:
            out.append(h)
    return out


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
) -> tuple[str, str, bool, Optional[str]]:
    if not hits:
        return (
            "No matching sources were found. Try broadening filters or unchecking embedded-only.",
            payload.model or "",
            False,
            "no_matching_sources",
        )

    max_tokens = payload.max_tokens
    if max_tokens is not None:
        max_tokens = min(max_tokens, 2048)

    context_lines = []
    source_budget = min(len(hits), max(1, min(4, payload.top_k)))
    for i, h in enumerate(hits[:source_budget], start=1):
        body = (h.content or "").strip().replace("\n", " ")
        context_lines.append(f"[{i}] {h.title or h.id}: {body[:220]}")
    prompt = (
        "You are a construction-domain assistant.\n"
        "Use ONLY the provided sources and do not use outside knowledge.\n"
        "If evidence is weak, say 'insufficient evidence'.\n"
        "You must include source citations like [1], [2] in every major claim.\n\n"
        f"Question: {payload.query}\n\n"
        "Sources:\n"
        + "\n".join(context_lines)
        + "\n\nReturn in this format:\n"
        "1) Answer\n2) Key Evidence bullets with citations\n3) Risks/Gaps"
    )

    try:
        out = await container.llm_gateway.invoke(
            LLMInvokeRequest(
                task="chat",
                provider=payload.provider,
                model=payload.model,
                text=prompt,
                temperature=0.1,
                max_tokens=max_tokens,
            )
        )
        text = (out.get("output_text") or "").strip()
        model_used = out.get("model") or (payload.model or payload.provider)
        if not text:
            return _build_extract_answer(payload.query, hits), model_used, False, "llm_empty_output"
        return text, model_used, True, None
    except Exception as exc:
        return (
            _build_extract_answer(payload.query, hits),
            payload.model or payload.provider,
            False,
            f"llm_generation_failed: {exc}",
        )


def _build_extract_answer(query: str, hits: list[Candidate]) -> str:
    top = hits[:3]
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





