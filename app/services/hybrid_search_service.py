from __future__ import annotations

import asyncio
import logging
import time

from app.clients.elasticsearch_client import ElasticsearchClient
from app.clients.graph_client import GraphSearchClient
from app.clients.milvus_client import MilvusVectorClient
from app.core.config import settings
from app.domain.construction import merge_top_down_filters
from app.models.schemas import SearchRequest, SearchResponse
from app.services.cache import SearchCache
from app.services.embedding_service import EmbeddingService
from app.services.llm_reranker import LLMReranker
from app.services.ranking import weighted_reciprocal_rank_fusion
from app.services.source_service import SourceService
from app.utils.compat import to_thread


_LOGGER = logging.getLogger(__name__)


class HybridSearchService:
    def __init__(
        self,
        elastic: ElasticsearchClient,
        vector: MilvusVectorClient,
        graph: GraphSearchClient,
        cache: SearchCache,
        embedding: EmbeddingService,
        reranker: LLMReranker,
        source_service: SourceService,
    ) -> None:
        self.elastic = elastic
        self.vector = vector
        self.graph = graph
        self.cache = cache
        self.embedding = embedding
        self.reranker = reranker
        self.source_service = source_service

    @staticmethod
    def _normalize_backends(requested: list[str] | None) -> list[str]:
        requested_set = [
            (item or "").strip().lower()
            for item in (requested or [])
            if isinstance(item, str)
        ]
        selected = [item for item in requested_set if item in {"elastic", "vector", "graph", "local"}]
        return sorted(set(selected)) or ["elastic", "vector", "graph", "local"]

    async def search(self, request: SearchRequest) -> SearchResponse:
        start = time.perf_counter()
        merged_filters = merge_top_down_filters(
            request.filters or {},
            request.top_down_context.model_dump(exclude_none=True) if request.top_down_context else {},
        )
        selected_ids = None if not request.selected_source_ids else sorted(request.selected_source_ids)
        backends = self._normalize_backends(request.search_backends)

        cache_key = self.cache.make_key(
            {
                "q": request.query,
                "top_k": request.top_k,
                "filters": merged_filters,
                "selected_source_ids": selected_ids,
                "embedded_only": request.embedded_only,
                "rerank": request.rerank_with_llm,
                "search_backends": backends,
            }
        )

        if request.use_cache:
            cached = await self.cache.get_json(cache_key)
            if cached:
                cached["cache_hit"] = True
                return SearchResponse.model_validate(cached)
            lock_token = await self.cache.acquire_lock(cache_key)
            if lock_token is None:
                waited = await self.cache.wait_for_value(cache_key)
                if waited:
                    waited["cache_hit"] = True
                    return SearchResponse.model_validate(waited)
        else:
            lock_token = None

        try:
            needs_embedding = "vector" in backends
            embedding: list[float] | None = None
            if needs_embedding:
                embedding = await self.embedding.embed(request.query)

            tasks: dict[str, asyncio.Task] = {}
            if "elastic" in backends:
                tasks["elastic"] = asyncio.create_task(self.elastic.search(request.query, request.top_k, merged_filters))
            if "vector" in backends:
                if embedding is None:
                    embedding = await self.embedding.embed(request.query)
                tasks["vector"] = asyncio.create_task(
                    self.vector.search(embedding, request.top_k, merged_filters, query_text=request.query)
                )
            if "graph" in backends:
                tasks["graph"] = asyncio.create_task(self.graph.search(request.query, request.top_k, merged_filters))
            if "local" in backends:
                tasks["local"] = asyncio.create_task(
                    to_thread(
                        self.source_service.local_search,
                        request.query,
                        request.top_k,
                        selected_ids=request.selected_source_ids,
                        embedded_only=request.embedded_only,
                        top_down_filters=merged_filters,
                    )
                )

            ranked_lists = {}
            if tasks:
                results = await asyncio.gather(*tasks.values(), return_exceptions=True)
                for name, result in zip(tasks.keys(), results):
                    if isinstance(result, Exception):
                        _LOGGER.warning("search backend %s failed: %s", name, result)
                        continue
                    ranked_lists[name] = result

                fused = weighted_reciprocal_rank_fusion(
                    ranked_lists=ranked_lists,
                    weights={
                        "elastic": settings.weight_elastic if "elastic" in ranked_lists else 0.0,
                        "vector": settings.weight_vector if "vector" in ranked_lists else 0.0,
                        "graph": settings.weight_graph if "graph" in ranked_lists else 0.0,
                        "local": settings.weight_local if "local" in ranked_lists else 0.0,
                    },
                )
            else:
                fused = []

            if request.rerank_with_llm:
                fused = await self.reranker.rerank(request.query, fused, request.top_k)
            else:
                fused = fused[: request.top_k]

            took_ms = int((time.perf_counter() - start) * 1000)
            response = SearchResponse(query=request.query, took_ms=took_ms, hits=fused, cache_hit=False)

            if request.use_cache:
                await self.cache.set_json(cache_key, response.model_dump())
            return response
        finally:
            if request.use_cache and lock_token:
                await self.cache.release_lock(cache_key, lock_token)
