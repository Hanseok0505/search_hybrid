from __future__ import annotations

import asyncio
import json
from typing import Any, List

from app.core.config import settings
from app.models.schemas import SearchRequest
from app.services.cache import SearchCache
from app.services.hybrid_search_service import HybridSearchService
from app.services.llm_reranker import LLMReranker
from app.services.source_service import SourceService
from app.services.embedding_service import EmbeddingService


class _NoopCache(SearchCache):
    def __init__(self) -> None:
        super().__init__()
        self._enabled = False


class _LocalOnlyClient:
    def __init__(self, source: str) -> None:
        self._source = source

    async def search(self, *args: Any, **kwargs: Any) -> List:
        return []

    async def close(self) -> None:
        return None


class _DryEmbedder(EmbeddingService):
    async def embed(self, text: str) -> list[float]:
        # keep deterministic tiny vector for service signature; content is not used by local-only search fallback.
        return [0.0]


def test_search_includes_local_documents_when_backends_empty(tmp_path, monkeypatch):
    doc = [
        {
            "id": "upl-local-test",
            "title": "Local Search Probe",
            "content": "Scope and method statement for tower b2 concrete pour readiness checks.",
            "metadata": {
                "project_id": "PROJ-SMART-CAMPUS",
                "building": "TOWER-A",
                "level": "B2",
                "work_type": "concrete",
            },
            "related_ids": [],
        }
    ]

    uploads_path = tmp_path / "uploaded_docs.json"
    uploads_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(settings, "sample_data_path", str(tmp_path / "empty.json"))
    monkeypatch.setattr(settings, "uploads_data_path", str(uploads_path))

    service = HybridSearchService(
        elastic=_LocalOnlyClient("elastic"),
        vector=_LocalOnlyClient("vector"),
        graph=_LocalOnlyClient("graph"),
        cache=_NoopCache(),
        embedding=_DryEmbedder(),
        reranker=LLMReranker(),
        source_service=SourceService(),
    )

    req = SearchRequest(
        query="tower b2 concrete pour readiness",
        top_k=5,
        use_cache=False,
        top_down_context={"project_id": "PROJ-SMART-CAMPUS", "building": "TOWER-A", "level": "B2"},
    )

    out = asyncio.run(service.search(req))
    assert any(hit.id == "upl-local-test" for hit in out.hits)

    asyncio.run(service.reranker.close())
    asyncio.run(service.embedding.close())
    asyncio.run(service.cache.close())
