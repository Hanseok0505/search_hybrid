from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import httpx

from app.core.config import settings
from app.domain.construction import (
    ELASTIC_QUERY_FIELDS,
    TOP_DOWN_FILTER_KEYS,
    build_context_should_clauses,
    build_function_score_functions,
    expand_construction_query,
)
from app.models.schemas import Candidate
from app.services.sample_store import SampleStore

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    def __init__(self) -> None:
        self._enabled = settings.elastic_enabled
        self._base_url = settings.elastic_url.rstrip("/")
        self._index = settings.elastic_index
        self._timeout = settings.elastic_timeout_sec
        self._client = httpx.AsyncClient(timeout=self._timeout)
        self._sample_mode = settings.sample_mode
        self._index_ready = False
        self._sample = (
            SampleStore(settings.sample_data_path, settings.uploads_data_path) if self._sample_mode else None
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _ensure_index(self) -> bool:
        if not self._enabled or self._sample_mode:
            return False
        if self._index_ready:
            return True

        try:
            head = await self._client.head(f"{self._base_url}/{self._index}")
            if head.status_code == 200:
                self._index_ready = True
                return True
        except Exception:
            pass

        mapping_path = Path("infra/schemas/elasticsearch_documents_mapping.json")
        mapping: Dict = {}
        if mapping_path.exists():
            try:
                mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            except Exception:
                mapping = {}

        if not mapping:
            mapping = {
                "mappings": {
                    "dynamic": True,
                    "properties": {
                        "id": {"type": "keyword"},
                        "title": {"type": "text"},
                        "summary": {"type": "text"},
                        "content": {"type": "text"},
                        "metadata": {"type": "object", "enabled": True},
                    },
                }
            }

        try:
            create = await self._client.put(f"{self._base_url}/{self._index}", json=mapping)
            if create.status_code in {200, 201}:
                self._index_ready = True
                return True
        except Exception as exc:
            logger.warning("Failed to create elasticsearch index %s: %s", self._index, exc)
            return False

        return False

    async def health(self) -> str:
        if not self._enabled:
            return "disabled"
        if self._sample_mode:
            return "up" if self._sample and self._sample.is_ready else "degraded"
        try:
            r = await self._client.get(f"{self._base_url}/_cluster/health")
            return "up" if r.status_code == 200 else "degraded"
        except Exception:
            return "down"

    async def search(self, query: str, top_k: int, filters: Optional[dict] = None) -> list[Candidate]:
        if not self._enabled:
            return []
        if self._sample_mode and self._sample:
            return self._sample.keyword_search(query=query, top_k=top_k, source="elastic", filters=filters)

        expanded_query = expand_construction_query(query)
        must_clause: list[dict] = [
            {
                "multi_match": {
                    "query": expanded_query,
                    "fields": ELASTIC_QUERY_FIELDS,
                    "type": "best_fields",
                }
            }
        ]
        should_clause: list[dict] = build_context_should_clauses(filters or {})
        function_score_functions: list[dict] = build_function_score_functions(filters or {})
        filter_clause: list[dict] = []
        if filters:
            for key, value in filters.items():
                mapped_key = f"metadata.{key}" if key in TOP_DOWN_FILTER_KEYS else key
                filter_clause.append({"term": {mapped_key: value}})

        body = {
            "size": min(top_k, settings.max_candidates_per_source),
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "must": must_clause,
                            "should": should_clause,
                            "minimum_should_match": 0,
                            "filter": filter_clause,
                        }
                    },
                    "functions": function_score_functions,
                    "score_mode": "sum",
                    "boost_mode": "sum",
                }
            },
        }

        try:
            r = await self._client.post(f"{self._base_url}/{self._index}/_search", json=body)
            r.raise_for_status()
            payload = r.json()
            hits = payload.get("hits", {}).get("hits", [])
            out: list[Candidate] = []
            for item in hits:
                src = item.get("_source", {})
                out.append(
                    Candidate(
                        id=str(item.get("_id")),
                        title=src.get("title"),
                        content=src.get("content"),
                        source="elastic",
                        raw_score=float(item.get("_score", 0.0)),
                        metadata=src.get("metadata", {}),
                    )
                )
            return out
        except Exception as exc:
            logger.warning("Elasticsearch search failed: %s", exc)
            return []

    async def index_document(self, doc: Dict) -> bool:
        if not self._enabled:
            return False
        if self._sample_mode:
            return True
        if not await self._ensure_index():
            return False

        metadata = dict(doc.get("metadata") or {})
        uploaded_at = metadata.get("uploaded_at") or datetime.now(timezone.utc).isoformat()

        body = {
            "id": doc.get("id"),
            "title": str(doc.get("title", ""))[:2048],
            "summary": str(doc.get("summary", ""))[:2000],
            "content": str(doc.get("content", "")),
            "scope_text": str(doc.get("scope_text", "")),
            "method_statement": str(doc.get("method_statement", "")),
            "risk_register": str(doc.get("risk_register", "")),
            "quality_checklist": str(doc.get("quality_checklist", "")),
            "schedule_notes": str(doc.get("schedule_notes", "")),
            "asset_tags": doc.get("asset_tags", []),
            "wbs_code": metadata.get("wbs_code", doc.get("wbs_code", "")),
            "package_code": metadata.get("package_code", doc.get("package_code", "")),
            "task_code": metadata.get("task_code", doc.get("task_code", "")),
            "csi_division": metadata.get("csi_division", doc.get("csi_division", "")),
            "spec_section": metadata.get("spec_section", doc.get("spec_section", "")),
            "updated_at": uploaded_at,
            "metadata": _sanitize_metadata(metadata),
        }

        try:
            resp = await self._client.post(
                f"{self._base_url}/{self._index}/_doc/{doc.get('id')}",
                json=body,
            )
            return resp.status_code < 300
        except Exception as exc:
            logger.warning("Elasticsearch indexing failed: %s", exc)
            return False


def _sanitize_metadata(metadata: Dict) -> Dict[str, object]:
    safe: Dict[str, object] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, bool):
            safe[key] = bool(value)
        elif isinstance(value, (int, float)):
            safe[key] = value
        elif isinstance(value, list):
            safe[key] = [str(item) for item in value]
        else:
            safe[key] = str(value)
    return safe
