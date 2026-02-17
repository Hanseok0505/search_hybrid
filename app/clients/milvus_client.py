from __future__ import annotations

import logging
import asyncio
from typing import Dict, Optional

try:
    from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient as PyMilvusClient
    from pymilvus.milvus_client import IndexParams
except Exception:  # pragma: no cover
    PyMilvusClient = None
    DataType = None
    FieldSchema = None
    CollectionSchema = None
    IndexParams = None

from app.core.config import settings
from app.domain.construction import TOP_DOWN_FILTER_KEYS
from app.models.schemas import Candidate
from app.services.sample_store import SampleStore
from app.utils.compat import to_thread

logger = logging.getLogger(__name__)


class MilvusVectorClient:
    def __init__(self) -> None:
        self._enabled = settings.milvus_enabled
        self._collection = settings.milvus_collection
        self._sample_mode = settings.sample_mode
        self._sample = (
            SampleStore(settings.sample_data_path, settings.uploads_data_path) if self._sample_mode else None
        )
        self._client: Optional[PyMilvusClient] = None
        self._uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
        self._collection_ready = False

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        if not self._enabled or self._sample_mode or PyMilvusClient is None:
            return False
        try:
            self._client = PyMilvusClient(uri=self._uri, timeout=settings.milvus_timeout_sec)
            return True
        except Exception as exc:
            logger.warning("Milvus init failed: %s", exc)
            self._client = None
            return False

    def _ensure_collection(self) -> bool:
        if self._sample_mode or not self._enabled or self._client is None:
            return False
        if self._collection_ready:
            return True
        if DataType is None or FieldSchema is None or CollectionSchema is None:
            return False

        try:
            if self._client.has_collection(self._collection):
                if not self._ensure_collection_index():
                    return False
                self._client.load_collection(collection_name=self._collection)
                self._collection_ready = True
                return True

            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="project_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="building", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="level", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="work_type", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="package_code", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="task_code", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="wbs_code", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="csi_division", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="spec_section", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.vector_dim),
            ]
            for key in TOP_DOWN_FILTER_KEYS:
                if key in {
                    "id",
                    "title",
                    "content",
                    "summary",
                    "embedding",
                    "project_id",
                    "building",
                    "level",
                    "work_type",
                    "package_code",
                    "task_code",
                    "wbs_code",
                    "csi_division",
                    "spec_section",
                }:
                    continue
                fields.append(
                    FieldSchema(name=key, dtype=DataType.VARCHAR, max_length=256),
                )

            schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
            self._client.create_collection(collection_name=self._collection, schema=schema)

            if not self._ensure_collection_index():
                return False

            self._client.load_collection(collection_name=self._collection)
            self._collection_ready = True
            return True
        except Exception as exc:
            logger.warning("Milvus collection setup failed: %s", exc)
            return False

    def _ensure_collection_index(self) -> bool:
        if self._client is None or IndexParams is None:
            return False
        try:
            indexes = self._client.list_indexes(self._collection)
            if "embedding" in indexes:
                return True
            ip = IndexParams()
            ip.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="IP", params={})
            self._client.create_index(collection_name=self._collection, index_params=ip)
            return True
        except Exception as exc:
            logger.warning("Milvus index setup failed: %s", exc)
            return False

    async def health(self) -> str:
        if not self._enabled:
            return "disabled"
        if self._sample_mode:
            return "up" if self._sample and self._sample.is_ready else "degraded"
        try:
            if not self._ensure_client():
                return "down"
            assert self._client is not None
            self._client.list_collections()
            return "up"
        except Exception:
            return "down"

    async def search(
        self,
        embedding: list[float],
        top_k: int,
        filters: Optional[dict] = None,
        query_text: Optional[str] = None,
    ) -> list[Candidate]:
        if not self._enabled:
            return []
        if self._sample_mode and self._sample:
            return self._sample.keyword_search(
                query=query_text or "",
                top_k=top_k,
                source="vector",
                filters=filters,
            )
        try:
            if not self._ensure_client():
                return []
            assert self._client is not None
            if not self._ensure_collection():
                return []

            expr = None
            if filters:
                clauses = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        clauses.append(f'{key} == "{value}"')
                    else:
                        clauses.append(f"{key} == {value}")
                expr = " && ".join(clauses)

            timeout_sec = max(settings.milvus_timeout_sec, 1.0)

            async def _do_search():
                return await to_thread(
                    self._client.search,
                    collection_name=self._collection,
                    data=[embedding],
                    anns_field="embedding",
                    limit=min(top_k, settings.max_candidates_per_source),
                    output_fields=["title", "content", "metadata"],
                    filter=expr,
                    timeout=timeout_sec,
                )

            results = await asyncio.wait_for(_do_search(), timeout=timeout_sec + 1.0)
            hits = results[0] if results else []
            out: list[Candidate] = []
            for hit in hits:
                entity = hit.get("entity", {})
                out.append(
                    Candidate(
                        id=str(hit.get("id")),
                        title=entity.get("title"),
                        content=entity.get("content"),
                        source="vector",
                        raw_score=float(hit.get("distance", 0.0)),
                        metadata=entity.get("metadata", {}),
                    )
                )
            return out
        except asyncio.TimeoutError:
            logger.warning("Milvus search timed out after %ss", settings.milvus_timeout_sec)
            return []
        except Exception as exc:
            logger.warning("Milvus search failed: %s", exc)
            if "collection not loaded" in str(exc):
                try:
                    if self._ensure_client() and self._client is not None:
                        self._client.load_collection(collection_name=self._collection)
                except Exception:
                    pass
            if "index not found" in str(exc):
                if self._ensure_collection_index():
                    try:
                        if self._client is not None:
                            self._client.load_collection(collection_name=self._collection)
                    except Exception:
                        pass
            return []

    async def index_document(self, doc: Dict, embedding: list[float]) -> bool:
        if not self._enabled:
            return False
        if self._sample_mode:
            return True
        if not self._ensure_client():
            return False
        assert self._client is not None
        if not self._ensure_collection():
            return False

        metadata = dict(doc.get("metadata") or {})
        data = {
            "id": str(doc.get("id")),
            "title": str(doc.get("title", ""))[:1024],
            "content": str(doc.get("content", ""))[:65000],
            "summary": str(doc.get("summary", ""))[:2048],
            "metadata": _to_jsonable(metadata),
            "project_id": metadata.get("project_id", ""),
            "building": metadata.get("building", ""),
            "level": metadata.get("level", ""),
            "work_type": metadata.get("work_type", ""),
            "package_code": metadata.get("package_code", ""),
            "task_code": metadata.get("task_code", ""),
            "wbs_code": metadata.get("wbs_code", ""),
            "csi_division": metadata.get("csi_division", ""),
            "spec_section": metadata.get("spec_section", ""),
            "embedding": embedding,
        }

        for key in TOP_DOWN_FILTER_KEYS:
            if key in data or key in {"project_id", "building", "level", "work_type", "package_code"}:
                continue
            value = metadata.get(key, "")
            data[key] = str(value) if value is not None else None

        try:
            await to_thread(self._client.insert, collection_name=self._collection, data=data)
            return True
        except Exception as exc:
            logger.warning("Milvus indexing failed: %s", exc)
            return False


def _to_jsonable(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return str(value)




