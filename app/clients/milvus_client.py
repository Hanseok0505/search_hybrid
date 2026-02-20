from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, Optional, Tuple

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
from app.domain.construction import TOP_DOWN_FILTER_KEYS, normalize_filter_variants
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
        self._collection_fields_cache: Optional[set[str]] = None

    @staticmethod
    def _coerce_collection_name(collection: object) -> str:
        if isinstance(collection, str):
            stripped = collection.strip()
            if not stripped:
                raise RuntimeError("Milvus collection name is empty")

            # Accept payloads like "['document_vectors']" and '["document_vectors"]'.
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(stripped)
                except Exception:
                    parsed = None
                if isinstance(parsed, (list, tuple)):
                    if len(parsed) != 1:
                        raise RuntimeError("Milvus collection name must resolve to exactly one value")
                    inner = str(parsed[0]).strip()
                    if not inner:
                        raise RuntimeError("Milvus collection name is empty")
                    stripped = inner
                else:
                    stripped = stripped[1:-1].strip()

            # Remove accidental extra quoting from env/template expansions.
            if (
                (stripped.startswith("'") and stripped.endswith("'"))
                or (stripped.startswith('"') and stripped.endswith('"'))
            ) and len(stripped) >= 2:
                stripped = stripped[1:-1].strip()

            if not stripped:
                raise RuntimeError("Milvus collection name is empty")
            return stripped

        if isinstance(collection, (tuple, list)):
            if len(collection) != 1:
                raise RuntimeError("Milvus collection name must resolve to one value")
            return MilvusVectorClient._coerce_collection_name(collection[0])

        raise RuntimeError(f"Milvus collection name is invalid: {type(collection)!r}")

    @property
    def _collection_name(self) -> str:
        return self._coerce_collection_name(self._collection)

    def _ensure_client(self) -> bool:
        return self._ensure_client_with_reason()[0]

    def _ensure_client_with_reason(self) -> Tuple[bool, str]:
        if self._sample_mode:
            return False, "sample mode"
        if not self._enabled:
            return False, "disabled"
        if PyMilvusClient is None:
            return False, "pymilvus not installed"
        if self._client is not None:
            return True, "client ready"

        try:
            self._client = PyMilvusClient(uri=self._uri, timeout=settings.milvus_timeout_sec)
            return True, "client connected"
        except Exception as exc:
            self._client = None
            return False, f"client init failed: {exc}"

    def _ensure_collection(self, collection_name: Optional[str] = None) -> bool:
        return self._ensure_collection_with_reason(collection_name=collection_name)[0]

    def _ensure_collection_with_reason(self, collection_name: Optional[str] = None) -> Tuple[bool, str]:
        collection_name = collection_name or self._collection_name
        if self._sample_mode:
            return False, "sample mode"
        if not self._enabled:
            return False, "disabled"
        if self._client is None:
            return False, "client not ready"
        if self._collection_ready:
            return True, "collection already ready"
        if DataType is None or FieldSchema is None or CollectionSchema is None:
            return False, "pymilvus schema modules unavailable"

        try:
            if self._client.has_collection(collection_name):
                index_ok, index_reason = self._ensure_collection_index_with_reason(collection_name)
                if not index_ok:
                    logger.warning("Milvus index not ready but proceeding: %s", index_reason)
                try:
                    self._client.load_collection(collection_name=collection_name)
                except Exception as exc:
                    return False, f"load_collection failed: {exc}"
                self._collection_ready = True
                return True, "existing collection loaded"

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
                FieldSchema(name="execution_readiness", dtype=DataType.FLOAT),
                FieldSchema(name="updated_at", dtype=DataType.VARCHAR, max_length=64),
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
                    "execution_readiness",
                    "updated_at",
                }:
                    continue
                fields.append(
                    FieldSchema(name=key, dtype=DataType.VARCHAR, max_length=256),
                )

            schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
            self._client.create_collection(collection_name=collection_name, schema=schema)
            index_ok, index_reason = self._ensure_collection_index_with_reason(collection_name)
            if not index_ok:
                logger.warning("Milvus index creation deferred: %s", index_reason)

            try:
                self._client.load_collection(collection_name=collection_name)
            except Exception as exc:
                return False, f"load_collection failed: {exc}"
            self._collection_ready = True
            return True, "collection created"
        except Exception as exc:
            return False, f"collection setup failed: {exc}"

    def _ensure_collection_index(self, collection_name: Optional[str] = None) -> bool:
        return self._ensure_collection_index_with_reason(collection_name=collection_name)[0]

    def _ensure_collection_index_with_reason(self, collection_name: Optional[str] = None) -> Tuple[bool, str]:
        collection_name = collection_name or self._collection_name
        if self._client is None:
            return False, "client not ready"
        if IndexParams is None:
            return False, "index params unavailable"
        try:
            indexes = self._client.list_indexes(collection_name)
            if _has_index(indexes, "embedding"):
                return True, "index exists"
            ip = IndexParams()
            ip.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="IP", params={})
            self._client.create_index(collection_name=collection_name, index_params=ip)
            return True, "index created"
        except Exception as exc:
            if "already exists" in str(exc).lower():
                return True, "index exists"
            return False, f"index failed: {exc}"

    def _collection_fields(self) -> set[str]:
        if self._collection_fields_cache is not None:
            return self._collection_fields_cache

        fallback = {
            "id",
            "title",
            "content",
            "summary",
            "metadata",
            "project_id",
            "building",
            "level",
            "work_type",
            "package_code",
            "task_code",
            "wbs_code",
            "csi_division",
            "spec_section",
            "embedding",
            "execution_readiness",
            "updated_at",
        }
        if not self._client:
            self._collection_fields_cache = fallback
            return fallback

        try:
            desc = self._client.describe_collection(self._collection_name)
            fields = {
                item.get("name")
                for item in desc.get("fields", [])
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            }
            if fields:
                fields = fields.union(fallback)
                self._collection_fields_cache = fields
                return fields
        except Exception:
            pass

        self._collection_fields_cache = fallback
        return fallback

    def _resolve_filter_field(self, key: str) -> Optional[str]:
        fields = self._collection_fields()
        if key in fields:
            return key
        meta_key = f"metadata.{key}"
        if meta_key in fields:
            return meta_key
        if key.startswith("metadata.") and key in fields:
            return key
        return None

    async def health(self) -> str:
        if not self._enabled:
            return "disabled"
        if self._sample_mode:
            return "up" if self._sample and self._sample.is_ready else "degraded"
        ok, reason = self._ensure_client_with_reason()
        if not ok:
            logger.debug("Milvus health failed: %s", reason)
            return "down"
        try:
            assert self._client is not None
            self._client.list_collections()
            return "up"
        except Exception as exc:
            logger.debug("Milvus health failed: %s", exc)
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
            ok, reason = self._ensure_client_with_reason()
            if not ok:
                logger.debug("Milvus search blocked: %s", reason)
                return []
            assert self._client is not None
            ok, reason = self._ensure_collection_with_reason(collection_name=self._collection_name)
            if not ok:
                logger.warning("Milvus search skipped: %s", reason)
                return []

            expr = self._build_filter_expression(filters)
            collection_name = self._collection_name
            timeout_sec = max(settings.milvus_timeout_sec, 1.0)

            async def _do_search():
                return await to_thread(
                    self._client.search,
                    collection_name=collection_name,
                    data=[embedding],
                    anns_field="embedding",
                    limit=min(top_k, settings.max_candidates_per_source),
                    output_fields=["title", "content", "metadata", "execution_readiness", "updated_at"],
                    filter=expr,
                    timeout=timeout_sec,
                )

            results = await asyncio.wait_for(_do_search(), timeout=timeout_sec + 1.0)
            if not isinstance(results, list) or not results:
                return []
            hits = results[0]
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
                        metadata=_ensure_metadata(entity.get("metadata", {}), entity),
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
                    ok, _ = self._ensure_client_with_reason()
                    if ok and self._client is not None:
                        self._client.load_collection(collection_name=self._collection_name)
                except Exception:
                    pass
            if "index not found" in str(exc):
                if self._ensure_collection_index_with_reason(collection_name=self._collection_name)[0]:
                    try:
                        if self._client is not None:
                            self._client.load_collection(collection_name=self._collection_name)
                    except Exception:
                        pass
            return []

    def _build_filter_expression(self, filters: Optional[dict]) -> Optional[str]:
        if not filters:
            return None
        clauses = []
        for key, value in filters.items():
            if value is None:
                continue

            resolved_field = self._resolve_filter_field(str(key))
            if not resolved_field:
                continue

            values = [_milvus_expr_value(v) for v in normalize_filter_variants(value)]
            values = [v for v in values if v is not None]
            if not values:
                continue
            if len(values) == 1:
                clauses.append(f"{resolved_field} == {values[0]}")
            else:
                unique = []
                for value in values:
                    if value not in unique:
                        unique.append(value)
                clauses.append(f"{resolved_field} in [{', '.join(unique)}]")
        return " && ".join(clauses) if clauses else None

    async def index_document(self, doc: Dict, embedding: list[float]) -> bool:
        ok, _reason = self.index_document_with_reason(doc, embedding)
        return ok

    def index_document_with_reason(self, doc: Dict, embedding: list[float]) -> tuple[bool, str]:
        if not self._enabled:
            return False, "disabled"
        if self._sample_mode:
            return True, "sample mode"

        collection_name = self._collection_name
        ok, reason = self._ensure_client_with_reason()
        if not ok:
            return False, reason
        assert self._client is not None
        ok, reason = self._ensure_collection_with_reason(collection_name=collection_name)
        if not ok:
            return False, reason

        metadata = dict(doc.get("metadata") or {})
        data = {
            "id": str(doc.get("id")),
            "title": str(doc.get("title", ""))[:1024],
            "content": str(doc.get("content", ""))[:65000],
            "summary": str(doc.get("summary", ""))[:2048],
            "metadata": _to_jsonable(metadata),
            "project_id": str(metadata.get("project_id", "")) or "",
            "building": str(metadata.get("building", "")) or "",
            "level": str(metadata.get("level", "")) or "",
            "work_type": str(metadata.get("work_type", "")) or "",
            "package_code": str(metadata.get("package_code", "")) or "",
            "task_code": str(metadata.get("task_code", "")) or "",
            "wbs_code": str(metadata.get("wbs_code", "")) or "",
            "csi_division": str(metadata.get("csi_division", "")) or "",
            "spec_section": str(metadata.get("spec_section", "")) or "",
            "execution_readiness": metadata.get("execution_readiness", 0.0),
            "updated_at": metadata.get("uploaded_at", ""),
            "embedding": embedding,
        }

        for key in TOP_DOWN_FILTER_KEYS:
            if key in data:
                continue
            value = metadata.get(key, "")
            data[key] = str(value or "")

        try:
            self._client.insert(collection_name=collection_name, data=[data])
            self._client.flush(collection_name=collection_name)
            self._collection_ready = True
            self._collection_fields_cache = None
            return True, "indexed"
        except Exception as exc:
            return False, str(exc)

    async def delete_document(self, doc_id: str) -> bool:
        if not self._enabled:
            return False
        if self._sample_mode:
            return True
        collection_name = self._collection_name
        ok, reason = self._ensure_client_with_reason()
        if not ok:
            logger.debug("Milvus delete blocked: %s", reason)
            return False
        assert self._client is not None
        ok, reason = self._ensure_collection_with_reason(collection_name=collection_name)
        if not ok:
            logger.debug("Milvus delete skipped: %s", reason)
            return False

        try:
            safe_id = str(doc_id).replace('"', '\\"')
            self._client.delete(collection_name=collection_name, filter=f'id == "{safe_id}"')
            self._client.flush(collection_name=collection_name)
            return True
        except Exception as exc:
            logger.warning("Milvus delete failed: %s", exc)
            return False


def _ensure_metadata(entity_metadata: object, entity: Optional[dict] = None) -> Dict:
    metadata = {}
    if isinstance(entity_metadata, dict):
        metadata.update(entity_metadata)
    if isinstance(entity, dict):
        for key in ("execution_readiness", "updated_at"):
            if key in entity and key not in metadata:
                metadata[key] = entity.get(key)
    return metadata


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


def _escape_milvus_string(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _milvus_expr_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return '""'
    return f'"{_escape_milvus_string(str(value))}"'


def _has_index(indexes, field: str) -> bool:
    if not indexes:
        return False
    if isinstance(indexes, list):
        for item in indexes:
            if isinstance(item, str) and item == field:
                return True
            if isinstance(item, dict):
                if item.get("field_name") == field or item.get("index_name") == field:
                    return True
    return False
