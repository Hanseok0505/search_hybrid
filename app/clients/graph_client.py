from __future__ import annotations
import logging
from typing import Dict, Optional

try:
    from neo4j import AsyncGraphDatabase
except Exception:  # pragma: no cover
    AsyncGraphDatabase = None

from app.core.config import settings
from app.domain.construction import TOP_DOWN_FILTER_KEYS
from app.models.schemas import Candidate

from app.services.sample_store import SampleStore

logger = logging.getLogger(__name__)


class GraphSearchClient:
    def __init__(self) -> None:
        self._enabled = settings.graph_enabled
        self._sample_mode = settings.sample_mode
        self._sample = (
            SampleStore(settings.sample_data_path, settings.uploads_data_path) if self._sample_mode else None
        )
        self._driver = None
        self._fts_initialized = False
        if self._enabled and not self._sample_mode and AsyncGraphDatabase is not None:
            self._driver = AsyncGraphDatabase.driver(
                settings.graph_uri,
                auth=(settings.graph_user, settings.graph_password),
            )

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()

    async def health(self) -> str:
        if not self._enabled:
            return "disabled"
        if self._sample_mode:
            return "up" if self._sample and self._sample.is_ready else "degraded"
        try:
            assert self._driver is not None
            await self._driver.verify_connectivity()
            return "up"
        except Exception:
            return "down"

    async def search(self, query: str, top_k: int, filters: Optional[dict] = None) -> list[Candidate]:
        if not self._enabled:
            return []
        if self._sample_mode and self._sample:
            return self._sample.graph_search(query, top_k, filters=filters)
        try:
            assert self._driver is not None
            where_clause = ""
            params = {"q": query, "limit": min(top_k, settings.max_candidates_per_source)}
            if filters:
                predicates = []
                for i, (key, value) in enumerate(filters.items()):
                    param_key = f"f{i}"
                    predicates.append(f"node.{key} = ${param_key}")
                    params[param_key] = value
                if predicates:
                    where_clause = "WHERE " + " AND ".join(predicates)

            cypher = f"""
            CALL db.index.fulltext.queryNodes('document_ft', $q) YIELD node, score
            {where_clause}
            RETURN node.id AS id, node.title AS title, node.content AS content, score
            ORDER BY score DESC
            LIMIT $limit
            """
            async with self._driver.session(database=settings.graph_database) as session:
                records = await session.run(cypher, **params)
                out: list[Candidate] = []
                async for record in records:
                    out.append(
                        Candidate(
                            id=str(record["id"]),
                            title=record.get("title"),
                            content=record.get("content"),
                            source="graph",
                            raw_score=float(record.get("score", 0.0)),
                            metadata={},
                        )
                    )
                return out
        except Exception as exc:
            logger.warning("Graph search failed: %s", exc)
            return []

    async def index_document(self, doc: Dict) -> bool:
        if not self._enabled:
            return False
        if self._sample_mode:
            return True
        if self._driver is None:
            return False

        await self._ensure_fulltext_index()

        metadata = dict(doc.get("metadata") or {})
        props = {
            "id": str(doc.get("id")),
            "title": str(doc.get("title", ""))[:2048],
            "summary": str(doc.get("summary", ""))[:2000],
            "content": str(doc.get("content", "")),
            "source": str(metadata.get("source", "upload")),
            "original_file_name": str(metadata.get("original_file_name", "")),
            "uploaded_at": metadata.get("uploaded_at"),
            "embedded": bool(metadata.get("embedded", False)),
            "parser": str(metadata.get("parser", "")),
        }
        for key in TOP_DOWN_FILTER_KEYS:
            if key in metadata:
                props[key] = metadata.get(key)
        if "execution_readiness" in metadata:
            props["execution_readiness"] = metadata.get("execution_readiness")

        cypher = (
            """
            MERGE (d:Document {id: $id})
            SET d += $props
            """
        )
        try:
            async with self._driver.session(database=settings.graph_database) as session:
                await session.run(cypher, id=str(doc.get("id")), props=props)
            return True
        except Exception as exc:
            logger.warning("Graph indexing failed: %s", exc)
            return False

    async def _ensure_fulltext_index(self) -> None:
        if self._fts_initialized:
            return
        if self._driver is None:
            return

        initialize = """
        CREATE FULLTEXT INDEX document_ft IF NOT EXISTS
        FOR (d:Document)
        ON EACH [d.title, d.content, d.summary, d.method_statement, d.risk_register];
        """
        try:
            async with self._driver.session(database=settings.graph_database) as session:
                await session.run(initialize)
            self._fts_initialized = True
        except Exception:
            self._fts_initialized = True


def _to_metadata_scalar(metadata: Dict) -> Dict:
    out = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out





