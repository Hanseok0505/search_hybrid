from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set

from app.core.config import settings
from app.models.schemas import Candidate, SourceItem

TOKEN_RE = re.compile(r"[A-Za-z0-9\uac00-\ud7a3_/-]+")


def _tokens(text: str) -> Set[str]:
    return set(TOKEN_RE.findall((text or "").lower()))


class SourceService:
    def __init__(self) -> None:
        self._sample_path = Path(settings.sample_data_path)
        self._uploaded_path = Path(settings.uploads_data_path)

    def list_sources(self) -> List[SourceItem]:
        rows = self.all_rows()
        items: List[SourceItem] = []
        for row in rows:
            items.append(
                SourceItem(
                    id=str(row.get("id")),
                    title=row.get("title", "(untitled)"),
                    source_type=str(row.get("_source_type", "upload")),
                    embedded=bool(row.get("metadata", {}).get("embedded", row.get("_source_type") == "sample")),
                    selected=False,
                    metadata=row.get("metadata", {}),
                )
            )
        items.sort(key=lambda x: (x.source_type, x.title.lower()))
        return items

    def all_rows(self) -> List[Dict]:
        rows: List[Dict] = []
        for row in self._load_json(self._sample_path):
            rows.append({**row, "_source_type": "sample"})
        for row in self._load_json(self._uploaded_path):
            rows.append({**row, "_source_type": "upload"})
        return rows

    def local_search(
        self,
        query: str,
        top_k: int,
        selected_ids: Optional[List[str]] = None,
        embedded_only: bool = False,
        top_down_filters: Optional[Dict] = None,
    ) -> List[Candidate]:
        selected = set(selected_ids or [])
        q_tokens = _tokens(query)
        scored: List[tuple[float, Dict]] = []

        for row in self.all_rows():
            rid = str(row.get("id", ""))
            selected_match = bool(selected and rid in selected)
            if selected and rid not in selected:
                continue

            metadata = row.get("metadata", {})
            is_embedded = bool(metadata.get("embedded", row.get("_source_type") == "sample"))
            if embedded_only and not is_embedded:
                continue
            if not selected_match and not self._passes_top_down_filters(metadata, top_down_filters):
                continue

            title = row.get("title", "")
            content = row.get("content", "")
            d_tokens = _tokens(f"{title} {content}")
            overlap = len(q_tokens.intersection(d_tokens))
            if overlap <= 0 and not selected_match:
                continue

            score = float(overlap + 2 * len(q_tokens.intersection(_tokens(title))))
            if selected_match:
                score += 2.0
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Candidate] = []
        for score, row in scored[:top_k]:
            out.append(
                Candidate(
                    id=str(row.get("id")),
                    title=row.get("title"),
                    content=row.get("content"),
                    source=str(row.get("_source_type", "local")),
                    raw_score=score,
                    metadata=row.get("metadata", {}),
                )
            )
        return out

    def fallback_context_docs(
        self,
        top_k: int,
        selected_ids: Optional[List[str]] = None,
        embedded_only: bool = False,
        top_down_filters: Optional[Dict] = None,
    ) -> List[Candidate]:
        selected = set(selected_ids or [])
        rows = self.all_rows()
        scored: List[tuple[float, Dict]] = []

        for row in rows:
            rid = str(row.get("id", ""))
            selected_match = bool(selected and rid in selected)
            if selected and not selected_match:
                continue

            metadata = row.get("metadata", {})
            is_embedded = bool(metadata.get("embedded", row.get("_source_type") == "sample"))
            if embedded_only and not is_embedded:
                continue
            if not selected_match and not self._passes_top_down_filters(metadata, top_down_filters):
                continue

            score = 0.0
            if selected_match:
                score += 100.0
            if row.get("_source_type") == "upload":
                score += 10.0
            uploaded_at = metadata.get("uploaded_at")
            if isinstance(uploaded_at, str):
                try:
                    score += datetime.fromisoformat(uploaded_at.replace("Z", "+00:00")).timestamp() / 1_000_000_000
                except Exception:
                    pass
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Candidate] = []
        for score, row in scored[:top_k]:
            out.append(
                Candidate(
                    id=str(row.get("id")),
                    title=row.get("title"),
                    content=row.get("content"),
                    source=str(row.get("_source_type", "local")),
                    raw_score=score,
                    metadata=row.get("metadata", {}),
                )
            )
        return out

    @staticmethod
    def _passes_top_down_filters(metadata: Dict, top_down_filters: Optional[Dict]) -> bool:
        if not top_down_filters:
            return True
        for key, val in top_down_filters.items():
            if val is None:
                continue
            # Soft match: missing keys do not exclude a document.
            if key in metadata and metadata.get(key) != val:
                return False
        return True

    @staticmethod
    def _load_json(path: Path) -> List[Dict]:
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
