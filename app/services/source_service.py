from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set

from app.core.config import settings
from app.models.schemas import Candidate, SourceItem
from app.domain.construction import normalize_filter_variants

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
            source_type = str(row.get("_source_type", "upload"))
            row_id = str(row.get("id"))
            can_delete = source_type == "upload" and row_id.startswith("upl-")
            items.append(
                SourceItem(
                    id=row_id,
                    title=row.get("title", "(untitled)"),
                    source_type=source_type,
                    can_delete=can_delete,
                    original_file_name=row.get("metadata", {}).get("original_file_name"),
                    stored_file_name=row.get("metadata", {}).get("stored_file_name"),
                    embedded=bool(row.get("metadata", {}).get("embedded", row.get("_source_type") == "sample")),
                    selected=False,
                    metadata=row.get("metadata", {}),
                )
            )
        items.sort(key=lambda x: (x.source_type, x.title.lower()))
        return items

    def get_source(self, source_id: str) -> Optional[Dict]:
        target = str(source_id).strip()
        if not target:
            return None
        for row in self.all_rows():
            if str(row.get("id")) == target:
                return row
        return None

    def read_source_content(self, source_id: str, max_chars: int = 120000) -> str:
        row = self.get_source(source_id)
        if not row:
            return ""
        metadata = row.get("metadata", {})
        text = str(row.get("content", "")) if row.get("content") is not None else ""
        if text:
            return text[:max_chars]

        stored_file_name = metadata.get("stored_file_name")
        if not stored_file_name:
            return ""
        safe_path = Path(settings.uploads_dir) / str(stored_file_name)
        if safe_path.exists() and safe_path.is_file():
            try:
                return safe_path.read_bytes().decode("utf-8", errors="ignore")[:max_chars]
            except Exception:
                pass
        return ""

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

    def delete_sources(self, source_ids: List[str]) -> tuple[List[str], Optional[str]]:
        if not source_ids:
            return [], "No source ids provided."

        selected = set(source_ids)
        source_rows = self._load_json(self._uploaded_path)
        if not source_rows:
            return [], "No uploaded source data available."

        deleted: List[str] = []
        kept_rows: List[Dict] = []
        warnings: List[str] = []

        for row in source_rows:
            row_id = str(row.get("id", ""))
            if row_id in selected:
                can_delete = str(row_id).startswith("upl-")
                if not can_delete:
                    kept_rows.append(row)
                    warnings.append(f"{row_id} is not deletable")
                    continue
                try:
                    stored_file_name = row.get("metadata", {}).get("stored_file_name")
                    file_path = self._resolve_uploaded_file_path(stored_file_name, row_id)
                    if file_path and file_path.exists():
                        file_path.unlink(missing_ok=True)
                    deleted.append(row_id)
                except Exception as exc:
                    warnings.append(f"{row_id}: {exc}")
                    kept_rows.append(row)
            else:
                kept_rows.append(row)

        if deleted:
            self._uploaded_path.write_text(json.dumps(kept_rows, ensure_ascii=False, indent=2), encoding="utf-8")

        warning: Optional[str] = None
        if warnings:
            warning = " / ".join(sorted(set(warnings)))
        return deleted, warning

    def _resolve_uploaded_file_path(self, stored_name: Optional[str], row_id: str) -> Optional[Path]:
        uploads_dir = Path(settings.uploads_dir)
        candidates = []
        if stored_name:
            candidates.append(uploads_dir / str(stored_name))
            candidates.append(uploads_dir / f"{row_id}_{stored_name}")
        candidates.append(uploads_dir / f"{row_id}_{stored_name}" if stored_name else uploads_dir / f"{row_id}_upload.txt")
        for path in candidates:
            try:
                if path.exists():
                    return path
            except Exception:
                continue
        return None

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
            candidates = [str(v) for v in normalize_filter_variants(val)]
            stored = metadata.get(key)
            if stored is None:
                continue
            if str(stored) not in candidates:
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
