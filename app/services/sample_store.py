from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from app.models.schemas import Candidate

TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣_/-]+")


def _tokens(text: str) -> List[str]:
    base = TOKEN_RE.findall((text or "").lower())
    expanded = set(base)
    for tok in base:
        if re.fullmatch(r"[가-힣]+", tok) and len(tok) >= 2:
            for i in range(len(tok) - 1):
                expanded.add(tok[i : i + 2])
        elif len(tok) >= 4:
            for i in range(len(tok) - 2):
                expanded.add(tok[i : i + 3])
    return list(expanded)


class SampleStore:
    def __init__(self, file_path: str, extra_file_path: Optional[str] = None) -> None:
        self._docs: List[Dict] = []
        self._path = Path(file_path)
        self._extra_path = Path(extra_file_path) if extra_file_path else None
        self._cache_sig: Optional[str] = None
        self._reload()

    def _reload(self) -> None:
        parts: List[str] = []
        if self._path.exists():
            parts.append(f"{self._path}:{self._path.stat().st_mtime_ns}")
        if self._extra_path and self._extra_path.exists():
            parts.append(f"{self._extra_path}:{self._extra_path.stat().st_mtime_ns}")
        sig = "|".join(parts)
        if sig == self._cache_sig:
            return

        docs: List[Dict] = []
        if self._path.exists():
            docs.extend(json.loads(self._path.read_text(encoding="utf-8")))
        if self._extra_path and self._extra_path.exists():
            try:
                docs.extend(json.loads(self._extra_path.read_text(encoding="utf-8")))
            except Exception:
                pass
        self._docs = docs
        self._cache_sig = sig

    @property
    def is_ready(self) -> bool:
        self._reload()
        return len(self._docs) > 0

    def keyword_search(
        self,
        query: str,
        top_k: int,
        source: str,
        filters: Optional[Dict] = None,
    ) -> List[Candidate]:
        self._reload()
        q_tokens = set(_tokens(query))
        scored = []
        for doc in self._docs:
            if not self._passes_filters(doc, filters):
                continue
            title = doc.get("title", "")
            content = doc.get("content", "")
            tokens = set(_tokens(f"{title} {content}"))
            overlap = sum(1 for t in tokens if t in q_tokens)
            if overlap <= 0:
                continue
            title_bonus = sum(2 for t in _tokens(title) if t in q_tokens)
            score = float(overlap + title_bonus)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)

        out = []
        for score, doc in scored[:top_k]:
            out.append(
                Candidate(
                    id=str(doc["id"]),
                    title=doc.get("title"),
                    content=doc.get("content"),
                    source=source,
                    raw_score=score,
                    metadata=doc.get("metadata", {}),
                )
            )
        return out

    def graph_search(self, query: str, top_k: int, filters: Optional[Dict] = None) -> List[Candidate]:
        self._reload()
        base = self.keyword_search(query, top_k * 2, source="graph", filters=filters)
        id_map = {str(d.get("id")): d for d in self._docs}
        q_tokens = set(_tokens(query))
        scored = []
        for cand in base:
            doc = id_map.get(cand.id, {})
            related_ids = doc.get("related_ids", [])
            relation_bonus = 0.0
            for rid in related_ids:
                rel_doc = id_map.get(str(rid))
                if not rel_doc:
                    continue
                rel_tokens = set(_tokens(rel_doc.get("title", "") + " " + rel_doc.get("content", "")))
                if q_tokens.intersection(rel_tokens):
                    relation_bonus += 1.0
            cand.raw_score += relation_bonus
            scored.append(cand)
        scored.sort(key=lambda x: x.raw_score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _passes_filters(doc: Dict, filters: Optional[Dict]) -> bool:
        if not filters:
            return True
        metadata = doc.get("metadata", {})
        for key, expected in filters.items():
            if key in doc and doc.get(key) == expected:
                continue
            if metadata.get(key) == expected:
                continue
            return False
        return True
