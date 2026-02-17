from __future__ import annotations
from collections import defaultdict

from app.models.schemas import Candidate


def weighted_reciprocal_rank_fusion(
    ranked_lists: dict[str, list[Candidate]],
    weights: dict[str, float],
    rrf_k: int = 60,
) -> list[Candidate]:
    scores: dict[str, float] = defaultdict(float)
    selected: dict[str, Candidate] = {}

    for source, hits in ranked_lists.items():
        w = weights.get(source, 0.0)
        if not hits or w <= 0:
            continue
        for rank, cand in enumerate(hits, start=1):
            scores[cand.id] += w * (1.0 / (rrf_k + rank))
            if cand.id not in selected:
                selected[cand.id] = cand

    fused = []
    for cid, score in scores.items():
        c = selected[cid]
        c.fused_score = score
        fused.append(c)
    fused.sort(key=lambda x: x.fused_score, reverse=True)
    return fused




