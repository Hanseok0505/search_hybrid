from __future__ import annotations

import asyncio

from app.api.routes.search import _answer_from_hits
from app.models.schemas import AskRequest, Candidate


class _FailingLLMGateway:
    async def invoke(self, *_args, **_kwargs) -> dict:
        raise RuntimeError("llm unavailable")


class _Container:
    llm_gateway = _FailingLLMGateway()


def test_ask_fallback_marks_no_synthesis_when_llm_fails():
    payload = AskRequest(
        query="Concrete slab pour quality checklist",
        top_k=3,
        top_down_context={"project_id": "PROJ-1"},
        provider="ollama",
        model="gpt-oss-120b-cloud",
    )
    hits = [
        Candidate(
            id="d1",
            title="doc-1",
            content="Concrete pour sequence requires vibration and curing.",
            source="local",
            metadata={"project_id": "PROJ-1", "building": "A"},
        )
    ]

    ans, model_used, synthesized, reason = asyncio.run(_answer_from_hits(payload, hits, _Container()))
    assert synthesized is False
    assert reason is not None
    assert "No matching sources" not in ans
    assert model_used == "gpt-oss-120b-cloud"
