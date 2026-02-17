from __future__ import annotations

import asyncio

from app.models.schemas import LLMInvokeRequest
from app.services.llm_gateway import LLMGateway


def test_llm_gateway_rejects_invalid_task():
    gateway = LLMGateway()
    try:
        req = LLMInvokeRequest(task="invalid")
        try:
            asyncio.run(gateway.invoke(req))
            assert False, "expected ValueError"
        except ValueError:
            assert True
    finally:
        asyncio.run(gateway.close())
