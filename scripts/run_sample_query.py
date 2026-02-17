from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

os.environ["SAMPLE_MODE"] = "true"
os.environ["REDIS_ENABLED"] = "false"
os.environ["ELASTIC_ENABLED"] = "true"
os.environ["MILVUS_ENABLED"] = "true"
os.environ["GRAPH_ENABLED"] = "true"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dependencies import Container
from app.models.schemas import ConstructionTopDownContext, SearchRequest


async def main() -> None:
    container = Container()
    try:
        req = SearchRequest(
            query="Tower A B2 slab pour quality checklist and risk controls",
            top_k=5,
            rerank_with_llm=False,
            use_cache=False,
            top_down_context=ConstructionTopDownContext(
                project_id="PROJ-SMART-CAMPUS",
                building="TOWER-A",
                level="B2",
                package_code="PKG-STR-001",
                work_type="concrete",
            ),
        )
        res = await container.hybrid.search(req)
        print(json.dumps(res.model_dump(), ensure_ascii=False, indent=2))
    finally:
        await container.close()


if __name__ == "__main__":
    asyncio.run(main())
