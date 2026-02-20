from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.search import get_container, router


@dataclass
class _SourceItem:
    id: str
    can_delete: bool


class _FakeSources:
    def __init__(self) -> None:
        self.deleted_ids: list[str] = []
        self._items = [
            _SourceItem(id="upl-a", can_delete=True),
            _SourceItem(id="upl-b", can_delete=True),
            _SourceItem(id="sample-x", can_delete=False),
        ]

    def list_sources(self):
        return list(self._items)

    def delete_sources(self, source_ids):
        self.deleted_ids = list(source_ids)
        deletable = {item.id for item in self._items if item.can_delete}
        deleted = [sid for sid in source_ids if sid in deletable]
        return deleted, None


class _FakeBackend:
    def __init__(self, status: str = "down") -> None:
        self.status = status
        self.deleted: list[str] = []

    async def health(self) -> str:
        return self.status

    async def delete_document(self, doc_id: str) -> bool:
        self.deleted.append(doc_id)
        return True


class _FakeContainer:
    def __init__(self, backend_status: str = "down") -> None:
        self.sources = _FakeSources()
        self.elastic = _FakeBackend(status=backend_status)
        self.vector = _FakeBackend(status=backend_status)
        self.graph = _FakeBackend(status=backend_status)


def _client_for(container: _FakeContainer) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_container] = lambda: container
    return TestClient(app)


def test_delete_all_uploads_flag_deletes_only_deletable_ids():
    container = _FakeContainer(backend_status="down")
    with _client_for(container) as client:
        response = client.post("/v1/sources/delete", json={"delete_all_uploads": True})

    assert response.status_code == 200
    payload = response.json()
    assert payload["deleted"] is True
    assert sorted(payload["deleted_source_ids"]) == ["upl-a", "upl-b"]
    assert sorted(container.sources.deleted_ids) == ["upl-a", "upl-b"]
    assert "backend cleanup skipped" in (payload.get("warning") or "")


def test_delete_skip_backend_cleanup_avoids_backend_calls():
    container = _FakeContainer(backend_status="up")
    with _client_for(container) as client:
        response = client.post(
            "/v1/sources/delete",
            json={
                "source_ids": ["upl-a"],
                "skip_backend_cleanup": True,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["deleted"] is True
    assert container.elastic.deleted == []
    assert container.vector.deleted == []
    assert container.graph.deleted == []
