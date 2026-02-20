from __future__ import annotations

import asyncio

from app.core.config import settings
from app.services.upload_document_service import UploadDocumentService


class _FakeUploadFile:
    def __init__(self, filename: str, data: bytes) -> None:
        self.filename = filename
        self._data = data

    async def read(self) -> bytes:
        return self._data


class _FakeEmbeddingService:
    async def embed(self, text: str) -> list[float]:
        return [0.1] * settings.vector_dim


class _FakeVectorClient:
    def index_document_with_reason(self, doc, embedding):
        return True, "indexed"


def _setup_isolated_storage(tmp_path, monkeypatch) -> UploadDocumentService:
    upload_dir = tmp_path / "uploads"
    index_path = tmp_path / "uploaded_docs.json"
    monkeypatch.setattr(settings, "uploads_dir", str(upload_dir))
    monkeypatch.setattr(settings, "uploads_data_path", str(index_path))
    monkeypatch.setattr(settings, "tika_url", "")
    monkeypatch.setattr(settings, "upstage_dpe_enabled", False)
    monkeypatch.setattr(settings, "upstage_dpe_api_key", "")
    return UploadDocumentService()


def test_cp949_txt_extracts_text_and_marks_metadata(tmp_path, monkeypatch):
    service = _setup_isolated_storage(tmp_path, monkeypatch)
    text = "안전 점검을 위한 건설 샘플 문서 본문"
    payload = asyncio.run(
        service.ingest(
            file=_FakeUploadFile("site_note_cp949.txt", text.encode("cp949")),
            top_down_context={"project_id": "PROJ"},
            already_embedded=True,
        )
    )

    assert payload["indexed"] is True
    assert payload["warning"] is None
    assert payload["extracted"] is True
    assert payload["chars_indexed"] > 0

    docs = service._load_docs()
    assert len(docs) == 1
    assert docs[0]["metadata"]["embedded"] is True
    assert docs[0]["metadata"]["project_id"] == "PROJ"
    assert text in docs[0]["content"]


def test_non_text_file_marked_as_not_extracted(tmp_path, monkeypatch):
    service = _setup_isolated_storage(tmp_path, monkeypatch)
    payload = asyncio.run(
        service.ingest(
            file=_FakeUploadFile("image.bin", b"\x00\x01\x02"),
            already_embedded=False,
        )
    )

    assert payload["extracted"] is False
    assert payload["warning"] is not None


def test_dpe_is_skipped_when_not_configured(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    index_path = tmp_path / "uploaded_docs.json"
    monkeypatch.setattr(settings, "uploads_dir", str(upload_dir))
    monkeypatch.setattr(settings, "uploads_data_path", str(index_path))
    monkeypatch.setattr(settings, "tika_url", "")
    monkeypatch.setattr(settings, "upstage_dpe_enabled", True)
    monkeypatch.setattr(settings, "upstage_dpe_api_key", "")
    service = UploadDocumentService()

    payload = asyncio.run(
        service.ingest(
            file=_FakeUploadFile("scan.pdf", b"\x00\x01\x02"),
            already_embedded=False,
        )
    )

    assert payload["extracted"] is False
    assert "upstage_dpe skipped" in (payload["warning"] or "")


def test_auto_context_extraction_from_filename_and_text(tmp_path, monkeypatch):
    service = _setup_isolated_storage(tmp_path, monkeypatch)
    payload = asyncio.run(
        service.ingest(
            file=_FakeUploadFile(
                "PROJ-SMART-CAMPUS_TOWER-A_B2_PKG-STR-001_slab.txt",
                b"Concrete slab pour checklist for Tower A B2 package PKG-STR-001",
            ),
            already_embedded=False,
        )
    )

    assert payload["indexed"] is True
    assert payload["automatic_context_extracted"] is True
    ctx = payload["top_down_context"]
    assert ctx.get("project_id") == "PROJ-SMART-CAMPUS"
    assert ctx.get("building") == "TOWER-A"
    assert ctx.get("level") == "B2"
    assert ctx.get("package_code") == "PKG-STR-001"
    assert ctx.get("work_type") == "concrete"


def test_manual_context_overrides_auto_context(tmp_path, monkeypatch):
    service = _setup_isolated_storage(tmp_path, monkeypatch)
    payload = asyncio.run(
        service.ingest(
            file=_FakeUploadFile(
                "PROJ-SMART-CAMPUS_TOWER-A_B2_concrete.txt",
                b"Tower A B2 concrete scope",
            ),
            top_down_context={"building": "TOWER-Z"},
            already_embedded=False,
        )
    )

    assert payload["indexed"] is True
    assert payload["top_down_context"].get("building") == "TOWER-Z"


def test_upload_sets_embedding_indexed_when_vector_index_succeeds(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    index_path = tmp_path / "uploaded_docs.json"
    monkeypatch.setattr(settings, "uploads_dir", str(upload_dir))
    monkeypatch.setattr(settings, "uploads_data_path", str(index_path))
    monkeypatch.setattr(settings, "tika_url", "")
    monkeypatch.setattr(settings, "upstage_dpe_enabled", False)

    service = UploadDocumentService(
        vector=_FakeVectorClient(),
        embedding=_FakeEmbeddingService(),
    )
    payload = asyncio.run(
        service.ingest(
            file=_FakeUploadFile("embedding_probe.txt", b"Embedding should be indexed."),
            already_embedded=False,
        )
    )

    assert payload["embedding_indexed"] is True
    docs = service._load_docs()
    assert docs[0]["metadata"]["embedded"] is True


def test_auto_context_extraction_avoids_duplicate_building_and_wbs(tmp_path, monkeypatch):
    service = _setup_isolated_storage(tmp_path, monkeypatch)
    payload = asyncio.run(
        service.ingest(
            file=_FakeUploadFile(
                "auto_context_check.txt",
                b"Project PROJ-SMART-CAMPUS Building TOWER-A Level B2 WBS WBS-STR-100 concrete slab pour",
            ),
            already_embedded=False,
        )
    )

    ctx = payload["top_down_context"]
    assert ctx.get("project_id") == "PROJ-SMART-CAMPUS"
    assert ctx.get("building") == "TOWER-A"
    assert ctx.get("wbs_code") == "WBS-STR-100"
