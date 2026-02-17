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


def _setup_isolated_storage(tmp_path, monkeypatch) -> UploadDocumentService:
    upload_dir = tmp_path / "uploads"
    index_path = tmp_path / "uploaded_docs.json"
    monkeypatch.setattr(settings, "uploads_dir", str(upload_dir))
    monkeypatch.setattr(settings, "uploads_data_path", str(index_path))
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
