"""ImportService 测试：落盘、document_id、调度后台工作流。

不真跑 MinerU：run_ingest 注入 fake，只断言被正确调度。
"""
import asyncio
from pathlib import Path

import pytest

from app.import_graph.api.services.import_service import ImportService


class FakeIngestRunner:
    def __init__(self):
        self.calls: list[dict] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)


@pytest.fixture()
def report_dir(tmp_path, monkeypatch):
    d = tmp_path / "report"
    d.mkdir()
    monkeypatch.setattr("app.import_graph.api.services.import_service.PROJECT_ROOT", tmp_path)
    return d


@pytest.fixture()
def runner():
    return FakeIngestRunner()


@pytest.fixture()
def service(runner):
    return ImportService(run_ingest=runner)


def test_save_rejects_non_pdf(service):
    with pytest.raises(ValueError, match="pdf"):
        service.save_upload(b"%PDF-1.4", filename="note.txt")


def test_save_rejects_empty_file(service):
    with pytest.raises(ValueError, match="空"):
        service.save_upload(b"", filename="a.pdf")


def test_save_writes_file_and_returns_document_id(service, report_dir):
    payload = b"%PDF-1.4 fake annual report"
    result = service.save_upload(payload, filename="600628_2024.pdf")

    saved = report_dir / "600628_2024.pdf"
    assert saved.read_bytes() == payload
    assert result["source_file"] == "600628_2024.pdf"
    assert len(result["document_id"]) == 40  # sha1 hex
    assert result["pdf_path"] == str(saved)


def test_document_id_is_sha1_of_content(service, report_dir):
    import hashlib

    payload = b"%PDF-1.4 same content"
    result = service.save_upload(payload, filename="a.pdf")
    assert result["document_id"] == hashlib.sha1(payload).hexdigest()


def test_start_ingest_schedules_background_run(service, runner):
    asyncio.run(service.start_ingest(pdf_path="/tmp/a.pdf", document_id="doc1"))
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["pdf_path"] == "/tmp/a.pdf"
    assert call["document_id"] == "doc1"
    assert call["progress_hub"] is not None


def test_filename_sanitized_against_path_traversal(service, report_dir):
    result = service.save_upload(b"%PDF-1.4", filename="../../evil.pdf")
    saved = Path(result["pdf_path"])
    assert saved.parent == report_dir
    assert saved.name == "evil.pdf"
