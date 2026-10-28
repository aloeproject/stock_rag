"""上传年报 + SSE 进度接口测试。

不真跑 MinerU：run_ingest 用 fake 覆盖 DI，hub 手工塞事件验证 SSE 输出。
"""
import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.import_graph.api.dependencies import get_import_service, get_progress_hub
from app.import_graph.api.routes.import_router import router as import_router
from app.import_graph.api.services.import_service import ImportService
from app.import_graph.progress_hub import ProgressHub


class FakeIngestRunner:
    def __init__(self):
        self.calls: list[dict] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)


@pytest.fixture()
def hub():
    return ProgressHub()


@pytest.fixture()
def runner():
    return FakeIngestRunner()


@pytest.fixture()
def client(hub, runner, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.import_graph.api.services.import_service.PROJECT_ROOT", tmp_path
    )
    (tmp_path / "report").mkdir()
    service = ImportService(run_ingest=runner, progress_hub=hub)

    app = FastAPI()
    app.include_router(import_router)
    app.dependency_overrides[get_import_service] = lambda: service
    app.dependency_overrides[get_progress_hub] = lambda: hub
    return TestClient(app)


def test_upload_rejects_non_pdf(client):
    r = client.post("/api/import/report", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 200
    body = r.json()
    assert body["code"] == 400
    assert "pdf" in body["message"]


def test_upload_rejects_empty_file(client):
    r = client.post("/api/import/report", files={"file": ("a.pdf", b"", "application/pdf")})
    assert r.status_code == 200
    assert r.json()["code"] == 400


def test_upload_returns_document_id_and_schedules_ingest(client, runner, hub):
    r = client.post(
        "/api/import/report",
        files={"file": ("600628_2024.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["code"] == 200
    data = body["data"]
    assert data["source_file"] == "600628_2024.pdf"
    assert len(data["document_id"]) == 40

    # TestClient 会在请求结束时跑完 BackgroundTasks
    assert len(runner.calls) == 1
    assert runner.calls[0]["pdf_path"].endswith("600628_2024.pdf")
    assert runner.calls[0]["document_id"] == data["document_id"]
    assert runner.calls[0]["progress_hub"] is hub


def test_upload_registers_document_id_in_hub(client, hub):
    """上传落盘即 register——否则后台首个事件前订阅会误判 unknown。"""
    r = client.post(
        "/api/import/report",
        files={"file": ("600628_2024.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    doc_id = r.json()["data"]["document_id"]
    assert hub.has(doc_id) is True


def test_progress_sse_registered_doc_without_events_not_unknown(client, hub):
    """已 register 但 history 空（后台尚未 publish）→ 不应报 unknown 关流。

    竞态：POST 立刻返回，start_ingest 丢 BackgroundTasks 后置执行，
    前端 EventSource 已经连上来订阅了。
    """
    hub.register("doc_race")

    # 用 asyncio 限制等待时间：SSE 应阻塞等事件而非立刻吐 error
    async def scenario():
        from app.import_graph.api.routes.import_router import _event_stream

        agen = _event_stream(hub, "doc_race")
        task = asyncio.create_task(agen.__anext__())
        done, pending = await asyncio.wait({task}, timeout=0.1)
        if task in done:
            # 若已产出，不应是 unknown 错误
            return task.result()
        # 未产出 = 正常阻塞等待；补一个事件让它走完
        task.cancel()
        hub.publish("doc_race", {"type": "finished", "parse_status": "done"})
        agen2 = _event_stream(hub, "doc_race")
        return await asyncio.wait_for(agen2.__anext__(), timeout=1)

    line = asyncio.run(scenario())
    assert "unknown" not in line


def test_progress_sse_streams_events(client, hub):
    hub.publish("doc1", {"type": "progress", "step": "解析PDF", "status": "running"})
    hub.publish("doc1", {"type": "finished", "parse_status": "done"})

    with client.stream("GET", "/api/import/progress/doc1") as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        lines = [line for line in r.iter_lines() if line.startswith("data: ")]
    assert len(lines) >= 2
    assert "解析PDF" in lines[0]
    assert "finished" in lines[-1]


def test_progress_sse_unknown_document_emits_error(client):
    with client.stream("GET", "/api/import/progress/nope") as r:
        lines = [line for line in r.iter_lines() if line.startswith("data: ")]
    assert len(lines) == 1
    assert "unknown" in lines[0] or "error" in lines[0]
