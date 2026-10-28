"""查询 SSE 路由测试（docs/05 §7/§10）。

TestClient + Fake service，断言 HTTP 层：状态码、Content-Type、帧格式、400。
不在这里重复断言事件内容（那是 service 的活）。
参数校验走项目全局 exception_handler → 400 + ResponseData（非 FastAPI 默认 422）。
"""
from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.import_graph.api.middleware.exception_handler import register_exception_handlers
from app.query_graph.api.dependencies import get_query_service
from app.query_graph.api.routes.query_router import router as query_router


class FakeQueryService:
    def __init__(self, events=None, raise_exc=False):
        self.events = events if events is not None else [{"type": "done", "request_id": "x"}]
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    async def stream_events(self, query, session_id, request_id):
        self.calls.append({"query": query, "session_id": session_id, "request_id": request_id})
        if self.raise_exc:
            raise RuntimeError("service blew up")
        for e in self.events:
            yield e


def _client(service):
    app = FastAPI()
    register_exception_handlers(app)  # 对齐真实 app：校验失败 → 400 ResponseData
    app.include_router(query_router)
    app.dependency_overrides[get_query_service] = lambda: service
    return TestClient(app)


def _frames(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[len("data: "):]))
    return out


# ---------- 200 + SSE ----------


def test_returns_event_stream():
    svc = FakeQueryService()
    r = _client(svc).post("/api/query/stream", json={"query": "毛利率", "session_id": "s1"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]


def test_frame_format_is_data_json():
    svc = FakeQueryService(events=[{"type": "token", "content": "hi"}, {"type": "done"}])
    r = _client(svc).post("/api/query/stream", json={"query": "q", "session_id": "s1"})
    text = r.text
    assert "data: " in text
    assert "\n\n" in text
    assert "event:" not in text  # 无 event: 行（对齐导入侧）
    frames = _frames(text)
    assert frames[0]["type"] == "token"
    assert frames[-1]["type"] == "done"


def test_request_id_has_q_prefix():
    svc = FakeQueryService()
    _client(svc).post("/api/query/stream", json={"query": "q", "session_id": "s1"})
    rid = svc.calls[0]["request_id"]
    assert rid.startswith("q_")
    assert len(rid) > 2


def test_body_passed_to_service():
    svc = FakeQueryService()
    _client(svc).post("/api/query/stream", json={"query": "茅台 2024 毛利率", "session_id": "abc"})
    assert svc.calls[0]["query"] == "茅台 2024 毛利率"
    assert svc.calls[0]["session_id"] == "abc"


# ---------- 参数校验 ----------


def test_missing_query_returns_400():
    svc = FakeQueryService()
    r = _client(svc).post("/api/query/stream", json={"session_id": "s1"})
    assert r.status_code == 400
    body = r.json()
    assert body["code"] == 400


def test_empty_query_returns_400():
    svc = FakeQueryService()
    r = _client(svc).post("/api/query/stream", json={"query": "", "session_id": "s1"})
    assert r.status_code == 400
    assert r.json()["code"] == 400


def test_too_long_query_returns_400():
    svc = FakeQueryService()
    r = _client(svc).post("/api/query/stream", json={"query": "x" * 501, "session_id": "s1"})
    assert r.status_code == 400
    assert r.json()["code"] == 400


def test_session_id_optional():
    svc = FakeQueryService()
    r = _client(svc).post("/api/query/stream", json={"query": "q"})
    assert r.status_code == 200
    assert svc.calls[0]["session_id"] == ""


# ---------- 未预期异常 ----------


def test_service_crash_yields_error_frame():
    svc = FakeQueryService(raise_exc=True)
    r = _client(svc).post("/api/query/stream", json={"query": "q", "session_id": "s1"})
    assert r.status_code == 200  # 流已开始，HTTP 层仍是 200
    frames = _frames(r.text)
    assert frames[-1]["type"] == "error"
    assert frames[-1]["code"] == "INTERNAL"
