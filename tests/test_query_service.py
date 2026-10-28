"""QueryService 测试（docs/05 §6/§10）。

Fake run_query_stream 注入，断言首帧 session、透传、终态关流、异常兜底。
不碰真实 LLM / DB / Qdrant / Redis。
"""
from __future__ import annotations

import asyncio

import pytest

from app.query_graph.api.services.query_service import TERMINAL_TYPES, QueryService


class FakeRunner:
    """按脚本产出事件；可选在指定步抛异常。"""

    def __init__(self, events: list[dict], raise_at: int | None = None, exc: Exception | None = None):
        self.events = events
        self.raise_at = raise_at
        self.exc = exc or RuntimeError("boom")
        self.calls: list[dict] = []

    async def __call__(self, query: str, session_id: str, request_id: str, context=None):
        self.calls.append({"query": query, "session_id": session_id, "request_id": request_id})
        for i, e in enumerate(self.events):
            if self.raise_at is not None and i == self.raise_at:
                raise self.exc
            yield e
        # raise_at 落在事件序列之外（含空列表 + raise_at=0）也要抛
        if self.raise_at is not None and self.raise_at >= len(self.events):
            raise self.exc


def _collect(runner: FakeRunner, query="q", session_id="s1", request_id="q_1"):
    svc = QueryService(run_query_stream=runner)

    async def _go():
        return [e async for e in svc.stream_events(query, session_id, request_id)]

    return asyncio.run(_go())


# ---------- 首帧 session ----------


def test_first_frame_is_session_event():
    out = _collect(FakeRunner([]))
    assert out[0]["type"] == "session"
    assert out[0]["session_id"] == "s1"
    assert out[0]["request_id"] == "q_1"


def test_empty_session_id_generates_uuid():
    out = _collect(FakeRunner([]), session_id="")
    sid = out[0]["session_id"]
    assert sid and sid != ""
    assert len(sid) >= 8
    # 生成的 id 也要传给 run_query_stream
    # （FakeRunner.calls 由 _collect 的 runner 持有，这里重新构造验证）


def test_generated_session_id_passed_to_runner():
    runner = FakeRunner([])
    out = _collect(runner, session_id="")
    sid = out[0]["session_id"]
    assert runner.calls[0]["session_id"] == sid
    assert runner.calls[0]["request_id"] == "q_1"


# ---------- 透传 ----------


def test_events_passed_through_unchanged():
    events = [
        {"type": "intent", "intent": "compare", "confidence": 0.9},
        {"type": "token", "content": "你好"},
        {"type": "done", "request_id": "q_1"},
    ]
    out = _collect(FakeRunner(events))
    assert out[0]["type"] == "session"
    assert out[1:] == events


def test_query_passed_to_runner():
    runner = FakeRunner([])
    _collect(runner, query="茅台毛利率", session_id="s9", request_id="q_9")
    assert runner.calls[0]["query"] == "茅台毛利率"
    assert runner.calls[0]["session_id"] == "s9"


# ---------- 终态关流 ----------


@pytest.mark.parametrize("terminal", sorted(TERMINAL_TYPES))
def test_terminal_event_closes_stream(terminal):
    events = [
        {"type": "token", "content": "a"},
        {"type": terminal, "code": "X"} if terminal == "error" else {"type": terminal},
        {"type": "token", "content": "不该出现"},
    ]
    out = _collect(FakeRunner(events))
    types = [e["type"] for e in out]
    assert terminal in types
    assert "token" not in types[types.index(terminal) :]  # 终态后无后续


def test_non_terminal_does_not_close():
    events = [
        {"type": "intent", "intent": "simple_qa"},
        {"type": "token", "content": "a"},
    ]
    out = _collect(FakeRunner(events))
    assert [e["type"] for e in out] == ["session", "intent", "token"]


def test_done_closes_without_later_events():
    events = [
        {"type": "done", "request_id": "q_1"},
        {"type": "token", "content": "x"},
    ]
    out = _collect(FakeRunner(events))
    assert [e["type"] for e in out] == ["session", "done"]


# ---------- 异常兜底 ----------


def test_graph_crash_emits_internal_error():
    runner = FakeRunner([{"type": "token", "content": "a"}], raise_at=0)
    out = _collect(runner)
    assert out[0]["type"] == "session"
    err = out[-1]
    assert err["type"] == "error"
    assert err["code"] == "INTERNAL"
    assert err["message"]


def test_crash_mid_stream_emits_error_after_partial():
    runner = FakeRunner(
        [{"type": "token", "content": "a"}, {"type": "token", "content": "b"}],
        raise_at=1,
    )
    out = _collect(runner)
    types = [e["type"] for e in out]
    assert types == ["session", "token", "error"]


def test_error_event_shape():
    runner = FakeRunner([], raise_at=0)
    out = _collect(runner)
    err = out[-1]
    assert err["type"] == "error"
    assert err["code"] == "INTERNAL"
    assert err.get("stage") == "api"


# ---------- 规约 ----------


def test_must_define_terminal_types():
    assert TERMINAL_TYPES == {"done", "error", "clarify", "out_of_scope"}
