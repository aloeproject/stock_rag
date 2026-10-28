"""run_query_stream 测试（docs/05 §4/§10）。

Fake 图注入，断言 default_state 三字段注入、custom 事件逐条产出。
不编译真图、不碰 LLM/DB。
"""
from __future__ import annotations

import asyncio

from app.query_graph import main_graph as mg


class FakeCompiled:
    def __init__(self, events):
        self.events = events
        self.calls: list[dict] = []

    async def astream(self, state, context=None, stream_mode=None):
        self.calls.append({"state": state, "context": context, "stream_mode": stream_mode})
        for e in self.events:
            yield ("custom", e)


class FakeBuilder:
    def __init__(self, compiled):
        self._compiled = compiled
        self.compile_calls = 0

    def compile(self):
        self.compile_calls += 1
        return self._compiled


def _run(events, monkeypatch, **kw):
    compiled = FakeCompiled(events)
    builder = FakeBuilder(compiled)
    monkeypatch.setattr(mg, "build_query_graph", lambda: builder)

    async def _go():
        return [
            e
            async for e in mg.run_query_stream(
                query=kw.get("query", "q"),
                session_id=kw.get("session_id", "s1"),
                request_id=kw.get("request_id", "q_1"),
                context=kw.get("context"),
            )
        ]

    out = asyncio.run(_go())
    return out, compiled


def test_default_state_injected(monkeypatch):
    out, compiled = _run([], monkeypatch, query="茅台毛利率", session_id="s9", request_id="q_9")
    st = compiled.calls[0]["state"]
    assert st["query"] == "茅台毛利率"
    assert st["session_id"] == "s9"
    assert st["request_id"] == "q_9"


def test_custom_events_yielded_in_order(monkeypatch):
    events = [
        {"type": "intent", "intent": "compare"},
        {"type": "token", "content": "2024"},
        {"type": "done", "request_id": "q_1"},
    ]
    out, _ = _run(events, monkeypatch)
    assert out == events


def test_astream_uses_custom_mode(monkeypatch):
    _, compiled = _run([], monkeypatch)
    assert compiled.calls[0]["stream_mode"] == ["custom"]


def test_context_passed_through(monkeypatch):
    sentinel = object()
    _, compiled = _run([], monkeypatch, context=sentinel)
    assert compiled.calls[0]["context"] is sentinel


def test_history_loaded_from_redis(monkeypatch):
    """起图前从 Redis 灌 history 进 state：intent/entity 指代消解读它（04a §4.4）。"""
    fake_hist = [{"role": "user", "content": "陕西煤业营收", "stock_codes": ["601225"], "periods": ["2024"]}]
    monkeypatch.setattr(mg, "get_window_messages", lambda h, k=None: fake_hist)
    _, compiled = _run([], monkeypatch, session_id="s1")
    assert compiled.calls[0]["state"]["history"] == fake_hist


def test_history_redis_fail_tolerated(monkeypatch):
    """Redis 挂了不阻断问答：history 退化为空，指代消解降级但不报错。"""
    monkeypatch.setattr(
        mg, "get_redis_chat_history", lambda sid: (_ for _ in ()).throw(RuntimeError("down"))
    )
    out, compiled = _run([{"type": "done"}], monkeypatch, session_id="s1")
    assert compiled.calls[0]["state"]["history"] == []
    assert out == [{"type": "done"}]
