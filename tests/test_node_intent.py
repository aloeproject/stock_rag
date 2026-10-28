"""node_intent 节点测试（docs/04a §3/§8）。

假 LLM 注入 QueryAgentContext，不打真模型；断言 State patch、SSE、重试与降级。
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from langgraph.runtime import Runtime

from app.query_graph.context import QueryAgentContext
from app.query_graph.nodes.intent import node_intent
from app.query_graph.state import default_state

# 拒答文案（04a §5.3）：固定模板，实现侧常量须与此同文
EXPECTED_OOS_MESSAGE = (
    "我是股票年报问答助手，只能回答上市公司年报相关的问题"
    "（如财务指标、经营情况、风险因素等）。您可以换一个与年报相关的问题。"
)


class FakeLLM:
    """按序返回 content；errors 优先抛出，模拟超时/5xx。"""

    def __init__(self, payloads=None, errors=None):
        self.calls: list[str] = []
        self.payloads = list(payloads or [])
        self.errors = list(errors or [])

    def invoke(self, prompt):
        self.calls.append(prompt)
        if self.errors:
            raise self.errors.pop(0)
        payload = self.payloads.pop(0) if self.payloads else {}
        content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return SimpleNamespace(content=content)


def _runtime(llm) -> Runtime:
    return Runtime(context=QueryAgentContext(llm=llm))  # type: ignore[arg-type]


def _state(query="茅台2024年毛利率", history=None, **kw):
    s = default_state(query=query, request_id="q1", session_id="s1", history=history)
    s.update(kw)
    return s


def _good_payload(**over):
    p = {
        "intent": "metric_calc",
        "confidence": 0.92,
        "stock_names": ["茅台"],
        "periods": ["2024"],
        "metrics": ["毛利率"],
        "rewritten_query": "贵州茅台2024年毛利率是多少",
    }
    p.update(over)
    return p


def test_success_writes_state_and_sse():
    llm = FakeLLM([_good_payload()])
    events = []
    patch = asyncio.run(node_intent(_state(), _runtime(llm), events.append))

    assert patch["intent"] == "metric_calc"
    assert patch["rewritten_query"] == "贵州茅台2024年毛利率是多少"
    assert patch["intent_raw"]["stock_names"] == ["茅台"]
    assert patch["error"] is None
    assert events == [{"type": "intent", "intent": "metric_calc", "confidence": 0.92}]


def test_prompt_includes_query_and_history():
    llm = FakeLLM([_good_payload()])
    history = [
        {"role": "user", "content": "贵州茅台2024年经营情况"},
        {"role": "assistant", "content": "营收1200亿"},
    ]
    asyncio.run(node_intent(_state(history=history), _runtime(llm), None))
    prompt = llm.calls[0]
    assert "茅台2024年毛利率" in prompt
    assert "贵州茅台2024年经营情况" in prompt
    assert "营收1200亿" in prompt


def _oos_payload(**over):
    p = _good_payload(
        intent="out_of_scope",
        stock_names=[],
        periods=[],
        metrics=[],
        rewritten_query="帮我写首诗",
    )
    p.update(over)
    return p


def test_out_of_scope_emits_intent_then_out_of_scope_event():
    """只发 intent + out_of_scope 两个事件，不发 error（04a §5.3）。"""
    llm = FakeLLM([_oos_payload()])
    events = []
    patch = asyncio.run(node_intent(_state(query="帮我写首诗"), _runtime(llm), events.append))
    assert patch["intent"] == "out_of_scope"
    assert events == [
        {"type": "intent", "intent": "out_of_scope", "confidence": 0.92},
        {"type": "out_of_scope", "message": EXPECTED_OOS_MESSAGE},
    ]
    assert all(e["type"] != "error" for e in events)


def test_out_of_scope_writes_error_triple():
    llm = FakeLLM([_oos_payload()])
    patch = asyncio.run(node_intent(_state(query="帮我写首诗"), _runtime(llm), None))
    assert patch["error"] == EXPECTED_OOS_MESSAGE
    assert patch["error_code"] == "OUT_OF_SCOPE"
    assert patch["error_stage"] == "intent"


def test_out_of_scope_skips_intent_raw():
    """entity 不会跑，空壳 intent_raw 没有意义（04a §3）。"""
    llm = FakeLLM([_oos_payload()])
    patch = asyncio.run(node_intent(_state(query="帮我写首诗"), _runtime(llm), None))
    assert "intent_raw" not in patch


def test_invalid_intent_enum_retries_then_internal():
    llm = FakeLLM([
        _good_payload(intent="chat"),
        _good_payload(intent="闲聊"),
    ])
    patch = asyncio.run(node_intent(_state(), _runtime(llm), None))
    assert len(llm.calls) == 2  # 重试 1 次
    assert patch["error_code"] == "INTERNAL"
    assert patch["error_stage"] == "intent"


def test_non_json_retries_then_internal():
    llm = FakeLLM(["not json", "still not json"])
    patch = asyncio.run(node_intent(_state(), _runtime(llm), None))
    assert len(llm.calls) == 2
    assert patch["error_code"] == "INTERNAL"
    assert patch["error_stage"] == "intent"


def test_timeout_retries_then_internal():
    llm = FakeLLM(errors=[TimeoutError("t1"), TimeoutError("t2")])
    patch = asyncio.run(node_intent(_state(), _runtime(llm), None))
    assert len(llm.calls) == 2
    assert patch["error_code"] == "INTERNAL"
    assert patch["error_stage"] == "intent"


def test_retry_recovers_on_second_call():
    llm = FakeLLM(
        payloads=[_good_payload(intent="trend", rewritten_query="贵州茅台近三年毛利率")],
        errors=[TimeoutError("t1")],
    )
    patch = asyncio.run(node_intent(_state(), _runtime(llm), None))
    assert len(llm.calls) == 2
    assert patch["intent"] == "trend"
    assert patch["error"] is None


def test_missing_arrays_default_empty():
    llm = FakeLLM([{"intent": "simple_qa", "rewritten_query": "茅台主业"}])
    patch = asyncio.run(node_intent(_state(), _runtime(llm), None))
    raw = patch["intent_raw"]
    assert raw["stock_names"] == []
    assert raw["periods"] == []
    assert raw["metrics"] == []
    assert patch["error"] is None


def test_missing_rewritten_falls_back_to_query():
    llm = FakeLLM([_good_payload(rewritten_query="")])
    patch = asyncio.run(node_intent(_state(query="茅台营收"), _runtime(llm), None))
    assert patch["rewritten_query"] == "茅台营收"


def test_confidence_default_when_absent():
    llm = FakeLLM([{"intent": "compare", "rewritten_query": "茅台五粮液对比"}])
    events = []
    asyncio.run(node_intent(_state(), _runtime(llm), events.append))
    assert events[0]["confidence"] == 0.5


def test_prompt_instructs_period_carry_from_history():
    """无代词省略句要靠 prompt 规则补期间，不只补公司名（04a §6.2）。"""
    llm = FakeLLM([_good_payload()])
    asyncio.run(node_intent(_state(query="有哪些风险？"), _runtime(llm), None))
    prompt = llm.calls[0]
    assert "缺期间" in prompt or "报告期" in prompt
    assert "有哪些风险" in prompt
