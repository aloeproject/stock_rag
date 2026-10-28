"""node_entity 节点测试（docs/04a §3）。

假仓储注入 QueryAgentContext，Redis 用 monkeypatch 替换 get_redis_chat_history；
不碰 MySQL/Redis，断言 State patch 与回写。
"""
from __future__ import annotations

import asyncio

from langgraph.runtime import Runtime

from app.entities.stock_dict import StockDictDbData
from app.query_graph.context import QueryAgentContext
from app.query_graph.nodes import entity as entity_mod
from app.query_graph.nodes.entity import node_entity
from app.query_graph.state import default_state
from app.repositories.financial_metric_repository import FinancialMetricRepository
from app.repositories.stock_dict_repository import StockDictRepository


class FakeStockDict(StockDictRepository):
    def __init__(self, rows: list[StockDictDbData]):
        self.rows = rows

    def list_all(self) -> list[StockDictDbData]:
        return self.rows


class FakeMetrics(FinancialMetricRepository):
    def __init__(self, latest: str | None = "2024"):
        self.latest = latest

    def latest_report_period(self) -> str | None:
        return self.latest


class FakeChatHistory:
    def __init__(self):
        self.written: list[dict] = []

    def add_user_message(
        self,
        content: str,
        stock_codes: list[str] | None = None,
        periods: list[str] | None = None,
    ) -> None:
        self.written.append(
            {
                "role": "user",
                "content": content,
                "stock_codes": list(stock_codes or []),
                "periods": list(periods or []),
            }
        )


def _runtime(rows=None, latest="2024") -> Runtime:
    rows = rows or [
        StockDictDbData("600519", "贵州茅台", "贵州茅台酒股份有限公司", ["茅台", "贵州茅台酒"], "SH"),
        StockDictDbData("000008", "神州高铁", "神州高铁技术股份有限公司", [], "SZ"),
    ]
    ctx = QueryAgentContext(
        stock_dict_repository=FakeStockDict(rows),
        financial_metric_repository=FakeMetrics(latest),
    )
    return Runtime(context=ctx)  # type: ignore[arg-type]


def _state(**kw):
    s = default_state(query=kw.pop("query", "贵州茅台2024年毛利率"),
                      request_id="q1", session_id="s1")
    s["intent_raw"] = kw.pop("intent_raw", {
        "intent": "metric_calc",
        "stock_names": ["茅台"],
        "periods": ["2024"],
        "metrics": ["gross_margin"],
        "rewritten_query": "贵州茅台2024年毛利率是多少",
    })
    s.update(kw)
    return s


def test_resolves_alias_and_writes_redis(monkeypatch):
    chat = FakeChatHistory()
    monkeypatch.setattr(entity_mod, "get_redis_chat_history", lambda sid: chat)
    patch = asyncio.run(node_entity(_state(), _runtime()))

    assert patch["entities"]["stock_codes"] == ["600519"]
    assert patch["resolved_names"] == {"茅台": "600519"}
    assert patch["entities"]["periods"] == ["2024"]
    assert patch["missing_fields"] == []
    assert patch["rewritten_query"] == "贵州茅台2024年毛利率是多少"
    assert chat.written == [
        {
            "role": "user",
            "content": "贵州茅台2024年毛利率",
            "stock_codes": ["600519"],
            "periods": ["2024"],
        }
    ]


def test_unresolved_name_goes_missing_fields():
    state = _state(
        query="茅台股份的营收",
        intent_raw={
            "intent": "metric_calc",
            "stock_names": ["茅台股份"],
            "periods": ["2024"],
            "metrics": ["revenue"],
            "rewritten_query": "茅台股份2024年营收",
        },
    )
    patch = asyncio.run(node_entity(state, _runtime()))
    assert patch["entities"]["unresolved_names"] == ["茅台股份"]
    assert "stock_names" in patch["missing_fields"]


def test_relative_periods_absolutized():
    state = _state(
        query="近三年毛利率",
        intent_raw={
            "intent": "trend",
            "stock_names": ["神州高铁"],
            "periods": ["近三年"],
            "metrics": ["gross_margin"],
            "rewritten_query": "神州高铁近三年毛利率",
        },
    )
    patch = asyncio.run(node_entity(state, _runtime()))
    assert patch["entities"]["periods"] == ["2022", "2023", "2024"]
    assert patch["entities"]["stock_codes"] == ["000008"]


def test_coref_carries_codes_from_history():
    state = _state(
        query="它的毛利率是多少",
        intent_raw={
            "intent": "metric_calc",
            "stock_names": [],
            "periods": ["2024"],
            "metrics": ["gross_margin"],
            "rewritten_query": "它的毛利率是多少",
        },
    )
    state["history"] = [
        {
            "role": "user",
            "content": "贵州茅台2024年经营情况",
            "stock_codes": ["600519"],
            "stock_names": ["贵州茅台"],
        }
    ]
    patch = asyncio.run(node_entity(state, _runtime()))
    assert patch["entities"]["stock_codes"] == ["600519"]
    assert "贵州茅台" in patch["rewritten_query"]


def test_periods_carried_without_pronoun():
    """无代词省略句（有哪些风险）盲沿用上一轮期间（04b §4.4）。"""
    state = _state(
        query="有哪些风险？",
        intent_raw={
            "intent": "simple_qa",
            "stock_names": ["贵州茅台"],
            "periods": [],
            "metrics": [],
            "rewritten_query": "贵州茅台有哪些风险？",
        },
    )
    state["history"] = [
        {
            "role": "user",
            "content": "贵州茅台2024年盈利多少",
            "stock_codes": ["600519"],
            "stock_names": ["贵州茅台"],
            "periods": ["2024"],
        }
    ]
    patch = asyncio.run(node_entity(state, _runtime()))
    assert patch["entities"]["periods"] == ["2024"]
    assert patch["entities"]["stock_codes"] == ["600519"]
    assert patch["missing_fields"] == []


def test_periods_not_overridden_when_user_specifies():
    state = _state(
        query="2023年呢？",
        intent_raw={
            "intent": "metric_calc",
            "stock_names": ["贵州茅台"],
            "periods": ["2023年"],
            "metrics": ["盈利"],
            "rewritten_query": "贵州茅台2023年盈利多少？",
        },
    )
    state["history"] = [
        {
            "role": "user",
            "content": "贵州茅台2024年盈利多少",
            "stock_codes": ["600519"],
            "stock_names": ["贵州茅台"],
            "periods": ["2024"],
        }
    ]
    patch = asyncio.run(node_entity(state, _runtime()))
    assert patch["entities"]["periods"] == ["2023"]


def test_periods_not_carried_without_history_periods():
    state = _state(
        query="有哪些风险？",
        intent_raw={
            "intent": "simple_qa",
            "stock_names": ["贵州茅台"],
            "periods": [],
            "metrics": [],
            "rewritten_query": "贵州茅台有哪些风险？",
        },
    )
    state["history"] = [
        {
            "role": "user",
            "content": "贵州茅台2024年盈利多少",
            "stock_codes": ["600519"],
            "stock_names": ["贵州茅台"],
            "periods": [],
        }
    ]
    patch = asyncio.run(node_entity(state, _runtime()))
    assert patch["entities"]["periods"] == []


def test_missing_intent_raw_is_internal_error():
    state = default_state(query="x", request_id="q", session_id="s")
    state["intent_raw"] = {}
    patch = asyncio.run(node_entity(state, _runtime()))
    assert patch["error_code"] == "INTERNAL"
    assert patch["error_stage"] == "entity"


def test_redis_failure_is_warn_not_error(monkeypatch):
    def boom(sid):
        raise RuntimeError("redis down")

    monkeypatch.setattr(entity_mod, "get_redis_chat_history", boom)
    patch = asyncio.run(node_entity(_state(), _runtime()))
    assert patch.get("error") is None
    assert patch["entities"]["stock_codes"] == ["600519"]
