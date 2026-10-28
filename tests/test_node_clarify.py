"""node_clarify / find_candidates 测试（docs/04c §9）。

假字典注入 QueryAgentContext，writer 收集 SSE 事件；
不调 LLM、不碰 Redis，断言文案拼装、missing_fields、State error_code。
"""
from __future__ import annotations

import asyncio

from langgraph.runtime import Runtime

from app.entities.stock_dict import StockDictDbData
from app.query_graph.context import QueryAgentContext
from app.query_graph.matching.stock_name import find_candidates
from app.query_graph.nodes.clarify import node_clarify
from app.query_graph.state import default_state
from app.repositories.stock_dict_repository import StockDictRepository

ROWS = [
    StockDictDbData("000008", "神州高铁", "神州高铁技术股份有限公司", [], "SZ"),
    StockDictDbData("002261", "神州信息", "神州数码信息服务集团股份有限公司", [], "SZ"),
    StockDictDbData("600519", "贵州茅台", "贵州茅台酒股份有限公司", ["茅台", "贵州茅台酒"], "SH"),
    StockDictDbData("600550", "保变电气", "保定天威保变电气股份有限公司", [], "SH"),
]


class FakeStockDict(StockDictRepository):
    def __init__(self, rows: list[StockDictDbData], boom: bool = False):
        self.rows = rows
        self.boom = boom

    def list_all(self) -> list[StockDictDbData]:
        if self.boom:
            raise RuntimeError("stock_dict down")
        return self.rows


class _Writer:
    def __init__(self):
        self.events: list[dict] = []

    def __call__(self, item: dict) -> None:
        self.events.append(item)


def _runtime(rows=None, boom: bool = False) -> Runtime:
    return Runtime(  # type: ignore[arg-type]
        context=QueryAgentContext(
            stock_dict_repository=FakeStockDict(rows if rows is not None else ROWS, boom)
        )
    )


def _state(intent: str, entities: dict, missing: list[str], query: str = "q"):
    s = default_state(query=query, request_id="q1", session_id="s1")
    s["intent"] = intent
    s["entities"] = entities
    s["missing_fields"] = missing
    return s


def _ent(**kw) -> dict:
    base = {
        "stock_names": [],
        "stock_codes": [],
        "unresolved_names": [],
        "periods": [],
        "report_types": [],
        "metrics": [],
    }
    base.update(kw)
    return base


def _run(intent: str, entities: dict, missing: list[str], runtime=None) -> tuple[dict, list[dict]]:
    w = _Writer()
    patch = asyncio.run(node_clarify(_state(intent, entities, missing), runtime or _runtime(), w))
    return patch, w.events


# ---------- find_candidates ----------


def test_find_candidates_no_hit():
    assert find_candidates("茅台股份", ROWS) == []


def test_find_candidates_prefix_multi():
    assert find_candidates("神州", ROWS) == ["神州高铁", "神州信息"]


def test_find_candidates_alias_exact_returns_owner():
    # 别名精确也算命中面，候选给 stock_name 而不是原词
    assert find_candidates("茅台", ROWS) == ["贵州茅台"]


def test_find_candidates_ignores_disabled():
    rows = ROWS + [StockDictDbData("999999", "神州停用", "神州停用股份有限公司", [], "SZ", status=0)]
    assert find_candidates("神州", rows) == ["神州高铁", "神州信息"]


def test_find_candidates_too_short():
    assert find_candidates("茅", ROWS) == []


# ---------- 单缺：主体 ----------


def test_missing_stock_codes_metric_calc():
    patch, events = _run("metric_calc", _ent(), ["stock_codes"])
    assert events[0]["type"] == "clarify"
    assert events[0]["message"] == "请问您想查询哪家公司？可以回复公司名称或股票代码。"
    assert events[0]["missing_fields"] == ["stock_codes"]
    assert patch["error_code"] == "MISSING_ENTITY"
    assert patch["error_stage"] == "clarify"
    assert patch["error"] == events[0]["message"]


def test_missing_stock_codes_compare_uses_plural():
    patch, events = _run("compare", _ent(), ["stock_codes"])
    assert events[0]["message"] == "请问您想比较哪几家公司？可以回复公司名称或股票代码。"
    assert patch["error_code"] == "MISSING_ENTITY"


# ---------- 单缺：期间 / 指标 ----------


def test_missing_periods_by_intent():
    expect = {
        "metric_calc": "请问要查询哪个报告期？例如 2024 年。",
        "trend": "请问要看哪个时间段？例如近三年。",
        "compare": "请问要对比哪个报告期？例如 2023 年和 2024 年。",
    }
    for intent, msg in expect.items():
        _, events = _run(intent, _ent(stock_codes=["600519"]), ["periods"])
        assert events[0]["message"] == msg, intent
        assert events[0]["missing_fields"] == ["periods"]


def test_missing_metrics_single():
    _, events = _run("metric_calc", _ent(stock_codes=["600519"], periods=["2024"]), ["metrics"])
    assert events[0]["message"] == "请问要查询哪个指标？例如营业收入、毛利率、归母净利润。"
    assert events[0]["missing_fields"] == ["metrics"]


# ---------- 单缺：歧义名 / 未识别名 ----------


def test_unresolved_with_candidates():
    _, events = _run(
        "metric_calc",
        _ent(unresolved_names=["神州"]),
        ["stock_names"],
    )
    assert events[0]["message"] == "您指的是「神州高铁」还是「神州信息」？也可以直接回复股票代码。"
    assert events[0]["missing_fields"] == ["stock_names"]
    assert events[0]["unresolved_names"] == ["神州"]
    assert events[0]["candidates"] == {"神州": ["神州高铁", "神州信息"]}


def test_unresolved_without_candidates():
    _, events = _run(
        "metric_calc",
        _ent(unresolved_names=["茅台股份"]),
        ["stock_names"],
    )
    assert events[0]["message"] == "「茅台股份」暂无法识别，请换用完整公司简称或 6 位股票代码。"
    assert "candidates" not in events[0] or events[0]["candidates"] == {}


def test_unresolved_many_candidates_caps_at_3():
    rows = ROWS + [
        StockDictDbData("600001", "神州一", "神州一股份有限公司", [], "SH"),
        StockDictDbData("600002", "神州二", "神州二股份有限公司", [], "SH"),
    ]
    _, events = _run("metric_calc", _ent(unresolved_names=["神州"]), ["stock_names"], _runtime(rows))
    assert events[0]["message"] == "您指的是「神州高铁」、「神州信息」还是「神州一」等 4 家？建议直接回复股票代码。"


def test_multiple_unresolved_joined_with_fallback():
    _, events = _run(
        "metric_calc",
        _ent(unresolved_names=["茅台股份", "神州"]),
        ["stock_names"],
    )
    assert events[0]["message"] == (
        "「茅台股份」暂无法识别，请换用完整公司简称或 6 位股票代码；"
        "另外「神州」您指的是「神州高铁」还是「神州信息」？"
    )


def test_candidate_lookup_failure_degrades_not_internal():
    patch, events = _run(
        "metric_calc",
        _ent(unresolved_names=["神州"]),
        ["stock_names"],
        _runtime(boom=True),
    )
    assert events[0]["message"] == "「神州」暂无法识别，请换用完整公司简称或 6 位股票代码。"
    assert patch["error_code"] == "MISSING_ENTITY"


# ---------- 多缺：一次列全 ----------


def test_multi_missing_subject_period_metric_numbered():
    _, events = _run("metric_calc", _ent(), ["stock_codes", "periods", "metrics"])
    assert events[0]["message"] == (
        "请补充以下信息：\n"
        "1. 请问您想查询哪家公司？可以回复公司名称或股票代码。\n"
        "2. 请问要查询哪个报告期？例如 2024 年。\n"
        "3. 请问要查询哪个指标？例如营业收入、毛利率、归母净利润。\n"
        "可以一句话回复，例如：贵州茅台 2024 年营业收入。"
    )
    assert events[0]["missing_fields"] == ["stock_codes", "periods", "metrics"]


def test_multi_missing_period_metric_example():
    _, events = _run("metric_calc", _ent(stock_codes=["600519"]), ["periods", "metrics"])
    assert events[0]["message"].endswith("可以一句话回复，例如：2024 年营业收入。")
    assert events[0]["missing_fields"] == ["periods", "metrics"]


def test_multi_missing_subject_period_example():
    _, events = _run("trend", _ent(), ["stock_codes", "periods"])
    assert events[0]["message"].endswith("可以一句话回复，例如：贵州茅台 2024 年。")
    assert events[0]["missing_fields"] == ["stock_codes", "periods"]


def test_multi_missing_subject_metric_example():
    _, events = _run("metric_calc", _ent(), ["stock_codes", "metrics"])
    assert events[0]["message"].endswith("可以一句话回复，例如：贵州茅台 毛利率。")
    assert events[0]["missing_fields"] == ["stock_codes", "metrics"]


def test_unresolved_overrides_stock_codes_slot():
    """unresolved 与 stock_codes 同缺时，主体槽用歧义文案，不重复问公司。"""
    _, events = _run(
        "metric_calc",
        _ent(unresolved_names=["神州"]),
        ["stock_names", "stock_codes", "periods"],
    )
    msg = events[0]["message"]
    assert "您指的是「神州高铁」还是「神州信息」" in msg
    assert "请问您想查询哪家公司" not in msg
    assert events[0]["missing_fields"] == ["stock_names", "periods"]


# ---------- 兜底 / 边界 ----------


def test_fallback_when_nothing_missing():
    patch, events = _run("metric_calc", _ent(stock_codes=["600519"]), [])
    assert events[0]["message"] == "请补充要查询的公司或报告期。"
    assert patch["error_code"] == "MISSING_ENTITY"


def test_simple_qa_does_not_ask_periods():
    """simple_qa 缺期间不追问；只缺 stock_codes 时单句问公司。"""
    _, events = _run("simple_qa", _ent(), ["stock_codes"])
    assert events[0]["message"] == "请问您想查询哪家公司？可以回复公司名称或股票代码。"
    assert events[0]["missing_fields"] == ["stock_codes"]
