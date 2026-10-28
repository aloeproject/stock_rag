"""node_metric_query 节点测试（docs/04f §8/§9）。

假仓储注入 QueryAgentContext；断言词→码映射、口径裁决、metric_rows、
SSE metric 形状、空/截尾/DEPENDENCY。不碰真 MySQL。
"""
from __future__ import annotations

import asyncio
from decimal import Decimal

from langgraph.runtime import Runtime

from app.entities.metric_definition import (
    MetricCategory,
    MetricDefinitionDbData,
    MetricLevel,
)
from app.entities.report_table import StatementType
from app.query_graph.context import QueryAgentContext
from app.query_graph.nodes.metric_query import node_metric_query
from app.query_graph.state import default_state
from app.repositories.business_fact_repository import BusinessFactRepository
from app.repositories.financial_metric_repository import FinancialMetricRepository
from app.repositories.metric_definition_repository import MetricDefinitionRepository


class FakeMetricDefRepo(MetricDefinitionRepository):
    def __init__(self, defs):
        self.defs = defs

    def list_all(self):
        return self.defs


class FakeMetricRepo(FinancialMetricRepository):
    """记录查询参数，按序吐 rows；errors 优先抛出。"""

    def __init__(self, rows=None, error=None):
        self.rows = list(rows or [])
        self.error = error
        self.calls: list[dict] = []

    def query_by_codes(self, stock_codes, metric_codes, metric_periods, report_types):
        self.calls.append(
            {
                "stock_codes": list(stock_codes),
                "metric_codes": list(metric_codes),
                "metric_periods": list(metric_periods),
                "report_types": list(report_types),
            }
        )
        if self.error:
            raise self.error
        return list(self.rows)


class FakeFactRepo(BusinessFactRepository):
    """business_facts 假仓储：dim_names 空 = 全量（与真仓储同语义）。"""

    def __init__(self, rows=None, error=None):
        self.rows = list(rows or [])
        self.error = error
        self.calls: list[dict] = []

    def query_by_dim_names(self, stock_codes, dim_names, metric_periods, report_types):
        self.calls.append(
            {
                "stock_codes": list(stock_codes),
                "dim_names": list(dim_names),
                "metric_periods": list(metric_periods),
                "report_types": list(report_types),
            }
        )
        if self.error:
            raise self.error
        if not stock_codes:
            return []
        return list(self.rows)


class _Writer:
    def __init__(self):
        self.events: list[dict] = []

    def __call__(self, item: dict) -> None:
        self.events.append(item)


def _def(code, name, aliases=None, level=MetricLevel.L1):
    return MetricDefinitionDbData(
        metric_code=code,
        metric_name=name,
        aliases=aliases,
        category=MetricCategory.HEADLINE,
        statement_type=StatementType.HEADLINE,
        metric_level=level,
        calc_formula=None,
        unit_default=None,
        description=None,
    )


DEFS = [
    _def("revenue", "营业收入", "营业总收入|营收"),
    _def("net_profit_attributable", "归属于上市公司股东的净利润", "归母净利润|净利润"),
    _def("gross_margin", "销售毛利率", "毛利率", MetricLevel.L2),
]


def _mrow(code="revenue", period="2024", value=20.82, scope="consolidated", ptype="current", **over):
    r = {
        "id": 1,
        "stock_code": "000008",
        "metric_code": code,
        "metric_name": code,
        "metric_period": period,
        "metric_period_type": ptype,
        "metric_value": None if value is None else Decimal(str(value)),
        "unit": "元",
        "statement_scope": scope,
        "page_no": 10,
    }
    r.update(over)
    return r


def _runtime(metric_repo=None, defs=None, fact_repo=None) -> Runtime:
    ctx = QueryAgentContext(
        financial_metric_repository=metric_repo or FakeMetricRepo(),
        metric_definition_repository=FakeMetricDefRepo(defs if defs is not None else DEFS),
        business_fact_repository=fact_repo or FakeFactRepo(),
    )
    return Runtime(context=ctx)  # type: ignore[arg-type]


def _sq(sub_id="sq1", question="神州高铁2024年营业收入是多少？", route="metric_sql", **over):
    f = {"stock_codes": ["000008"], "report_periods": ["2024"], "report_types": ["annual"], "sections": []}
    f.update(over.pop("filters", {}))
    item = {"sub_id": sub_id, "question": question, "route": route, "filters": f, "hyde_used": False}
    item.update(over)
    return item


def _state(sub_queries=None, metrics=None, **kw):
    s = default_state(query="q", request_id="q1", session_id="s1")
    s["sub_queries"] = sub_queries if sub_queries is not None else [_sq()]
    ent = {
        "stock_names": ["神州高铁"],
        "stock_codes": ["000008"],
        "unresolved_names": [],
        "periods": ["2024"],
        "report_types": ["annual"],
        "metrics": metrics if metrics is not None else ["营业收入"],
    }
    s["entities"] = ent
    s.update(kw)
    return s


# ---------- 成功路径 ----------


def test_success_maps_word_and_writes_rows():
    repo = FakeMetricRepo(rows=[_mrow()])
    events = _Writer()
    patch = asyncio.run(node_metric_query(_state(), _runtime(repo), events))

    assert patch["error"] is None
    rows = patch["metric_rows"]
    assert len(rows) == 1
    assert rows[0]["metric_code"] == "revenue"
    assert rows[0]["metric_period"] == "2024"
    assert rows[0]["sub_id"] == "sq1"
    # WHERE 参数：通俗词已映射成 metric_code
    assert repo.calls[0]["metric_codes"] == ["revenue"]
    assert repo.calls[0]["stock_codes"] == ["000008"]
    assert repo.calls[0]["metric_periods"] == ["2024"]


def test_sse_metric_event_shape():
    repo = FakeMetricRepo(rows=[_mrow(code="gross_margin", value=41.73, unit="%")])
    events = _Writer()
    asyncio.run(
        node_metric_query(_state(metrics=["毛利率"]), _runtime(repo), events)
    )
    assert events.events[0]["type"] == "metric"
    assert "rows" in events.events[0]


def test_alias_maps_to_code():
    repo = FakeMetricRepo(rows=[_mrow(code="net_profit_attributable")])
    state = _state(
        metrics=["归母净利润"],
        sub_queries=[_sq(question="神州高铁2024年归母净利润是多少？")],
    )
    asyncio.run(node_metric_query(state, _runtime(repo), None))
    assert repo.calls[0]["metric_codes"] == ["net_profit_attributable"]


def test_question_surface_picks_known_names():
    """entities.metrics 为空时，问句里出现的已知名/别名仍要进词表（04f §3.1）。"""
    repo = FakeMetricRepo(rows=[_mrow()])
    state = _state(
        metrics=[],
        sub_queries=[_sq(question="神州高铁2024年营业收入是多少？")],
    )
    patch = asyncio.run(node_metric_query(state, _runtime(repo), None))
    assert repo.calls[0]["metric_codes"] == ["revenue"]
    assert patch["error"] is None


def test_question_scan_no_substring_bleed():
    """问句含「归母净利润」时，不得再扫出「净利润」（net_profit 是独立码）。"""
    defs = [
        _def("net_profit", "净利润"),
        _def("net_profit_attributable", "归属于上市公司股东的净利润", "归母净利润"),
    ]
    repo = FakeMetricRepo(rows=[_mrow(code="net_profit_attributable")])
    state = _state(
        metrics=[],
        sub_queries=[_sq(question="神州高铁2024年归母净利润是多少？")],
    )
    asyncio.run(node_metric_query(state, _runtime(repo, defs), None))
    assert repo.calls[0]["metric_codes"] == ["net_profit_attributable"]


def test_glued_metric_phrase_splits_known_terms():
    """intent 粘连短语「归母净利润与扣非净利润」要先抽已知名，不能整词进 words。

    整词进词表会把子问题扫出的真词当子串跳过，name-in-word 只剩「净利润」
    → net_profit 查空（600467+601225 实测）。
    """
    defs = [
        _def("revenue", "营业收入", "营业总收入"),
        _def(
            "net_profit_attributable",
            "归属于上市公司股东的净利润",
            "归属于母公司股东的净利润|归母净利润",
        ),
        _def(
            "net_profit_attributable_deducted",
            "归属于上市公司股东的扣除非经常性损益的净利润",
            "扣非归母净利润|扣非净利润",
        ),
        _def("net_profit", "净利润"),
    ]
    repo = FakeMetricRepo(
        rows=[_mrow(code="net_profit_attributable"), _mrow(code="net_profit_attributable_deducted", id=2)]
    )
    state = _state(
        metrics=["归母净利润与扣非净利润"],
        sub_queries=[_sq(question="好当家2024年归母净利润与扣非净利润是多少？")],
    )
    asyncio.run(node_metric_query(state, _runtime(repo, defs), None))
    assert set(repo.calls[0]["metric_codes"]) == {
        "net_profit_attributable",
        "net_profit_attributable_deducted",
    }
    assert "net_profit" not in repo.calls[0]["metric_codes"]


# ---------- 筛选 / 口径 ----------


def test_skips_non_metric_routes():
    repo = FakeMetricRepo()
    state = _state(sub_queries=[_sq(route="vector"), _sq(sub_id="sq2", route="calc")])
    patch = asyncio.run(node_metric_query(state, _runtime(repo), None))
    assert patch["metric_rows"] == []
    assert repo.calls == []


def test_scope_consolidated_wins():
    repo = FakeMetricRepo(rows=[_mrow(scope="parent", id=1), _mrow(scope="consolidated", id=2)])
    patch = asyncio.run(node_metric_query(_state(), _runtime(repo), None))
    assert patch["metric_rows"][0]["statement_scope"] == "consolidated"


def test_restated_wins_over_current():
    repo = FakeMetricRepo(rows=[_mrow(ptype="current", id=1), _mrow(ptype="restated", id=2)])
    patch = asyncio.run(node_metric_query(_state(), _runtime(repo), None))
    assert patch["metric_rows"][0]["metric_period_type"] == "restated"


def test_sections_ignored_in_where():
    """sections 仅 vector 路消费（04f §5.2）。"""
    repo = FakeMetricRepo(rows=[_mrow()])
    sq = _sq(filters={"stock_codes": ["000008"], "report_periods": ["2024"], "sections": ["risk"]})
    asyncio.run(node_metric_query(_state(sub_queries=[sq]), _runtime(repo), None))
    assert "sections" not in repo.calls[0]
    assert repo.calls[0]["metric_periods"] == ["2024"]


# ---------- 降级 / 失败 ----------


def test_unmapped_word_dropped_not_error():
    repo = FakeMetricRepo(rows=[_mrow()])
    patch = asyncio.run(node_metric_query(_state(metrics=["重要指标", "营业收入"]), _runtime(repo), None))
    assert patch["error"] is None
    assert repo.calls[0]["metric_codes"] == ["revenue"]


def test_all_unmapped_returns_empty_rows():
    repo = FakeMetricRepo(rows=[_mrow()])
    state = _state(
        metrics=["重要指标"],
        sub_queries=[_sq(question="神州高铁2024年经营情况")],
    )
    patch = asyncio.run(node_metric_query(state, _runtime(repo), None))
    assert patch["metric_rows"] == []
    assert patch["error"] is None
    assert repo.calls == []  # 没有可查的 metric_code，不打库


def test_null_value_rows_dropped():
    repo = FakeMetricRepo(rows=[_mrow(value=None), _mrow(id=2, value=20.82)])
    patch = asyncio.run(node_metric_query(_state(), _runtime(repo), None))
    assert len(patch["metric_rows"]) == 1
    assert patch["metric_rows"][0]["metric_value"] == 20.82


def test_empty_result_is_legal():
    repo = FakeMetricRepo(rows=[])
    patch = asyncio.run(node_metric_query(_state(), _runtime(repo), None))
    assert patch["metric_rows"] == []
    assert patch["error"] is None


def test_mysql_unavailable_returns_dependency():
    repo = FakeMetricRepo(error=ConnectionError("mysql down"))
    patch = asyncio.run(node_metric_query(_state(), _runtime(repo), None))
    assert patch["error_code"] == "DEPENDENCY_UNAVAILABLE"
    assert patch["error_stage"] == "metric_query"


# ---------- 业务词 step_6（docs/04f §3.6 / 02g §8） ----------


def _fact_row(dim_name, dim_type="industry", code="revenue", name="营业收入", value=300.0):
    return {
        "id": 1,
        "stock_code": "000008",
        "metric_code": code,
        "metric_name": name,
        "metric_period": "2024",
        "metric_value": Decimal(str(value)),
        "unit": "元",
        "dim_type": dim_type,
        "dim_name": dim_name,
        "page_no": 12,
    }


def test_business_word_exact_match():
    """未映射词精确命中 dim_name → 进 metric_rows 并带 dim 字段（04f §3.6）。"""
    fact_repo = FakeFactRepo(rows=[_fact_row("煤炭采掘业")])
    state = _state(metrics=["煤炭采掘业"])
    patch = asyncio.run(node_metric_query(state, _runtime(fact_repo=fact_repo), None))
    assert patch["error"] is None
    rows = patch["metric_rows"]
    assert len(rows) == 1
    assert rows[0]["dim_name"] == "煤炭采掘业"
    assert rows[0]["dim_type"] == "industry"
    assert rows[0]["metric_period_type"] == "current"
    assert rows[0]["statement_scope"] is None
    # 取全量后由节点做包含匹配
    assert fact_repo.calls[0]["dim_names"] == []


def test_business_word_contains_match():
    """「煤炭」包含命中「煤炭采掘业」；精确优先于包含（04f §3.6）。"""
    fact_repo = FakeFactRepo(
        rows=[_fact_row("煤炭采掘业"), _fact_row("电力", value=200.0)]
    )
    state = _state(metrics=["煤炭"])
    patch = asyncio.run(node_metric_query(state, _runtime(fact_repo=fact_repo), None))
    names = {r["dim_name"] for r in patch["metric_rows"]}
    assert names == {"煤炭采掘业"}


def test_business_word_still_unmapped_dropped():
    """仍未命中的词丢弃，不进 metric_rows（04f §3.6）。"""
    fact_repo = FakeFactRepo(rows=[_fact_row("煤炭采掘业")])
    state = _state(metrics=["重要指标"])
    patch = asyncio.run(node_metric_query(state, _runtime(fact_repo=fact_repo), None))
    assert patch["metric_rows"] == []
    assert patch["error"] is None


def test_business_word_measure_column_follows_metric_words():
    """度量列跟指标词走：有指标码时只留这些码/这些词命中的列（04f §3.6）。"""
    fact_repo = FakeFactRepo(
        rows=[
            _fact_row("煤炭采掘业", code="revenue", name="营业收入"),
            _fact_row("煤炭采掘业", code=None, name="毛利率(%)", value=30.0),
            _fact_row("煤炭采掘业", code=None, name="营业收入比上年增减(%)", value=-1.0),
        ]
    )
    state = _state(metrics=["煤炭采掘业", "营业收入"])
    patch = asyncio.run(node_metric_query(state, _runtime(fact_repo=fact_repo), None))
    names = {r["metric_name"] for r in patch["metric_rows"]}
    assert "营业收入" in names
    # 「毛利率(%)」「比上年增减」不在指标词里，被度量列过滤掉
    assert "毛利率(%)" not in names
    assert "营业收入比上年增减(%)" not in names


def test_business_word_exact_beats_contains():
    """「煤炭」已有「煤炭」精确命中时不再捞「煤炭采掘业」（04f §3.6）。"""
    fact_repo = FakeFactRepo(
        rows=[_fact_row("煤炭", value=100.0), _fact_row("煤炭采掘业", value=300.0)]
    )
    state = _state(metrics=["煤炭"])
    patch = asyncio.run(node_metric_query(state, _runtime(fact_repo=fact_repo), None))
    names = {r["dim_name"] for r in patch["metric_rows"]}
    assert names == {"煤炭"}


def test_business_word_too_short_skipped():
    """长度 <2 的词不做包含匹配，防单字误伤（04f §3.6）。"""
    fact_repo = FakeFactRepo(rows=[_fact_row("煤炭采掘业")])
    state = _state(metrics=["煤"])
    patch = asyncio.run(node_metric_query(state, _runtime(fact_repo=fact_repo), None))
    assert patch["metric_rows"] == []


def test_business_fact_error_returns_dependency():
    fact_repo = FakeFactRepo(error=ConnectionError("mysql down"))
    state = _state(metrics=["煤炭采掘业"])
    patch = asyncio.run(node_metric_query(state, _runtime(fact_repo=fact_repo), None))
    assert patch["error_code"] == "DEPENDENCY_UNAVAILABLE"
    assert patch["error_stage"] == "metric_query"
