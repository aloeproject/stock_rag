"""node_calc 节点测试（docs/04g §3.2/§9）。

纯函数层：假 metric_rows 注入 State，断言 kind 判定、status 三态、
calc_results 形状、SSE 只推 ok。不碰 DB / LLM。
"""
from __future__ import annotations

import asyncio

from langgraph.runtime import Runtime

from app.query_graph.context import QueryAgentContext
from app.query_graph.nodes.calc import MAX_CALC_RESULTS, node_calc, step_2_kind
from app.query_graph.state import default_state


class _Writer:
    def __init__(self):
        self.events: list[dict] = []

    def __call__(self, item: dict) -> None:
        self.events.append(item)


def _runtime() -> Runtime:
    return Runtime(context=QueryAgentContext())  # type: ignore[arg-type]


def _mrow(code="revenue", period="2024", value=100.0, stock="000008", **over):
    r = {
        "stock_code": stock,
        "metric_code": code,
        "metric_name": {"revenue": "营业收入", "gross_margin": "销售毛利率"}.get(code, code),
        "metric_period": period,
        "metric_period_type": "current",
        "metric_value": value,
        "unit": "元",
        "statement_scope": "consolidated",
        "page_no": 10,
        "sub_id": "sq1",
    }
    r.update(over)
    return r


def _sq(sub_id="sq1", question="2024年营收同比", route="calc", **over):
    item = {"sub_id": sub_id, "question": question, "route": route, "filters": {}, "hyde_used": False}
    item.update(over)
    return item


def _state(rows=None, sub_queries=None, periods=None, **kw):
    s = default_state(query="q", request_id="q1", session_id="s1")
    s["metric_rows"] = rows if rows is not None else [_mrow()]
    s["sub_queries"] = sub_queries if sub_queries is not None else [_sq()]
    s["entities"]["periods"] = periods if periods is not None else ["2024"]
    s.update(kw)
    return s


def _run(state, writer=None):
    return asyncio.run(node_calc(state, _runtime(), writer))


# ---------- step_2 kind 判定 ----------


def test_kind_yoy_keywords():
    assert step_2_kind("2024年营收同比") == "yoy"
    assert step_2_kind("YoY growth") == "yoy"


def test_kind_qoq_keyword():
    assert step_2_kind("2024Q4营收环比") == "qoq"


def test_kind_cagr_keyword():
    assert step_2_kind("近三年营收CAGR") == "cagr"
    assert step_2_kind("复合增长率") == "cagr"


def test_kind_multi_period_keywords():
    assert step_2_kind("近三年毛利率变化") == "multi_period"
    assert step_2_kind("营收走势") == "multi_period"


def test_kind_priority_yoy_over_multi():
    """「同比」与「变化」同现 → 取 yoy（04g §3.2 优先级）。"""
    assert step_2_kind("营收同比变化") == "yoy"


def test_kind_unsupported_keyword():
    assert step_2_kind("毛利率波动率") == "unsupported"


def test_kind_no_keyword_returns_none():
    assert step_2_kind("公司经营情况如何") is None


def test_kind_share_keywords():
    assert step_2_kind("煤炭营收占比") == "share"
    assert step_2_kind("主营业务构成") == "share"
    assert step_2_kind("收入比重") == "share"


def test_kind_share_before_multi_period():
    """「构成」是占比不是趋势：share 先于 multi_period（04g §3.2）。"""
    assert step_2_kind("收入构成变化") == "share"


# ---------- 成功路径四函数 ----------


def test_yoy_two_periods():
    rows = [_mrow(period="2023", value=100.0), _mrow(period="2024", value=200.0)]
    patch = _run(_state(rows=rows))
    assert patch["error"] is None
    [r] = patch["calc_results"]
    assert r["kind"] == "yoy"
    assert r["status"] == "ok"
    assert r["value"] == 100.0
    assert r["unit"] == "%"
    assert r["periods"] == ["2024", "2023"]
    assert r["metric_code"] == "revenue_yoy"
    assert r["formula"]


def test_qoq_two_periods():
    rows = [_mrow(period="2024Q3", value=80.0), _mrow(period="2024Q4", value=100.0)]
    patch = _run(_state(rows=rows, sub_queries=[_sq(question="2024Q4环比")]))
    [r] = patch["calc_results"]
    assert r["kind"] == "qoq"
    assert r["status"] == "ok"
    assert r["value"] == 25.0


def test_cagr_start_end_years():
    rows = [_mrow(period="2022", value=100.0), _mrow(period="2024", value=200.0)]
    patch = _run(_state(rows=rows, sub_queries=[_sq(question="CAGR")]))
    [r] = patch["calc_results"]
    assert r["kind"] == "cagr"
    assert r["status"] == "ok"
    # 2022→2024 两年：(2^(1/2)-1)*100
    assert abs(r["value"] - ((2**0.5 - 1) * 100)) < 1e-9


def test_multi_period_series_ordered():
    rows = [
        _mrow(period="2022", value=100.0),
        _mrow(period="2023", value=150.0),
        _mrow(period="2024", value=200.0),
    ]
    patch = _run(_state(rows=rows, sub_queries=[_sq(question="近三年变化")]))
    [r] = patch["calc_results"]
    assert r["kind"] == "multi_period"
    assert r["status"] == "ok"
    assert r["value"] is None
    assert r["values"] == [100.0, 150.0, 200.0]
    assert r["periods"] == ["2022", "2023", "2024"]


def test_multi_period_uses_l2_materialized_value():
    """L2 不现算：物化 gross_margin 直接进序列（04g §4.1）。"""
    rows = [_mrow(code="gross_margin", period="2023", value=30.0), _mrow(code="gross_margin", period="2024", value=40.0)]
    patch = _run(_state(rows=rows, sub_queries=[_sq(question="毛利率变化")]))
    [r] = patch["calc_results"]
    assert r["values"] == [30.0, 40.0]
    assert r["metric_code"] == "gross_margin_multi_period"


# ---------- share 配对 ----------


def _frow(dim_name, value, dim_type="industry", code="revenue", name="营业收入", period="2024"):
    return {
        "stock_code": "601225",
        "metric_code": code,
        "metric_name": name,
        "metric_period": period,
        "metric_value": value,
        "unit": "元",
        "dim_type": dim_type,
        "dim_name": dim_name,
        "page_no": 12,
        "sub_id": "sq1",
    }


def test_share_pairs_with_total():
    rows = [
        _frow("煤炭采掘业", 300.0),
        _frow("电力", 200.0),
        _frow("合计", 1000.0),
    ]
    patch = _run(_state(rows=rows, sub_queries=[_sq(question="主营业务构成")]))
    results = {r["dim_name"]: r for r in patch["calc_results"]}
    assert set(results) == {"煤炭采掘业", "电力"}
    assert results["煤炭采掘业"]["kind"] == "share"
    assert results["煤炭采掘业"]["value"] == 30.0
    assert results["煤炭采掘业"]["metric_code"] == "revenue_share"
    assert results["煤炭采掘业"]["dim_type"] == "industry"
    assert results["电力"]["value"] == 20.0


def test_share_missing_total_insufficient():
    """缺合计不猜 0/100，标 insufficient（docs/02g §8.3）。"""
    rows = [_frow("煤炭采掘业", 300.0)]
    patch = _run(_state(rows=rows, sub_queries=[_sq(question="主营业务构成")]))
    [r] = patch["calc_results"]
    assert r["status"] == "insufficient_periods"
    assert r["reason"] == "缺合计行"
    assert r["value"] is None


def test_share_total_zero_insufficient():
    rows = [_frow("煤炭采掘业", 300.0), _frow("合计", 0.0)]
    patch = _run(_state(rows=rows, sub_queries=[_sq(question="主营业务构成")]))
    [r] = patch["calc_results"]
    assert r["status"] == "insufficient_periods"
    assert r["reason"] == "分母为 0"


def test_share_same_name_different_dim_type_separate():
    """同名「电力」在 industry 与 product 是两个事实（docs/02g §3.3）。"""
    rows = [
        _frow("电力", 300.0, dim_type="industry"),
        _frow("电力", 200.0, dim_type="product"),
        _frow("合计", 1000.0, dim_type="industry"),
        _frow("合计", 500.0, dim_type="product"),
    ]
    patch = _run(_state(rows=rows, sub_queries=[_sq(question="主营业务构成")]))
    by_type = {r["dim_type"]: r for r in patch["calc_results"]}
    assert by_type["industry"]["value"] == 30.0
    assert by_type["product"]["value"] == 40.0


def test_share_skips_total_row_itself():
    rows = [_frow("煤炭采掘业", 300.0), _frow("合计", 1000.0)]
    patch = _run(_state(rows=rows, sub_queries=[_sq(question="主营业务构成")]))
    assert all(r["dim_name"] != "合计" for r in patch["calc_results"])


# ---------- status 三态 ----------


def test_yoy_missing_prior_insufficient():
    rows = [_mrow(period="2024", value=100.0)]
    patch = _run(_state(rows=rows))
    [r] = patch["calc_results"]
    assert r["status"] == "insufficient_periods"
    assert r["value"] is None
    assert r["reason"]


def test_yoy_div_zero_insufficient():
    rows = [_mrow(period="2023", value=0.0), _mrow(period="2024", value=100.0)]
    patch = _run(_state(rows=rows))
    [r] = patch["calc_results"]
    assert r["status"] == "insufficient_periods"
    assert r["value"] is None


def test_unsupported_kind_marks_status():
    rows = [_mrow(period="2024", value=100.0)]
    patch = _run(_state(rows=rows, sub_queries=[_sq(question="毛利率波动率")]))
    [r] = patch["calc_results"]
    assert r["status"] == "unsupported"
    assert r["kind"] == "unsupported"
    assert r["value"] is None
    assert r["reason"]


# ---------- 空/截尾 ----------


def test_empty_metric_rows_legal():
    patch = _run(_state(rows=[]))
    assert patch["calc_results"] == []
    assert patch["error"] is None


def test_no_calc_route_single_period_empty():
    """无 calc 子问题且单期 → 空合法（04g §8）。"""
    patch = _run(_state(sub_queries=[_sq(route="metric_sql", question="毛利率是多少")], periods=["2024"]))
    assert patch["calc_results"] == []


def test_multi_period_without_calc_route_still_runs():
    """多期即使 route=metric_sql 也进 calc（route_after_metric_query）。"""
    rows = [_mrow(period="2023", value=100.0), _mrow(period="2024", value=150.0)]
    patch = _run(_state(rows=rows, sub_queries=[_sq(route="metric_sql", question="营收")], periods=["2023", "2024"]))
    [r] = patch["calc_results"]
    assert r["kind"] == "multi_period"


def test_max_results_truncated():
    rows = [_mrow(code=f"m{i}", period="2023", value=1.0) for i in range(MAX_CALC_RESULTS + 5)]
    rows += [_mrow(code=f"m{i}", period="2024", value=2.0) for i in range(MAX_CALC_RESULTS + 5)]
    patch = _run(_state(rows=rows, sub_queries=[_sq(question="同比")]))
    assert len(patch["calc_results"]) == MAX_CALC_RESULTS


# ---------- SSE / sub_id ----------


def test_sse_only_ok_items():
    ok_rows = [_mrow(code="revenue", period="2023", value=100.0), _mrow(code="revenue", period="2024", value=200.0)]
    bad_rows = [_mrow(code="gross_margin", period="2024", value=30.0)]  # 缺期
    w = _Writer()
    _run(_state(rows=ok_rows + bad_rows, sub_queries=[_sq(question="同比")]), w)
    [ev] = w.events
    assert ev["type"] == "calc"
    codes = {row["metric_code"] for row in ev["rows"]}
    assert "revenue_yoy" in codes
    assert "gross_margin_yoy" not in codes


def test_sse_no_event_when_all_insufficient():
    w = _Writer()
    _run(_state(rows=[_mrow(period="2024", value=1.0)]), w)
    assert w.events == []


def test_sub_id_from_single_calc_sq():
    rows = [_mrow(period="2023", value=1.0), _mrow(period="2024", value=2.0)]
    patch = _run(_state(rows=rows, sub_queries=[_sq(sub_id="sq9", question="同比")]))
    [r] = patch["calc_results"]
    assert r["sub_id"] == "sq9"


# ---------- 多指标分组 ----------


def test_each_metric_code_computed():
    rows = [
        _mrow(code="revenue", period="2023", value=100.0),
        _mrow(code="revenue", period="2024", value=200.0),
        _mrow(code="gross_margin", period="2023", value=20.0),
        _mrow(code="gross_margin", period="2024", value=40.0),
    ]
    patch = _run(_state(rows=rows, sub_queries=[_sq(question="同比")]))
    codes = {r["metric_code"] for r in patch["calc_results"]}
    assert codes == {"revenue_yoy", "gross_margin_yoy"}
