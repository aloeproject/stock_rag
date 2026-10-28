"""查询图条件边路由测试（docs/04 §6/04g §5.1）。

纯函数路由：给定 State 片段断言分支，不跑整图。
"""
from __future__ import annotations

from app.query_graph.routes import (
    route_after_entity,
    route_after_intent,
    route_after_metric_query,
    route_after_plan,
)
from app.query_graph.state import default_state


def test_route_after_intent_out_of_scope_goes_end():
    state = default_state(query="今天天气怎么样")
    state["intent"] = "out_of_scope"
    assert route_after_intent(state) == "end"


def test_route_after_intent_normal_goes_entity():
    state = default_state(query="茅台毛利率")
    state["intent"] = "metric_calc"
    assert route_after_intent(state) == "entity"


def test_route_after_entity_unresolved_names_goes_clarify():
    state = default_state(query="茅台股份的营收")
    state["entities"]["unresolved_names"] = ["茅台股份"]
    assert route_after_entity(state) == "clarify"


def test_route_after_entity_missing_fields_goes_clarify():
    state = default_state(query="近三年毛利率")
    state["missing_fields"] = ["stock_codes"]
    assert route_after_entity(state) == "clarify"


def test_route_after_entity_ok_goes_plan():
    state = default_state(query="贵州茅台2024年毛利率")
    state["entities"]["stock_codes"] = ["600519"]
    assert route_after_entity(state) == "plan"


def test_route_after_plan_vector_only():
    state = default_state(query="经营情况如何")
    state["sub_queries"] = [
        {"sub_id": "sq1", "question": "经营情况", "route": "vector", "filters": {}, "hyde_used": False},
    ]
    assert route_after_plan(state) == ("retrieve",)


def test_route_after_plan_metric_only_skips_retrieve():
    state = default_state(query="毛利率是多少")
    state["sub_queries"] = [
        {"sub_id": "sq1", "question": "毛利率", "route": "metric_sql", "filters": {}, "hyde_used": False},
    ]
    assert route_after_plan(state) == ("metric_query",)


def test_route_after_plan_calc_route_still_runs_metric_query():
    """route=calc 也要跑 metric_query 取基期值，calc 串行在其后（04g §5.1）。"""
    state = default_state(query="同比变化")
    state["sub_queries"] = [
        {"sub_id": "sq1", "question": "同比", "route": "calc", "filters": {}, "hyde_used": False},
    ]
    assert route_after_plan(state) == ("metric_query",)


def test_route_after_plan_mixed_runs_both():
    state = default_state(query="毛利率多少，原因是什么")
    state["sub_queries"] = [
        {"sub_id": "sq1", "question": "混合", "route": "mixed", "filters": {}, "hyde_used": False},
    ]
    assert route_after_plan(state) == ("retrieve", "metric_query")


def test_route_after_plan_multi_period_needs_metric_query():
    """多期 → 并行段发 metric_query；calc 由 route_after_metric_query 决定。"""
    state = default_state(query="近两年营收对比")
    state["entities"]["periods"] = ["2023", "2024"]
    state["sub_queries"] = [
        {"sub_id": "sq1", "question": "营收", "route": "metric_sql", "filters": {}, "hyde_used": False},
    ]
    assert route_after_plan(state) == ("metric_query",)


# ---------- route_after_metric_query（04g §5.1 串行） ----------


def test_route_after_metric_query_calc_route_goes_calc():
    state = default_state(query="同比变化")
    state["sub_queries"] = [
        {"sub_id": "sq1", "question": "同比", "route": "calc", "filters": {}, "hyde_used": False},
    ]
    assert route_after_metric_query(state) == "calc"


def test_route_after_metric_query_multi_period_goes_calc():
    state = default_state(query="近两年营收对比")
    state["entities"]["periods"] = ["2023", "2024"]
    state["sub_queries"] = [
        {"sub_id": "sq1", "question": "营收", "route": "metric_sql", "filters": {}, "hyde_used": False},
    ]
    assert route_after_metric_query(state) == "calc"


def test_route_after_metric_query_plain_metric_skips_calc():
    """单期纯指标题：metric_query 出来直接 fuse，不进 calc。"""
    state = default_state(query="毛利率是多少")
    state["entities"]["periods"] = ["2024"]
    state["sub_queries"] = [
        {"sub_id": "sq1", "question": "毛利率", "route": "metric_sql", "filters": {}, "hyde_used": False},
    ]
    assert route_after_metric_query(state) == "fuse"


def test_route_after_metric_query_vector_only_goes_fuse():
    state = default_state(query="经营情况")
    state["sub_queries"] = [
        {"sub_id": "sq1", "question": "经营情况", "route": "vector", "filters": {}, "hyde_used": False},
    ]
    assert route_after_metric_query(state) == "fuse"
