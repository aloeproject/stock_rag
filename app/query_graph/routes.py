"""查询图条件边（docs/04 §6）。

path_map 键必须与路由返回值一一对应，缺键运行时炸（避坑清单）。
"""
from __future__ import annotations

from app.query_graph.state import QueryState


def route_after_intent(state: QueryState) -> str:
    """out_of_scope → END；其余进 entity。"""
    return "end" if state.get("intent") == "out_of_scope" else "entity"


def route_after_entity(state: QueryState) -> str:
    """有未解析名称或缺关键字段 → clarify 追问；否则 plan。"""
    entities = state.get("entities") or {}
    if entities.get("unresolved_names") or state.get("missing_fields"):
        return "clarify"
    return "plan"


def route_after_plan(state: QueryState) -> tuple[str, ...]:
    """按子问题 route 集合决定 fan-out 中哪几路执行（04g §5.1 串行）。

    calc 不在并行段：route=calc 也先跑 metric_query 取基期值，calc 由
    route_after_metric_query 决定是否串行接上。返回元组并行 fan-out，node_fuse 汇合。
    """
    routes = {sq["route"] for sq in state.get("sub_queries") or []}
    periods = set((state.get("entities") or {}).get("periods") or [])
    todo: list[str] = []
    if routes & {"vector", "mixed"}:
        todo.append("retrieve")
    # route=calc 同样要打 metric_query：基期值在 L2 物化表里，calc 只做纯函数（04g §3）
    if routes & {"metric_sql", "mixed", "calc"} or len(periods) > 1:
        todo.append("metric_query")
    return tuple(todo)


def route_after_metric_query(state: QueryState) -> str:
    """metric_query 出来进 calc 还是直接 fuse（04g §5.1 串行）。

    calc 触发：显式 route=calc，或问句涉及多期对比（同比/CAGR/多期序列需 L3）。
    """
    routes = {sq["route"] for sq in state.get("sub_queries") or []}
    periods = set((state.get("entities") or {}).get("periods") or [])
    return "calc" if ("calc" in routes or len(periods) > 1) else "fuse"
