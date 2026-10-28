"""Q7 node_calc（docs/04g）。

纯 Python L3：多期已物化值 → yoy/qoq/cagr/multi_period。
为什么单独节点：L3 跨期跨文档，不在 import/metric_query/generate 算（04g §1.3）。
为什么不解析 calc_formula：表达式求值的缺期/除零/单位边界远大于收益（04g §3.4）。
缺期/不支持标 status 而不是静默丢：不支持 ≠ 数据缺，generate 话术不同（04g §4.2）。
不插值不外推：缺 2023 就不编 2023，编数是造假（04g §4.3）。
"""
from __future__ import annotations

import re

from langgraph.runtime import Runtime
from langgraph.types import StreamWriter

from app.core.logger import logger
from app.metrics.calc import cagr, multi_period, qoq, share, yoy
from app.query_graph.context import QueryAgentContext
from app.query_graph.state import QueryState

CALC_ROUTES = {"calc", "mixed"}
MAX_CALC_RESULTS = 50

# 专有词先于泛化词：「复合增长率」字面含「增长率」(yoy 词)，
# 若按 yoy>qoq>cagr 字面优先级会误判，故 cagr 专有词提前（04g §3.2 语义主 kind）
# share 在 multi_period 前：「构成」是占比不是趋势（04g §3.2）
_KIND_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("unsupported", ("波动率", "相关系数", "夏普", "标准差", "方差")),
    ("cagr", ("CAGR", "cagr", "复合增长", "年均增长")),
    ("yoy", ("同比", "YoY", "yoy", "增长率")),
    ("qoq", ("环比", "QoQ", "qoq")),
    ("share", ("占比", "构成", "比重")),
    ("multi_period", ("变化", "走势", "趋势", "多期")),
)
_NEAR_N_YEAR = re.compile(r"近[0-9一二三四五六七八九十两]+年")


def step_1_work_items(state: QueryState) -> list[tuple[str, str | None]]:
    """待算项 (kind, sub_id)：calc/mixed 子问题按问句判 kind；多期兜底 multi_period。"""
    sub_queries = state.get("sub_queries") or []
    calc_sqs = [sq for sq in sub_queries if sq.get("route") in CALC_ROUTES]
    periods = (state.get("entities") or {}).get("periods") or []

    items: list[tuple[str, str | None]] = []
    for sq in calc_sqs:
        kind = step_2_kind(sq.get("question") or "")
        if kind:
            items.append((kind, sq.get("sub_id")))
    # 多期即使 route=metric_sql 也要序列对比——route_after_metric_query 会把本节点串进来
    if not items and len(periods) > 1:
        items.append(("multi_period", None))
    return items


def step_2_kind(question: str) -> str | None:
    """问句关键词 → L3 kind；无信号返回 None（纯叙述题不该硬凑 L3）。"""
    q = question or ""
    for kind, words in _KIND_RULES:
        if any(w in q for w in words):
            return kind
    if _NEAR_N_YEAR.search(q):
        return "multi_period"
    return None


def step_3_series(metric_rows: list[dict]) -> dict[tuple[str, str], tuple[str, list[tuple[str, float]]]]:
    """(stock_code, metric_code) → (metric_name, [(period, value)] 按期字典序)。

    期间字典序对 YYYY / YYYYQn 成立；None 值不进序列，缺期靠段数体现而不是插位。
    """
    groups: dict[tuple[str, str], tuple[str, list[tuple[str, float]]]] = {}
    for row in metric_rows:
        value = row.get("metric_value")
        if value is None:
            continue
        key = (str(row.get("stock_code")), str(row.get("metric_code")))
        name = str(row.get("metric_name") or key[1])
        if key not in groups:
            groups[key] = (name, [])
        groups[key][1].append((str(row.get("metric_period")), float(value)))
    for _name, series in groups.values():
        series.sort(key=lambda x: x[0])
    return groups


def _years_between(start: str, end: str) -> float | None:
    """期间串取年份差；解析失败给 None，由 cagr 归 insufficient 而不是编 0 年。"""
    try:
        return float(end[:4]) - float(start[:4])
    except ValueError:
        return None


def step_4_apply(kind: str, series: list[tuple[str, float]]) -> tuple[list[str], dict]:
    """单组序列 → (实际参与期次, 指标函数结果)。不支持的 L3 不调函数直接标 status。"""
    periods = [p for p, _ in series]
    if kind == "unsupported":
        return periods, {
            "kind": "unsupported",
            "value": None,
            "values": None,
            "unit": "",
            "status": "unsupported",
            "reason": "暂不支持该 L3，目前仅支持同比/环比/CAGR/多期对比/占比",
            "formula": "",
        }

    if kind in ("yoy", "qoq"):
        if not series:
            return [], {"kind": kind, "value": None, "values": None, "unit": "%", "status": "insufficient_periods", "reason": "无有效序列", "formula": ""}
        current_p, current = series[-1]
        if len(series) >= 2:
            prior_p, prior = series[-2]
            used = [current_p, prior_p]
        else:
            prior = None
            used = [current_p]
        fn = yoy if kind == "yoy" else qoq
        return used, fn(current, prior)

    if kind == "cagr":
        if not series:
            return [], {"kind": "cagr", "value": None, "values": None, "unit": "%", "status": "insufficient_periods", "reason": "无有效序列", "formula": ""}
        start_p, start = series[0]
        end_p, end = series[-1]
        return [start_p, end_p], cagr(start, end, _years_between(start_p, end_p))

    return periods, multi_period(series)


def step_3_share_pairs(
    metric_rows: list[dict],
) -> list[dict]:
    """切片行 → share 计算对（docs/02g §8.3 / 04g §3.3）。

    分母只认 dim_name='合计'；缺合计不猜 0/100，标 insufficient（04g §3.3）。
    同名不同 dim_type 是两个事实，按 (stock, dim_type, metric, period) 分组。
    """
    groups: dict[tuple, dict[str, dict]] = {}
    for row in metric_rows:
        dim_name = str(row.get("dim_name") or "").strip()
        if not dim_name:
            continue
        value = row.get("metric_value")
        if value is None:
            continue
        key = (
            str(row.get("stock_code")),
            str(row.get("dim_type") or ""),
            str(row.get("metric_code") or row.get("metric_name") or ""),
            str(row.get("metric_period")),
        )
        groups.setdefault(key, {})[dim_name] = row

    pairs: list[dict] = []
    for (stock_code, dim_type, metric_key, period), by_name in groups.items():
        total_row = by_name.get("合计")
        total = float(total_row["metric_value"]) if total_row else None
        for dim_name, row in by_name.items():
            if dim_name == "合计":
                continue
            pairs.append(
                {
                    "stock_code": stock_code,
                    "dim_type": dim_type,
                    "dim_name": dim_name,
                    "metric_code": row.get("metric_code"),
                    "metric_name": row.get("metric_name"),
                    "metric_period": period,
                    "part": float(row["metric_value"]),
                    "total": total,
                    "page_no": row.get("page_no"),
                    "sub_id": row.get("sub_id"),
                }
            )
    return pairs


async def node_calc(
    state: QueryState,
    runtime: Runtime[QueryAgentContext],
    writer: StreamWriter = None,
) -> dict:
    empty = {"calc_results": [], "error": None, "error_code": None, "error_stage": None}

    items = step_1_work_items(state)
    if not items:
        return empty

    metric_rows = state.get("metric_rows") or []
    if not metric_rows:
        # 空 metric_rows 合法：叙述题/指标未映射，不是错误（04g §8）
        return empty

    try:
        groups = step_3_series(metric_rows)
        results: list[dict] = []
        for kind, sub_id in items:
            # share 走分片/合计配对，不是多期序列（docs/02g §8.3）
            if kind == "share":
                for pair in step_3_share_pairs(metric_rows):
                    raw = share(pair["part"], pair["total"])
                    results.append(
                        {
                            "stock_code": pair["stock_code"],
                            "metric_code": f"{pair['metric_code'] or pair['metric_name']}_share",
                            "metric_name": f"{pair['dim_name']}{pair['metric_name'] or ''}占比",
                            "kind": "share",
                            "periods": [pair["metric_period"]],
                            "value": raw["value"],
                            "values": None,
                            "unit": raw["unit"],
                            "formula": raw["formula"],
                            "status": raw["status"],
                            "reason": raw["reason"],
                            "sub_id": pair.get("sub_id") or sub_id,
                            "dim_type": pair["dim_type"],
                            "dim_name": pair["dim_name"],
                        }
                    )
                continue
            for (stock_code, code), (name, series) in groups.items():
                used_periods, raw = step_4_apply(kind, series)
                results.append(
                    {
                        "stock_code": stock_code,
                        "metric_code": f"{code}_{raw['kind']}",
                        "metric_name": name,
                        "kind": raw["kind"],
                        "periods": used_periods,
                        "value": raw["value"],
                        "values": raw["values"],
                        "unit": raw["unit"],
                        "formula": raw["formula"],
                        "status": raw["status"],
                        "reason": raw["reason"],
                        "sub_id": sub_id,
                    }
                )
        if len(results) > MAX_CALC_RESULTS:
            logger.warning(f"[node_calc] calc_results {len(results)} 超限截尾至 {MAX_CALC_RESULTS}")
            results = results[:MAX_CALC_RESULTS]

        # SSE 只推 ok：流式层只展示可引用数字，status≠ok 归最终话术（04g §7.2）
        ok_rows = [r for r in results if r["status"] == "ok"]
        if writer is not None and ok_rows:
            writer(
                {
                    "type": "calc",
                    "rows": [
                        {
                            "metric_code": r["metric_code"],
                            "kind": r["kind"],
                            "periods": r["periods"],
                            "value": r["value"],
                            "unit": r["unit"],
                            "status": r["status"],
                        }
                        for r in ok_rows
                    ],
                }
            )
        logger.info(
            f"[node_calc] request_id={state.get('request_id')} "
            f"items={len(items)} results={len(results)} ok={len(ok_rows)}"
        )
        return {"calc_results": results, "error": None, "error_code": None, "error_stage": None}
    except Exception as e:
        # 缺期/不支持已走 status；此处只接真意外（04g §8 禁止吞异常）
        logger.error(f"[node_calc] 内部异常: {e}")
        return {
            "calc_results": [],
            "error": str(e),
            "error_code": "INTERNAL",
            "error_stage": "calc",
        }


if __name__ == "__main__":
    # 纯函数冒烟：不打库不调 LLM，造假 metric_rows 走完整节点。
    # 覆盖：kind 判定、yoy/cagr/多期、unsupported、SSE 只推 ok、空合法。
    import asyncio

    from app.query_graph.state import default_state

    def _row(code, period, value):
        return {
            "stock_code": "000008",
            "metric_code": code,
            "metric_name": {"revenue": "营业收入", "gross_margin": "销售毛利率"}.get(code, code),
            "metric_period": period,
            "metric_value": value,
            "unit": "%",
            "sub_id": "sq1",
        }

    def _state(rows, question, route="calc", periods=("2024",)):
        s = default_state(query=question, request_id="smoke", session_id="smoke")
        s["metric_rows"] = rows
        s["sub_queries"] = [
            {"sub_id": "sq1", "question": question, "route": route, "filters": {}, "hyde_used": False}
        ]
        s["entities"]["periods"] = list(periods)
        return s

    class _W:
        def __init__(self):
            self.events = []

        def __call__(self, item):
            self.events.append(item)

    async def main():
        rt = Runtime(context=QueryAgentContext())

        print("== kind 判定 ==")
        for q in ("2024年营收同比", "复合增长率", "2024Q4环比", "近三年变化", "毛利率波动率", "经营情况"):
            print(f"  {q!r:20} -> {step_2_kind(q)}")

        print("== sq1 yoy 2023/2024 ==")
        w = _W()
        patch = await node_calc(
            _state([_row("revenue", "2023", 100.0), _row("revenue", "2024", 200.0)], "营收同比"),
            rt,
            w,
        )
        r = patch["calc_results"][0]
        print(f"  {r['metric_code']} value={r['value']:.2f}{r['unit']} status={r['status']} periods={r['periods']}")
        print(f"  SSE type={w.events[0]['type']} rows={w.events[0]['rows']}")

        print("== sq2 缺期 + unsupported ==")
        w2 = _W()
        patch2 = await node_calc(
            _state([_row("gross_margin", "2024", 30.0)], "毛利率同比"),
            rt,
            w2,
        )
        r2 = patch2["calc_results"][0]
        print(f"  {r2['metric_code']} status={r2['status']} reason={r2['reason']}")
        print(f"  SSE events（应为空，只推 ok）= {w2.events}")

        print("== sq3 unsupported 波动率 ==")
        patch3 = await node_calc(
            _state([_row("gross_margin", "2024", 30.0)], "毛利率波动率"),
            rt,
            None,
        )
        r3 = patch3["calc_results"][0]
        print(f"  kind={r3['kind']} status={r3['status']} reason={r3['reason']}")

        print("== sq4 多期 series ==")
        patch4 = await node_calc(
            _state(
                [_row("revenue", p, v) for p, v in (("2022", 100.0), ("2023", 150.0), ("2024", 200.0))],
                "近三年变化",
                route="metric_sql",
                periods=("2022", "2023", "2024"),
            ),
            rt,
            None,
        )
        r4 = patch4["calc_results"][0]
        print(f"  {r4['kind']} values={r4['values']} periods={r4['periods']}")

        print("== sq5 空 metric_rows 合法 ==")
        patch5 = await node_calc(_state([], "营收同比"), rt, None)
        print(f"  calc_results={patch5['calc_results']} error={patch5['error']}")

    asyncio.run(main())
