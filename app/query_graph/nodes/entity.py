"""Q2 node_entity（docs/04a）。

复用 node_intent 的共享 LLM 结果；名称走 stock_dict 四级匹配；
periods 绝对化（基准=最新可查报告期）；指代消解；Redis 回写本轮 user 条。

为什么匹配/绝对化/指代都拆纯函数：可单测到每条规则，节点只做编排。
为什么 Redis 写失败仅 WARN：解析结果是增强，不是正确性前提。
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from langgraph.runtime import Runtime
from langgraph.types import StreamWriter

from app.core.logger import logger
from app.query_graph.context import QueryAgentContext
from app.query_graph.matching.coref import carry_forward_from_history, resolve_coreference
from app.query_graph.matching.periods import absolutize_periods, normalize_report_types
from app.query_graph.matching.stock_name import match_stock_names
from app.query_graph.state import QueryState
from app.repositories.redis_chat_history import get_redis_chat_history

# metric_calc / trend 必须有指标；compare / trend 用多期对比，缺期数没法算
_INTENT_NEED_METRICS = {"metric_calc", "trend"}
_INTENT_NEED_PERIODS = {"compare", "metric_calc", "trend"}


def step_1_shared_llm_result(state: QueryState) -> dict:
    """取 node_intent 暂存的 LLM JSON；缺失视为内部错误（intent 应已写入）。"""
    raw = state.get("intent_raw") or {}
    if not raw:
        raise ValueError("intent_raw 缺失：node_intent 应先写入共享 LLM 结果")
    return raw


def step_5_missing_fields(intent: str, entities: dict, missing: list[str]) -> list[str]:
    """缺关键字段 → clarify 追问（04a §3.5），非异常路径。"""
    if entities.get("unresolved_names"):
        missing.append("stock_names")
    if intent in _INTENT_NEED_PERIODS and not entities.get("periods"):
        missing.append("periods")
    if intent in _INTENT_NEED_METRICS and not entities.get("metrics"):
        missing.append("metrics")
    if not entities.get("stock_codes"):
        missing.append("stock_codes")
    return missing


def _latest_period(runtime: Runtime[QueryAgentContext]) -> str:
    """基准期：最新可查报告期；查不到用当前年并 WARN（04a §5.1）。"""
    repo = runtime.context.financial_metric_repository
    if repo is not None:
        try:
            latest = repo.latest_report_period()
            if latest and str(latest).isdigit():
                return str(latest)
        except Exception as e:
            logger.warning(f"[node_entity] 基准期查询失败，改用自然年: {e}")
    return str(datetime.now(ZoneInfo("Asia/Shanghai")).year)


async def node_entity(
    state: QueryState,
    runtime: Runtime[QueryAgentContext],
    writer: StreamWriter = None,
) -> dict:
    try:
        raw = step_1_shared_llm_result(state)
    except ValueError as e:
        logger.error(f"[node_entity] {e}")
        return {"error": str(e), "error_code": "INTERNAL", "error_stage": "entity"}

    query = state.get("query") or ""
    history = state.get("history") or []
    names = list(raw.get("stock_names") or [])
    periods_raw = list(raw.get("periods") or [])
    report_types_raw = list(raw.get("report_types") or [])
    metrics = list(raw.get("metrics") or [])
    rewritten = (raw.get("rewritten_query") or query).strip()
    intent = (raw.get("intent") or state.get("intent") or "").strip()

    # step_2 名称四级匹配
    resolved: dict[str, str] = {}
    unresolved: list[str] = []
    repo = runtime.context.stock_dict_repository
    try:
        rows = repo.list_all() if repo is not None else []
        resolved, unresolved = match_stock_names(names, rows)
    except Exception as e:
        # 问句有 6 位码则直用，否则依赖不可用（04a §8）
        codes_in_query = [n for n in names if n.isdigit() and len(n) == 6]
        if not codes_in_query:
            logger.error(f"[node_entity] stock_dict 不可用且无直用代码: {e}")
            return {
                "error": str(e),
                "error_code": "DEPENDENCY_UNAVAILABLE",
                "error_stage": "entity",
            }
        logger.warning(f"[node_entity] stock_dict 查询失败，跳过字典校验: {e}")
        resolved = {c: c for c in codes_in_query}
        unresolved = [n for n in names if n not in resolved]

    stock_codes = list(dict.fromkeys(resolved.values()))
    stock_names = [n for n in names if n in resolved]

    # step_3 periods 绝对化（基准=最新可查报告期）
    latest = _latest_period(runtime)
    periods = absolutize_periods(periods_raw, latest)
    report_types = normalize_report_types(report_types_raw)

    # step_4 指代消解 + 无新实体时沿用历史（04a §4.4 / §6）
    coref = resolve_coreference(query, history)
    hist_codes, hist_names, hist_periods = carry_forward_from_history(history)
    if coref.codes:
        stock_codes = list(dict.fromkeys(stock_codes + coref.codes))
        stock_names = list(dict.fromkeys(stock_names + coref.names))
        if coref.rewritten_query:
            rewritten = coref.rewritten_query
    elif not names:
        stock_codes = list(dict.fromkeys(stock_codes + hist_codes))
        stock_names = list(dict.fromkeys(stock_names + hist_names))
    # 期间盲沿用（独立于指代触发）：本轮没写期间就是还在聊上一轮那期（04b §4.4）
    if not periods:
        periods = list(hist_periods)

    entities = {
        "stock_names": stock_names,
        "stock_codes": stock_codes,
        "unresolved_names": unresolved,
        "periods": periods,
        "report_types": report_types,
        "metrics": metrics,
    }

    # step_5 缺关键字段
    missing = step_5_missing_fields(intent, entities, [])

    # step_6 Redis 回写（失败仅 WARN，不阻断）
    try:
        history = get_redis_chat_history(state.get("session_id") or "")
        history.add_user_message(query, stock_codes=stock_codes, periods=periods)
    except Exception as e:
        logger.warning(f"[node_entity] Redis 回写失败，忽略: {e}")

    if writer is not None:
        writer(
            {
                "type": "entities",
                "stock_codes": stock_codes,
                "periods": periods,
                "unresolved_names": unresolved,
            }
        )

    logger.info(
        f"[node_entity] request_id={state.get('request_id')} "
        f"codes={stock_codes} unresolved={unresolved} periods={periods} missing={missing}"
    )
    return {
        "intent": intent,
        "entities": entities,
        "resolved_names": resolved,
        "missing_fields": missing,
        "rewritten_query": rewritten,
        "error": None,
        "error_code": None,
        "error_stage": None,
    }

if __name__ == "__main__":
    # 手工冒烟：假字典/假基准期跑 3 条典型句（别名+近三年 / 指代沿用 / 歧义缺字段）
    # 不依赖 MySQL；Redis 不可用只 WARN，正好顺带看降级
    import asyncio

    from langgraph.runtime import Runtime

    from app.entities.stock_dict import StockDictDbData
    from app.query_graph.state import default_state
    from app.repositories.financial_metric_repository import FinancialMetricRepository
    from app.repositories.stock_dict_repository import StockDictRepository

    class _FakeDict(StockDictRepository):
        def __init__(self, rows):
            self.rows = rows

        def list_all(self):
            return self.rows

    class _FakeMetrics(FinancialMetricRepository):
        def __init__(self):
            pass  # 不建 session，冒烟只用 latest_report_period

        def latest_report_period(self):
            return "2024"

    ROWS = [
        StockDictDbData("600519", "贵州茅台", "贵州茅台酒股份有限公司", ["茅台", "贵州茅台酒"], "SH"),
        StockDictDbData("000008", "神州高铁", "神州高铁技术股份有限公司", [], "SZ"),
        StockDictDbData("000007", "全新好", "全新好股份有限公司", [], "SZ"),
    ]

    async def _main() -> None:
        runtime = Runtime(  # type: ignore[arg-type]
            context=QueryAgentContext(
                stock_dict_repository=_FakeDict(ROWS),
                financial_metric_repository=_FakeMetrics(),
            )
        )

        cases = [
            {
                "query": "茅台近三年毛利率",
                "history": [],
                "intent_raw": {
                    "intent": "trend",
                    "stock_names": ["茅台"],
                    "periods": ["近三年"],
                    "metrics": ["毛利率"],
                    "rewritten_query": "贵州茅台近三年毛利率变化",
                },
            },
            {
                "query": "它的毛利率是多少",
                "history": [
                    {
                        "role": "user",
                        "content": "神州高铁2024年经营情况",
                        "stock_codes": ["000008"],
                        "stock_names": ["神州高铁"],
                    }
                ],
                "intent_raw": {
                    "intent": "metric_calc",
                    "stock_names": [],
                    "periods": ["2024"],
                    "metrics": ["毛利率"],
                    "rewritten_query": "它的毛利率是多少",
                },
            },
            {
                "query": "茅台股份的营收",
                "history": [],
                "intent_raw": {
                    "intent": "metric_calc",
                    "stock_names": ["茅台股份"],
                    "periods": ["2024"],
                    "metrics": ["营收"],
                    "rewritten_query": "茅台股份2024年营收",
                },
            },
        ]

        for i, case in enumerate(cases, 1):
            state = default_state(
                query=case["query"],
                request_id=f"smoke-{i}",
                session_id="smoke",
                history=case["history"],
            )
            state["intent_raw"] = case["intent_raw"]
            patch = await node_entity(state, runtime, None)
            ent = patch.get("entities") or {}
            print(f"\n--- case {i} ---")
            print(f"Q     : {case['query']}")
            print(f"codes : {ent.get('stock_codes')} periods={ent.get('periods')}")
            print(f"unres : {ent.get('unresolved_names')} missing={patch.get('missing_fields')}")
            print(f"meta  : resolved={patch.get('resolved_names')} rewrite={patch.get('rewritten_query')}")

    asyncio.run(_main())