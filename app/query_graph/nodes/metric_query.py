"""Q6 node_metric_query 分支 B（docs/04f）。

词→码映射 + 一次 SQL 取回 L1/L2 已物化值。
为什么不走 text-to-SQL：LLM 写码必错，词→码是查表（04f §6）。
为什么只取 L1/L2 不现算 L2：02e S6 已物化，再算就是两套口径。
为什么不算同比/CAGR：L3 归 node_calc，越权计算口径不一致。
为什么 SQL 不 fan-out：IN 精确过滤无排序挤占，fan-out 只是多打库。
"""
from __future__ import annotations

from langgraph.runtime import Runtime
from langgraph.types import StreamWriter

from app.core.logger import logger
from app.query_graph.context import QueryAgentContext
from app.query_graph.matching.metric_map import map_metric_words, pick_metric_row
from app.query_graph.state import QueryState

VECTOR_ROUTES = {"metric_sql", "mixed"}
MAX_ROWS = 200


def _known_terms_in(text: str, definitions: list) -> list[str]:
    """text 里出现的已知指标名/别名，长词优先。"""
    hits: list[str] = []
    for d in definitions:
        names = [d.metric_name] + [a for a in (d.aliases or "").split("|") if a]
        for n in names:
            if n and n in text and n not in hits:
                hits.append(n)
    hits.sort(key=len, reverse=True)
    return hits


def step_1_metric_words(state: QueryState, definitions: list) -> list[str]:
    """指标词两处来源合并：entities.metrics + 子问题词面（问句中出现的已知名/别名）。

    词面是「找已知指标名出现在文本里」，不是拿整句去映射——整句包含匹配必误伤。
    长词优先去碎片：问句含「归母净利润」时「净利润」不再进词表，
    否则 net_profit 被 net_profit_attributable 的子串误激活（真库两码并存）。
    实体指标词同样先抽已知名再进词表：intent 会给粘连短语
    「归母净利润与扣非净利润」，原文直接进 words 会把子问题扫出的真词当子串跳过，
    最后 name-in-word 只剩「净利润」→ net_profit 查空（600467+601225 实测）。
    抽不出已知名的原词保留，交给 map 的包含匹配兜底。
    """
    words: list[str] = []
    for metric in (state.get("entities") or {}).get("metrics") or []:
        hits = _known_terms_in(metric, definitions) or ([metric] if metric else [])
        for n in hits:
            if any(n in w for w in words):
                continue
            words.append(n)
    for sq in state.get("sub_queries") or []:
        if sq.get("route") not in VECTOR_ROUTES:
            continue
        for n in _known_terms_in(sq.get("question") or "", definitions):
            if any(n in w for w in words):
                continue
            words.append(n)
    return words


def step_2_map_codes(words: list[str], definitions: list) -> tuple[list[str], list[str]]:
    """词→码三层映射；未映射先留着给 step_6 扫业务词（04f §3.6）。"""
    codes, unmapped = map_metric_words(words, definitions)
    return codes, unmapped


def step_6_business_words(
    unmapped: list[str], fact_rows: list[dict]
) -> tuple[list[dict], list[str]]:
    """未映射词 → business_facts.dim_name 匹配（docs/04f §3.6 / 02g §8）。

    为什么业务词在映射失败之后：「利润」这类碰巧出现在业务名里的词先被字典截住，
    反过来先扫 dim_name 会误伤（04f §3.6）。
    精确优先于包含：「煤炭」不该在已有「煤炭采掘业」精确命中时再捞一堆。
    """
    if not unmapped or not fact_rows:
        return [], list(unmapped)

    exact: list[str] = []
    contains: list[str] = []
    for word in unmapped:
        if any(r.get("dim_name") == word for r in fact_rows):
            exact.append(word)
        elif len(word) >= 2 and any(
            word in str(r.get("dim_name") or "") or str(r.get("dim_name") or "") in word
            for r in fact_rows
        ):
            contains.append(word)

    hit_words = exact + contains
    still_unmapped = [w for w in unmapped if w not in hit_words]
    if not hit_words:
        return [], still_unmapped

    picked: list[dict] = []
    for r in fact_rows:
        dim_name = str(r.get("dim_name") or "")
        if dim_name in exact:
            picked.append(r)
        elif any(w in contains for w in (dim_name,)) or any(
            w in dim_name or dim_name in w for w in contains
        ):
            picked.append(r)
    return picked, still_unmapped


def step_4_adjudicate(rows: list[dict]) -> list[dict]:
    """按 (stock_code, metric_code, metric_period) 分组裁决，每组取 1 行（04f §5.1）。"""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r.get("metric_value") is None:
            logger.warning(f"[node_metric_query] 丢弃空值行 code={r.get('metric_code')} period={r.get('metric_period')}")
            continue
        key = (r.get("stock_code"), r.get("metric_code"), r.get("metric_period"))
        groups.setdefault(key, []).append(r)

    picked: list[dict] = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        row = pick_metric_row(groups[key])
        if row is not None:
            picked.append(row)
    return picked


async def node_metric_query(
    state: QueryState,
    runtime: Runtime[QueryAgentContext],
    writer: StreamWriter = None,
) -> dict:
    sub_queries = [
        sq for sq in (state.get("sub_queries") or []) if sq.get("route") in VECTOR_ROUTES
    ]
    if not sub_queries:
        logger.warning(f"[node_metric_query] request_id={state.get('request_id')} 无 metric_sql/mixed 子问题")
        return {"metric_rows": [], "error": None, "error_code": None, "error_stage": None}

    entities = state.get("entities") or {}
    stock_codes = list(entities.get("stock_codes") or [])
    if not stock_codes:
        logger.error(f"[node_metric_query] stock_codes 为空")
        return {
            "metric_rows": [],
            "error": "stock_codes 为空",
            "error_code": "INTERNAL",
            "error_stage": "metric_query",
        }

    metric_repo = runtime.context.financial_metric_repository
    def_repo = runtime.context.metric_definition_repository
    fact_repo = runtime.context.business_fact_repository
    if metric_repo is None or def_repo is None:
        return {
            "metric_rows": [],
            "error": "MySQL 不可用",
            "error_code": "DEPENDENCY_UNAVAILABLE",
            "error_stage": "metric_query",
        }

    # step_1 + step_2 词→码（扫词面要字典，先取再扫）
    try:
        definitions = def_repo.list_all()
        words = step_1_metric_words(state, definitions)
        metric_codes, unmapped = step_2_map_codes(words, definitions)
    except Exception as e:
        logger.error(f"[node_metric_query] 字典查询失败: {e}")
        return {
            "metric_rows": [],
            "error": str(e),
            "error_code": "DEPENDENCY_UNAVAILABLE",
            "error_stage": "metric_query",
        }

    periods = list(entities.get("periods") or [])
    report_types = list(entities.get("report_types") or [])

    # step_3 + step_4 一次 SQL + 口径裁决
    picked: list[dict] = []
    if metric_codes:
        try:
            raw_rows = metric_repo.query_by_codes(stock_codes, metric_codes, periods, report_types)
        except Exception as e:
            logger.error(f"[node_metric_query] MySQL 查询失败: {e}")
            return {
                "metric_rows": [],
                "error": str(e),
                "error_code": "DEPENDENCY_UNAVAILABLE",
                "error_stage": "metric_query",
            }
        picked = step_4_adjudicate(raw_rows)
        if len(picked) > MAX_ROWS:
            logger.warning(f"[node_metric_query] 行数 {len(picked)} 超限截尾至 {MAX_ROWS}")
            picked = picked[:MAX_ROWS]
    else:
        logger.warning(f"[node_metric_query] 无可查 metric_code words={words}")

    # step_6 未映射业务词 → business_facts（docs/04f §3.6）；有 dim 才补切片行
    fact_picked: list[dict] = []
    still_unmapped = list(unmapped)
    if unmapped and fact_repo is not None:
        try:
            all_facts = fact_repo.query_by_dim_names(stock_codes, [], periods, report_types)
            fact_picked, still_unmapped = step_6_business_words(unmapped, all_facts)
            # 度量列跟指标词走：有指标码时只留这些码/这些词命中的列（04f §3.6）。
            # 词命中用精确等值不用包含——「营业收入」会把「营业收入比上年增减(%)」
            # 误留下，那是披露增减列不是本期值（02g §5.2）
            if fact_picked and metric_codes:
                fact_picked = [
                    r
                    for r in fact_picked
                    if (r.get("metric_code") in metric_codes)
                    or any(w == str(r.get("metric_name") or "") for w in words)
                ]
        except Exception as e:
            logger.error(f"[node_metric_query] business_facts 查询失败: {e}")
            return {
                "metric_rows": [],
                "error": str(e),
                "error_code": "DEPENDENCY_UNAVAILABLE",
                "error_stage": "metric_query",
            }
    if still_unmapped:
        logger.warning(f"[node_metric_query] 未映射词丢弃: {still_unmapped}")

    # step_5 归一 + SSE
    sub_id = sub_queries[0].get("sub_id") if len(sub_queries) == 1 else None
    metric_rows = [
        {
            "stock_code": r.get("stock_code"),
            "metric_code": r.get("metric_code"),
            "metric_name": r.get("metric_name"),
            "metric_period": r.get("metric_period"),
            "metric_period_type": r.get("metric_period_type"),
            "metric_value": float(r["metric_value"]),
            "unit": r.get("unit"),
            "statement_scope": r.get("statement_scope"),
            "page_no": r.get("page_no"),
            "sub_id": sub_id,
        }
        for r in picked
    ]
    for r in fact_picked:
        metric_rows.append(
            {
                "stock_code": r.get("stock_code"),
                "metric_code": r.get("metric_code"),
                "metric_name": r.get("metric_name"),
                "metric_period": r.get("metric_period"),
                "metric_period_type": "current",
                "metric_value": float(r["metric_value"]) if r.get("metric_value") is not None else None,
                "unit": r.get("unit"),
                "statement_scope": None,
                "page_no": r.get("page_no"),
                "sub_id": sub_id,
                # 切片定位键：fuse content 前缀、share 分子分母都靠它（02g §8.2）
                "dim_type": r.get("dim_type"),
                "dim_name": r.get("dim_name"),
            }
        )

    if writer is not None and metric_rows:
        flat: dict[str, float] = {}
        for row in metric_rows:
            if row["metric_code"] not in flat:
                flat[row["metric_code"]] = row["metric_value"]
        writer(
            {
                "type": "metric",
                "rows": [
                    {
                        "stock_code": metric_rows[0]["stock_code"],
                        "period": metric_rows[0]["metric_period"],
                        **flat,
                    }
                ],
            }
        )

    logger.info(
        f"[node_metric_query] request_id={state.get('request_id')} "
        f"words={words} codes={metric_codes} rows={len(metric_rows)}"
    )
    return {
        "metric_rows": metric_rows,
        "error": None,
        "error_code": None,
        "error_stage": None,
    }


if __name__ == "__main__":
    # 真实链路冒烟：真实 MySQL financial_metrics + metric_definitions。
    # 前置：metric_normalize 已物化 L1/L2（02e S6）；.env 有 MySQL 配置。
    # 覆盖：精确名/别名/L2、未映射丢弃、多期、口径裁决、SSE metric 形状。
    import asyncio

    from langgraph.runtime import Runtime

    from app.clients.mysql_client import MySQLClientManager
    from app.query_graph.context import QueryAgentContext
    from app.query_graph.state import default_state
    from app.repositories.financial_metric_repository import FinancialMetricRepository
    from app.repositories.metric_definition_repository import MetricDefinitionRepository

    mgr = MySQLClientManager()
    mgr.init()
    session = mgr.session()
    metric_repo = FinancialMetricRepository(session)
    def_repo = MetricDefinitionRepository(session)

    defs = def_repo.list_all()
    print(f"== MySQL metric_definitions {len(defs)} 条 ==")
    assert defs, "字典为空，先跑 metric_normalize 的 _seed_definitions"

    # 挑一只有数的票，不写死库态（不同机器导入进度不同）
    candidates = [ ("000068", "华控赛格")]
    stock_code, stock_name = "", ""
    all_periods: list[str] = []
    for code, name in candidates:
        probe = metric_repo.query_by_codes([code], ["revenue"], [], [])
        if probe:
            stock_code, stock_name = code, name
            all_periods = sorted({str(r["metric_period"]) for r in probe if r.get("metric_period")})
            break
    print(f"== financial_metrics hit {stock_name}({stock_code}) periods={all_periods} ==")
    assert stock_code, "financial_metrics 无 revenue 行，先跑 import 流水（02e）"
    # 优先 YYYY 年报期：sorted 后 2024Q4 会排在 2024 后面，单季缺 L2
    annual = [p for p in all_periods if p.isdigit()]
    period = annual[-1] if annual else all_periods[-1]
    assert period, "revenue 行缺 metric_period"

    class _Writer:
        def __init__(self):
            self.events: list[dict] = []

        def __call__(self, item: dict) -> None:
            self.events.append(item)

    def _build_state(metrics: list[str], question: str, sub_id: str, sel: list[str]):
        state = default_state(query=question, request_id=f"smoke-{sub_id}", session_id="smoke")
        state["sub_queries"] = [  # type: ignore[typeddict-item]
            {
                "sub_id": sub_id,
                "question": question,
                "route": "metric_sql",
                "filters": {
                    "stock_codes": [stock_code],
                    "report_periods": sel,
                    "report_types": ["annual"],
                    "sections": [],
                },
                "hyde_used": False,
            }
        ]
        state["entities"] = {  # type: ignore[typeddict-item]
            "stock_names": [stock_name],
            "stock_codes": [stock_code],
            "unresolved_names": [],
            "periods": sel,
            "report_types": ["annual"],
            "metrics": metrics,
        }
        return state

    async def _main() -> None:
        runtime = Runtime(  # type: ignore[arg-type]
            context=QueryAgentContext(
                financial_metric_repository=metric_repo,
                metric_definition_repository=def_repo,
            )
        )

        async def _run(metrics, question, sub_id, sel):
            writer = _Writer()
            patch = await node_metric_query(_build_state(metrics, question, sub_id, sel), runtime, writer)
            return patch, writer

        # 用例1 精确名 + 别名 + L2：一次取回三行
        patch, writer = await _run(
            ["营业收入", "归母净利润", "毛利率"],
            f"{stock_name}{period}年营业收入、归母净利润和毛利率是多少？",
            "sq1",
            [period],
        )
        print(f"== sq1 error={patch.get('error_code')} rows={len(patch['metric_rows'])} ==")
        for r in patch["metric_rows"]:
            print(
                f"   {r['metric_code']}({r['metric_name']}) "
                f"period={r['metric_period']} value={r['metric_value']} "
                f"scope={r['statement_scope']} ptype={r['metric_period_type']} page={r['page_no']}"
            )
        assert patch["error_code"] is None, patch.get("error")
        codes = {r["metric_code"] for r in patch["metric_rows"]}
        assert "revenue" in codes, codes
        assert "net_profit_attributable" in codes, codes
        assert "gross_margin" in codes, codes
        assert writer.events and writer.events[0]["type"] == "metric", writer.events
        flat = writer.events[0]["rows"][0]
        print(f"== SSE metric flat = {flat} ==")

        # 用例2 未映射词丢弃 + 已知词仍可查（词面扫描）
        patch2, _ = await _run(
            ["重要指标"],
            f"{stock_name}{period}年营业收入表现如何？",
            "sq2",
            [period],
        )
        print(f"== sq2 rows={len(patch2['metric_rows'])} ==")
        assert patch2["error_code"] is None
        assert patch2["metric_rows"], "词面应扫到「营业收入」"
        assert patch2["metric_rows"][0]["metric_code"] == "revenue"

        # 用例3 无任何指标词：空 rows 合法，不打库报错（04f §5.3）
        patch3, writer3 = await _run(
            [],
            f"{stock_name}经营情况",
            "sq3",
            [period],
        )
        print(f"== sq3 rows={len(patch3['metric_rows'])} events={len(writer3.events)} ==")
        assert patch3["error_code"] is None
        assert patch3["metric_rows"] == []
        assert writer3.events == []

        # 用例4 多期：行内 period 唯一；优先两个年报期（同比场景）
        multi = annual[-2:] if len(annual) >= 2 else all_periods[-2:]
        if len(multi) >= 2:
            sel = multi
            patch4, _ = await _run(
                ["营业收入"],
                f"{stock_name}{sel[0]}年和{sel[1]}年营业收入是多少？",
                "sq4",
                sel,
            )
            got = sorted({r["metric_period"] for r in patch4["metric_rows"]})
            print(f"== sq4 periods={got} rows={len(patch4['metric_rows'])} ==")
            assert got == sel, got

    asyncio.run(_main())
    print("== smoke ok ==")
