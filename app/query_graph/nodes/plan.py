"""Q4 node_plan（docs/04d）。

一次 json_mode 把 rewritten_query 拆成 sub_queries，标 route + 组 filters。
为什么拆条不在 intent 共享 LLM：entities 那时未解析，filters 是空的（04d §1.3）。
为什么 filters 主体键禁止 LLM 填写：猜码必错，一律继承 entities（04d §3.3）。
为什么 sub_id 重编：LLM id 不可控，SSE/evidence 对齐要求 sq1..sn。
为什么只拆不检：延迟预算留给检索，拆错是文案层问题（04d §1.2）。
为什么 route 只标不裁剪 fan-out：映射只在 route_after_plan 一处维护（04d §5.3）。
"""
from __future__ import annotations

import json

from langgraph.runtime import Runtime
from langgraph.types import StreamWriter

from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.query_graph.context import QueryAgentContext
from app.query_graph.state import QueryState

ROUTE_ENUM = {"vector", "metric_sql", "calc", "mixed"}
MAX_SUB_QUERIES = 8
LLM_MAX_RETRIES = 1
LLM_TIMEOUT_SECONDS = 8  # 与 04a 同值；客户端超时预算，此处不另包 asyncio.wait_for


def step_1_build_prompt(
    rewritten_query: str, entities: dict, resolved_names: dict, intent: str
) -> str:
    """拼 prompt：rewritten_query 是唯一文本依据，entities 摘要只给已解析值。

    不给 unresolved_names——能走到 plan 说明 clarify 已短路；给了反而诱导 LLM 编主体。
    """
    return load_prompt(
        "plan_subqueries",
        rewritten_query=rewritten_query or "",
        intent=intent or "",
        entities_summary=_entities_summary(entities or {}, resolved_names or {}),
    )


def _entities_summary(entities: dict, resolved_names: dict) -> str:
    """公司(代码)/期间/类型/指标，约束 LLM 不得引入新主体新期间。"""
    lines: list[str] = []
    names = resolved_names or {}
    codes = list(entities.get("stock_codes") or [])
    if names:
        pairs = [f"{n}({c})" for n, c in names.items() if n and c]
        if pairs:
            lines.append("公司: " + ", ".join(pairs))
    if codes and not names:
        lines.append("公司代码: " + ", ".join(codes))
    for key, label in (
        ("periods", "报告期"),
        ("report_types", "报告类型"),
        ("metrics", "指标"),
    ):
        vals = [str(v) for v in (entities.get(key) or []) if v]
        if vals:
            lines.append(f"{label}: " + ", ".join(vals))
    return "\n".join(lines) if lines else "（无）"


def step_2_llm_json(llm, prompt: str) -> list:
    """json_mode 调用 + 结构校验；超时/非 JSON/非空数组/route 越界都算失败，重试 1 次。

    route 越界按解析失败整轮重试：route 决定 fan-out 走哪几路，静默丢路比多调一次更糟。
    """
    last_err: Exception | None = None
    for _attempt in range(LLM_MAX_RETRIES + 1):
        try:
            resp = llm.invoke(prompt)
            data = json.loads(resp.content)
            if not isinstance(data, list) or not data:
                raise ValueError(f"LLM 输出不是非空数组: {type(data)}")
            for item in data:
                if not isinstance(item, dict):
                    raise ValueError(f"子问题不是对象: {type(item)}")
                route = (item.get("route") or "").strip()
                if route not in ROUTE_ENUM:
                    raise ValueError(f"route 越出枚举: {item.get('route')!r}")
            return data
        except Exception as e:
            last_err = e
            logger.warning(f"[node_plan] LLM 调用/解析失败: {e}")
    raise last_err if last_err else RuntimeError("LLM 调用失败")


def step_3_assemble(
    raw_list: list, entities: dict, resolved_names: dict
) -> list[dict]:
    """校验丢弃 + 截尾 + 重编 sub_id + 组 filters。question 空是局部降级，全空才失败。"""
    items: list[dict] = []
    for item in raw_list:
        q = item.get("question")
        if not isinstance(q, str) or not q.strip():
            logger.warning("[node_plan] 丢弃空 question 子问题")
            continue
        items.append(
            {
                "question": q.strip(),
                "route": (item.get("route") or "").strip(),
                "sections": _normalize_sections(item.get("sections")),
            }
        )

    if not items:
        raise ValueError("全部子问题 question 非法")

    if len(items) > MAX_SUB_QUERIES:
        logger.warning(
            f"[node_plan] 子问题 {len(items)} 条超过上限，截尾前 {MAX_SUB_QUERIES} 条"
        )
        items = items[:MAX_SUB_QUERIES]

    codes_all = list(entities.get("stock_codes") or [])
    out: list[dict] = []
    for i, it in enumerate(items, 1):
        filters = {
            # 主体键继承 entities；按公司拆出的子问题收窄为该公司的 1 个码
            "stock_codes": _narrow_codes(it["question"], codes_all, resolved_names),
            "report_periods": list(entities.get("periods") or []),
            "report_types": list(entities.get("report_types") or []),
            "sections": it["sections"],
        }
        out.append(
            {
                "sub_id": f"sq{i}",  # 一律重编，不信任 LLM 的 id
                "question": it["question"],
                "route": it["route"],
                "filters": filters,
                "hyde_used": False,
            }
        )
    return out


def _normalize_sections(raw) -> list[str]:
    """sections 是 filters 里唯一可来自 LLM 的键；非列表/非字符串宁缺勿猜。"""
    if not isinstance(raw, list):
        return []
    return [str(s) for s in raw if isinstance(s, str) and s]


def _narrow_codes(
    question: str, codes: list[str], resolved_names: dict
) -> list[str]:
    """question 唯一命中一家公司时收窄 stock_codes；否则继承全集。

    compare 展开要求单条 filter 键干净，但匹配不上时不猜——多码查宽好过查空。
    """
    if len(codes) <= 1:
        return list(codes)
    matched: set[str] = set()
    for name, code in (resolved_names or {}).items():
        if name and name in question:
            matched.add(code)
    for code in codes:
        if code in question:
            matched.add(code)
    if len(matched) == 1:
        return [next(iter(matched))]
    return list(codes)


async def node_plan(
    state: QueryState,
    runtime: Runtime[QueryAgentContext],
    writer: StreamWriter = None,
) -> dict:
    llm = runtime.context.json_llm or runtime.context.llm
    if llm is None:
        logger.error("[node_plan] llm 未注入 QueryAgentContext")
        return {
            "error": "llm 未注入",
            "error_code": "INTERNAL",
            "error_stage": "plan",
        }

    entities = state.get("entities") or {}
    if not entities.get("stock_codes"):
        # clarify 已短路的不变量；仍守一道，避免无主体空转检索
        logger.error(
            f"[node_plan] request_id={state.get('request_id')} stock_codes 为空"
        )
        return {
            "error": "缺少股票代码",
            "error_code": "INTERNAL",
            "error_stage": "plan",
        }

    try:
        prompt = step_1_build_prompt(
            state.get("rewritten_query") or "",
            entities,
            state.get("resolved_names") or {},
            state.get("intent") or "",
        )
        raw_list = step_2_llm_json(llm, prompt)
        sub_queries = step_3_assemble(raw_list, entities, state.get("resolved_names") or {})
    except Exception as e:
        logger.error(f"[node_plan] request_id={state.get('request_id')} 失败: {e}")
        return {
            "error": str(e),
            "error_code": "INTERNAL",
            "error_stage": "plan",
        }

    if writer is not None:
        for sq in sub_queries:
            writer(
                {
                    "type": "sub_query",
                    "sub_id": sq["sub_id"],
                    "question": sq["question"],
                    "status": "running",
                }
            )

    logger.info(
        f"[node_plan] request_id={state.get('request_id')} "
        f"n={len(sub_queries)} routes={[s['route'] for s in sub_queries]} "
        f"questions={[s['question'][:30] for s in sub_queries]}"
    )
    return {
        "sub_queries": sub_queries,
        "error": None,
        "error_code": None,
        "error_stage": None,
    }
