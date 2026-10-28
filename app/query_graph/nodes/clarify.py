"""Q3 node_clarify（docs/04c）。

纯模板追问，不调 LLM；一轮列全缺失项后短路 END。

为什么一轮列全而不是拆轮：clarify→END 无图内循环，拆轮=多次 HTTP 往返，
漏答项下轮 entity 自然再判，期望往返 ≤2；拆轮最差 4 次。
为什么不调 LLM：追问是「缺什么问什么」，模板可单测；LLM 编造候选名比不问更糟。
为什么只发 clarify 不发 error：用户看到的是可行动的追问，不是故障。
"""
from __future__ import annotations

from langgraph.runtime import Runtime
from langgraph.types import StreamWriter

from app.core.logger import logger
from app.query_graph.context import QueryAgentContext
from app.query_graph.matching.stock_name import find_candidates
from app.query_graph.state import QueryState

# 展示上限：超出用「等 N 家」收尾，防止把整页候选砸用户脸上
MAX_CANDIDATES = 3
ERROR_CODE_MISSING = "MISSING_ENTITY"

_COMPANY_ONE = "请问您想查询哪家公司？可以回复公司名称或股票代码。"
_COMPANY_MANY = "请问您想比较哪几家公司？可以回复公司名称或股票代码。"
_METRIC = "请问要查询哪个指标？例如营业收入、毛利率、归母净利润。"
_FALLBACK = "请补充要查询的公司或报告期。"

_PERIOD_BY_INTENT = {
    "metric_calc": "请问要查询哪个报告期？例如 2024 年。",
    "trend": "请问要看哪个时间段？例如近三年。",
    "compare": "请问要对比哪个报告期？例如 2023 年和 2024 年。",
}

# 多缺末尾示例：教用户一句话答全，压低漏答率（04c §5.5）
_EXAMPLE_BY_SLOTS = {
    (True, True, True): "可以一句话回复，例如：贵州茅台 2024 年营业收入。",
    (True, True, False): "可以一句话回复，例如：贵州茅台 2024 年。",
    (True, False, True): "可以一句话回复，例如：贵州茅台 毛利率。",
    (False, True, True): "可以一句话回复，例如：2024 年营业收入。",
}


def _candidate_text(word: str, names: list[str]) -> str:
    """单个未解析词的主体文案；无候选走「未识别」，不编造名字。"""
    if len(names) < 2:
        return f"「{word}」暂无法识别，请换用完整公司简称或 6 位股票代码。"
    shown = names[:MAX_CANDIDATES]
    if len(names) > MAX_CANDIDATES:
        # 「A」、「B」还是「C」等 N 家 —— 3 个展示位固定
        opts = "、".join(f"「{n}」" for n in shown[:-1])
        return f"您指的是{opts}还是「{shown[-1]}」等 {len(names)} 家？建议直接回复股票代码。"
    opts = "、".join(f"「{n}」" for n in shown[:-1])
    return f"您指的是{opts}还是「{shown[-1]}」？也可以直接回复股票代码。"


def _candidate_fragment(word: str, names: list[str]) -> str:
    """多词拼接用的片段：去掉句末标点，由「；另外」连接（04c §5.2）。"""
    if len(names) < 2:
        return f"「{word}」暂无法识别，请换用完整公司简称或 6 位股票代码"
    shown = names[:MAX_CANDIDATES]
    if len(names) > MAX_CANDIDATES:
        opts = "、".join(f"「{n}」" for n in shown[:-1])
        return f"「{word}」您指的是{opts}还是「{shown[-1]}」等 {len(names)} 家？建议直接回复股票代码"
    opts = "、".join(f"「{n}」" for n in shown[:-1])
    return f"「{word}」您指的是{opts}还是「{shown[-1]}」？"


def _subject_text(
    unresolved: list[str], candidates: dict[str, list[str]], intent: str
) -> str | None:
    """主体槽位文案。unresolved 优先于「缺公司」——歧义比缺省更具体。"""
    if unresolved:
        if len(unresolved) == 1:
            return _candidate_text(unresolved[0], candidates.get(unresolved[0]) or [])
        frags = [_candidate_fragment(w, candidates.get(w) or []) for w in unresolved]
        text = "；另外".join(frags)
        return text if text.endswith(("？", "。")) else text + "。"
    return _COMPANY_MANY if intent == "compare" else _COMPANY_ONE


def _build(
    intent: str,
    unresolved: list[str],
    missing: list[str],
    candidates: dict[str, list[str]],
) -> tuple[str, list[str]]:
    """按固定顺序拼装：主体 → 期间 → 指标。顺序保证文案可复现，不是优先级隐藏。"""
    fields: list[str] = []
    items: list[str] = []

    # 槽位 1/2 互斥：有 unresolved 就用歧义文案，不重复问「哪家公司」
    if unresolved or "stock_codes" in missing:
        field = "stock_names" if unresolved else "stock_codes"
        text = _subject_text(unresolved, candidates, intent)
        if text:
            fields.append(field)
            items.append(text)

    if "periods" in missing:
        text = _PERIOD_BY_INTENT.get(intent)
        if text:
            fields.append("periods")
            items.append(text)

    if "metrics" in missing:
        fields.append("metrics")
        items.append(_METRIC)

    if not items:
        return _FALLBACK, []

    if len(items) == 1:
        return items[0], fields

    # 多缺：编号列表 + 一条合并示例，示范「一句话答全」
    lines = ["请补充以下信息："]
    lines.extend(f"{i}. {item}" for i, item in enumerate(items, 1))
    example = _EXAMPLE_BY_SLOTS.get(
        (
            any(f in fields for f in ("stock_names", "stock_codes")),
            "periods" in fields,
            "metrics" in fields,
        )
    )
    if example:
        lines.append(example)
    return "\n".join(lines), fields


async def node_clarify(
    state: QueryState,
    runtime: Runtime[QueryAgentContext],
    writer: StreamWriter = None,
) -> dict:
    entities = state.get("entities") or {}
    unresolved = list(entities.get("unresolved_names") or [])
    missing = list(state.get("missing_fields") or [])
    intent = (state.get("intent") or "").strip()

    candidates: dict[str, list[str]] = {}
    if unresolved:
        repo = runtime.context.stock_dict_repository
        try:
            rows = repo.list_all() if repo is not None else []
            for word in unresolved:
                names = find_candidates(word, rows)
                if names:
                    candidates[word] = names
        except Exception as e:
            # 候选查询失败降级「未识别」模板，不写 INTERNAL——追问仍可发（04c §8）
            logger.warning(f"[node_clarify] 候选查询失败，退未识别文案: {e}")

    msg, fields = _build(intent, unresolved, missing, candidates)

    if not fields:
        # 误入 clarify 是图逻辑 bug：文案给用户下一步，WARN 抓路由（04c §5.6）
        logger.warning(
            f"[node_clarify] request_id={state.get('request_id')} 无缺失字段仍进入 clarify"
        )

    if writer is not None:
        event = {
            "type": "clarify",
            "message": msg,
            "missing_fields": fields,
        }
        if unresolved:
            event["unresolved_names"] = unresolved
        if candidates:
            event["candidates"] = candidates
        writer(event)

    logger.info(
        f"[node_clarify] request_id={state.get('request_id')} "
        f"fields={fields} unresolved={unresolved}"
    )
    return {
        "error": msg,
        "error_code": ERROR_CODE_MISSING,
        "error_stage": "clarify",
    }
