"""Q10 node_generate（docs/04j）。

编号 context + 强制 CoT + [n] 引用 + 流式 token。
为什么强制 CoT「列事实再结论」而不是逐步推理：逐步推理会诱导模型在推理链里做算术，
列事实只是搬运带 [n] 的原文陈述，数字对账时每句都能落回 evidence（04j §4.2）。
为什么 answer 保留完整 CoT：validate 对全文数字对账，推理段的幻觉数字也拦得住（04j §1.2）。
为什么 citations 生成后一次性抽取而不是流式抽：[n] 可能跨 chunk 到达，流式正则会漏（04j §1.3）。
为什么 context 用 ref 编号不重编：validate 用 answer[n] ↔ evidence.ref 对账（04j §3.1）。
为什么截断整条不切半条：半截数字比丢弃更危险（04j §3.1）。
为什么越界 [n] 丢弃而不是整条作废：少引优于错引（04j §3.4）。
为什么不发 done：终态事件归 validate（04j §3.5）。
"""
from __future__ import annotations

import re

from langgraph.runtime import Runtime
from langgraph.types import StreamWriter

from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.query_graph.context import QueryAgentContext
from app.query_graph.state import QueryState
from app.repositories.redis_chat_history import get_redis_chat_history

__all__ = [
    "HISTORY_TURNS",
    "LLM_MAX_RETRIES",
    "MAX_CONTEXT_CHARS",
    "node_generate",
]

MAX_CONTEXT_CHARS = 12000
HISTORY_TURNS = 4
LLM_MAX_RETRIES = 1

# evidence=[] 是合法终态不是故障：说「没找到」比编造诚实（04j §8）
EMPTY_EVIDENCE_MESSAGE = "未找到相关年报证据，无法回答该问题。"

_VECTOR_TYPES = ("narrative", "table")


def _title_of(e: dict) -> str:
    """向量类填 stock_code+report_period；metric/calc 填 stock_code+metric_code（04j §3.1）。

    evidence 无独立 title 字段（fuse 未搬运 hits 的 title），用可定位的元组代替。
    metric/calc 必须带 stock_code：compare 题两家同码指标只靠 metric_code 区分会串引用。
    """
    if e.get("type") not in _VECTOR_TYPES:
        return f"{e.get('stock_code') or ''} {e.get('metric_code') or ''}".strip()
    return f"{e.get('stock_code') or ''} {e.get('report_period') or ''}".strip()


def step_1_build_context(evidence: list[dict]) -> tuple[str, int]:
    """编号拼接 + 预算截断。返回 (context, truncated_count)。

    为什么按 evidence 列表顺序拼接：rerank 后已按相关度排好，metric/calc 在尾部。
    """
    parts: list[str] = []
    budget = 0
    truncated = 0
    for i, e in enumerate(evidence):
        content = e.get("content") or ""
        if budget + len(content) > MAX_CONTEXT_CHARS:
            truncated = len(evidence) - i
            break
        budget += len(content)
        score = float(e.get("score") or 0.0)
        header = (
            f"[{e.get('ref')}][type={e.get('type')}][source={e.get('source')}]"
            f"[title={_title_of(e)}][score={score:.2f}]"
        )
        parts.append(f"{header}\n{content}")
    return "\n\n".join(parts), truncated


def _format_history(history: list[dict]) -> str:
    items = (history or [])[-HISTORY_TURNS:]
    if not items:
        return "无历史对话"
    lines = []
    for m in items:
        role = "用户" if m.get("role") == "user" else "助手"
        lines.append(f"{role}: {m.get('content') or ''}")
    return "\n".join(lines)


def _format_entities(entities: dict) -> str:
    codes = entities.get("stock_codes") or []
    periods = entities.get("periods") or []
    parts = []
    if codes:
        parts.append(f"股票：{', '.join(codes)}")
    if periods:
        parts.append(f"期间：{', '.join(periods)}")
    return "；".join(parts) if parts else "无"


def step_2_build_prompt(state: QueryState, context: str) -> str:
    """渲染 generate_answer.prompt；question 优先 rewritten_query（指代已消解）。"""
    return load_prompt(
        "generate_answer",
        question=state.get("rewritten_query") or state.get("query") or "",
        context=context,
        history=_format_history(state.get("history") or []),
        entities=_format_entities(state.get("entities") or {}),
        intent=state.get("intent") or "",
    )


def step_3_stream(llm, prompt: str, writer: StreamWriter | None) -> str:
    """流式生成 + token SSE；异常重试 1 次。

    为什么 token 全量推送含 CoT：流式层不替用户做内容裁剪（04j §1.2）。
    为什么重试期间已推送 token 不撤回：SSE 无事务，answer 以重试成功结果为准（04j §3.3）。
    """
    last_err: Exception | None = None
    for _attempt in range(LLM_MAX_RETRIES + 1):
        try:
            answer = ""
            for chunk in llm.stream(prompt):
                delta = getattr(chunk, "content", None) or ""
                answer += delta
                if writer is not None and delta:
                    writer({"type": "token", "content": delta})
            return answer
        except Exception as e:
            last_err = e
            logger.warning(f"[node_generate] 流式生成失败: {e}")
    raise last_err if last_err else RuntimeError("LLM 流式生成失败")


def step_4_parse_citations(answer: str, evidence: list[dict]) -> tuple[list[dict], int]:
    """解析 [n] → citations。返回 (citations, 越界丢弃数)。

    为什么只收 evidence 中存在的 ref：LLM 瞎写的 [99] 丢弃 + WARN（04j §3.4）。
    为什么按 answer 首现序不是 evidence 序：citation 事件应反映行文引用顺序。
    """
    by_ref = {e.get("ref"): e for e in evidence}
    refs: list[int] = []
    dropped = 0
    for n in re.findall(r"\[(\d+)\]", answer or ""):
        ref = int(n)
        if ref not in by_ref:
            dropped += 1
            continue
        if ref not in refs:
            refs.append(ref)
    citations = []
    for ref in refs:
        e = by_ref[ref]
        citations.append(
            {
                "ref": e.get("ref"),
                "document_id": e.get("document_id"),
                "stock_code": e.get("stock_code"),
                "report_period": e.get("report_period"),
                "page_no": e.get("page_no"),
                "snippet": e.get("snippet"),
                "type": e.get("type"),
            }
        )
    return citations, dropped


def _has_digits(s: str) -> bool:
    return bool(re.search(r"\d", s or ""))


async def node_generate(
    state: QueryState,
    runtime: Runtime[QueryAgentContext],
    writer: StreamWriter = None,
) -> dict:
    llm = runtime.context.llm
    if llm is None:
        logger.error("[node_generate] llm 未注入 QueryAgentContext")
        return {
            "answer": "",
            "citations": [],
            "error": "llm 未注入",
            "error_code": "DEPENDENCY_UNAVAILABLE",
            "error_stage": "generate",
        }

    evidence = state.get("evidence") or []
    # 空证据分支不走引用路径：固定话术 + citations=[]（04j §8）
    # 必须 writer 推 token：前端靠 token 事件渲染答案，不推则页面空白
    if not evidence:
        # 实体已解析（stock_dict 命中）时带出公司信息，别一刀切「未找到」
        entities = state.get("entities") or {}
        names = entities.get("stock_names") or []
        codes = entities.get("stock_codes") or []
        if codes:
            label = "、".join(
                f"{n}（{c}）" if n else c for n, c in zip(names + [""] * len(codes), codes)
            )
            msg = f"已识别{label}，但未在已导入的年报中找到相关内容。"
        else:
            msg = EMPTY_EVIDENCE_MESSAGE
        if writer is not None:
            writer({"type": "token", "content": msg})
        return {
            "answer": msg,
            "citations": [],
            "error": None,
            "error_code": None,
            "error_stage": None,
        }

    try:
        context, truncated = step_1_build_context(evidence)
        if truncated:
            logger.warning(f"[node_generate] context_truncated={truncated}")

        prompt = step_2_build_prompt(state, context)

        answer = step_3_stream(llm, prompt, writer)
        citations, dropped = step_4_parse_citations(answer, evidence)
        if dropped:
            logger.warning(f"[node_generate] 越界 [n] 丢弃 count={dropped}")

        # 含数字却零引用 → 重试 1 次；仍无则原样落 answer 交 validate 拦（04j §3.4/§8）
        if _has_digits(answer) and not citations:
            logger.warning("[node_generate] 含数字却零引用，重试生成")
            try:
                retry_answer = step_3_stream(llm, prompt, writer)
                retry_citations, retry_dropped = step_4_parse_citations(retry_answer, evidence)
                answer = retry_answer
                citations = retry_citations
                if retry_dropped:
                    logger.warning(f"[node_generate] 重试后越界 [n] 丢弃 count={retry_dropped}")
                if _has_digits(answer) and not citations:
                    logger.warning("[node_generate] 重试后仍无引用，原样落盘交 validate 兜底")
            except Exception as e:
                # 重试失败保留首次结果：已生成的答案可交 validate，不丢弃
                logger.warning(f"[node_generate] 引用重试失败，保留首次结果: {e}")
    except Exception as e:
        logger.error(f"[node_generate] request_id={state.get('request_id')} 失败: {e}")
        return {
            "answer": "",
            "citations": [],
            "error": str(e),
            "error_code": "INTERNAL",
            "error_stage": "generate",
        }

    # citation 事件按首现序推送（04j §3.5）；不发 done
    if writer is not None:
        for c in citations:
            writer(
                {
                    "type": "citation",
                    "ref": c["ref"],
                    "document_id": c.get("document_id"),
                    "stock_code": c.get("stock_code"),
                    "page_no": c.get("page_no"),
                }
            )

    # Redis assistant 条：含 rewritten_query；失败仅 WARN 不阻断（04j §3.5）
    try:
        history = get_redis_chat_history(state.get("session_id") or "")
        history.add_ai_message(answer, state.get("rewritten_query") or "")
    except Exception as e:
        logger.warning(f"[node_generate] Redis 回写失败，忽略: {e}")

    logger.info(
        f"[node_generate] request_id={state.get('request_id')} "
        f"citations={len(citations)} answer_len={len(answer)}"
    )
    return {
        "answer": answer,
        "citations": citations,
        "error": None,
        "error_code": None,
        "error_stage": None,
    }


if __name__ == "__main__":
    # 真实 LLM 冒烟：get_llm_client() 走 Qwen 流式，假 evidence + mock Redis。
    # 覆盖：context 拼接、token 流式、[n] 抽取、空证据话术。
    import asyncio

    from unittest.mock import patch

    from app.lm.llm_utils import get_llm_client
    from app.query_graph.state import default_state

    def _ev(ref, type="narrative", score=0.04, content="正文段落", **over):
        r = {
            "ref": ref,
            "type": type,
            "content": content,
            "snippet": content[:120],
            "page_no": 10 + ref,
            "document_id": "doc1",
            "stock_code": "000008",
            "report_period": "2024",
            "source": "dense",
            "score": score,
            "sub_id": "sq1",
            "metric_code": None,
            "status": None,
        }
        r.update(over)
        return r

    class _Writer:
        def __init__(self):
            self.events = []

        def __call__(self, e):
            self.events.append(e)

    async def main():
        llm = get_llm_client()
        rt = Runtime(context=QueryAgentContext(llm=llm))

        print("== sq1 指标+叙述证据 → 真实 LLM 流式生成 ==")
        w = _Writer()
        with patch(f"{__name__}.get_redis_chat_history", side_effect=RuntimeError("no redis")):
            out = await node_generate(
                {
                    **default_state(
                        query="贵州茅台2024年销售毛利率是多少？同比变化如何？",
                        request_id="smoke",
                        session_id="smoke",
                    ),
                    "rewritten_query": "贵州茅台2024年销售毛利率是多少？同比变化如何？",
                    "intent": "metric_calc",
                    "entities": {"stock_codes": ["600519"], "periods": ["2024"]},
                    "evidence": [
                        _ev(12, type="metric", metric_code="gross_margin",
                            content="贵州茅台 2024 年销售毛利率：34.2%", score=0.0, stock_code="600519"),
                        _ev(13, type="calc", metric_code="gross_margin_yoy",
                            content="贵州茅台 2024 年销售毛利率同比：+1.1pct", score=0.0, status="ok", stock_code="600519"),
                        _ev(3, content="公司2024年实现营业收入1,241亿元，同比增长12.4%；归母净利润627亿元。",
                            stock_code="600519", page_no=45),
                    ],
                },
                rt,
                w,
            )
        tokens = [e["content"] for e in w.events if e["type"] == "token"]
        print(f"  token 数: {len(tokens)}")
        print(f"  token 流（前 3）: {tokens[:3]}")
        print(f"  answer:\n{out['answer']}")
        print(f"  citations: {[(c['ref'], c['type'], c['page_no']) for c in out['citations']]}")
        cites = [e["ref"] for e in w.events if e["type"] == "citation"]
        print(f"  citation SSE: {cites}")

        print("\n== sq2 空证据话术（不调 LLM）==")
        out2 = await node_generate(
            {**default_state(query="随便问点什么", request_id="smoke2"), "evidence": []},
            rt,
            None,
        )
        print(f"  answer={out2['answer']!r}")
        print(f"  citations={out2['citations']}")

    asyncio.run(main())
