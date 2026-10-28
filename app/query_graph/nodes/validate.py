"""Q11 node_validate（docs/04k）。

两检 + 宁拒勿猜：① 引用 page_no 在证据中存在 ② answer 数字与 evidence/metric 对账（容差 0.01）。
为什么 validate 在 generate 之后：对账需完整文本，token 一旦推出无法撤回（04k §1.1）。
为什么宁拒勿猜整条作废不截段放行：一个错数字比没有答案更危险（04k §1.2）。
为什么规则校验不用 LLM：精确比较可复现可单测，LLM 验证本身可能出错（04k §6）。
为什么 answer 含 CoT 全文对账：推理段幻觉数字也拦得住（04k §1.2）。
为什么容差 0.01 不放大：0.1 会放过 34.2% vs 34.3% 的实质偏差（04k §3.2）。
为什么不修 answer：改写引入新错 + 越权 + 不诚实（04k §5.3）。
为什么不逐个查「数字无 [n]」：年份/页码是合法无标记数字，整体粗粒度安全网（04k §3.3）。
"""
from __future__ import annotations

import re

from langgraph.runtime import Runtime
from langgraph.types import StreamWriter

from app.core.logger import logger
from app.query_graph.context import QueryAgentContext
from app.query_graph.state import QueryState

__all__ = ["TOLERANCE", "node_validate"]

TOLERANCE = 0.01

# 「数字 + 可选单位 + 紧邻 [n]」——04j §5.2 书写规范的对账镜像
# 单位最长优先：个百分点/亿元 放在 亿/元 之前，防短单位截断长单位
_NUM_BEFORE_REF = re.compile(
    r"([+-]?\d[\d,]*(?:\.\d+)?)\s*(?:个百分点|亿元|pct|%|亿|万|倍|元)?\[(\d+)\]"
)
_ANY_NUMBER = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")
_HAS_DIGIT = re.compile(r"\d")


def _extract_numbers(text: str) -> list[float]:
    """从文本中提取所有数字（去千分位逗号）。"""
    nums: list[float] = []
    for m in _ANY_NUMBER.finditer(text or ""):
        s = m.group().replace(",", "")
        try:
            nums.append(float(s))
        except ValueError:
            continue
    return nums


def step_1_check_citations(
    citations: list[dict], evidence: list[dict]
) -> str | None:
    """页码存在性 + 一致性。返回失败原因，None 表示通过。

    page_no 允许双侧同时为 None（metric/calc 可能无页码，04k §3.1）。
    页码不做「文档中存在」检查——真实性由导入侧保证，查询侧不二次溯源。
    """
    by_ref = {e.get("ref"): e for e in evidence}
    for c in citations or []:
        ref = c.get("ref")
        ev = by_ref.get(ref)
        if ev is None:
            return f"引用 [{ref}] 不在证据中"
        cp, ep = c.get("page_no"), ev.get("page_no")
        if cp is not None and ep is not None and cp != ep:
            return f"引用 [{ref}] 页码 {cp} 与证据页码 {ep} 不一致"
        if (cp is None) != (ep is None):
            return f"引用 [{ref}] 页码不一致（一侧为空）"
    return None


def _nums_of(ev: dict) -> list[float]:
    nums = _extract_numbers(ev.get("content") or "")
    mv = ev.get("metric_value")
    if mv is not None:
        try:
            nums.append(float(mv))
        except (TypeError, ValueError):
            pass
    return nums


def step_2_reconcile_numbers(
    answer: str, evidence: list[dict]
) -> str | None:
    """数字对账：answer「数+标」对 → evidence[n] content/metric_value 复核。

    对账覆盖 CoT 段（全文），推理段幻觉数字也拦（04k §1.2）。
    拒答时回查数字实际落在哪条证据——「数字对标号」串号比「数字凭空」常见，
    直接给出真 ref 能把排查从猜映射缩到看一眼（43,427,553.57[2] 实测）。
    """
    if not answer:
        return None
    by_ref = {e.get("ref"): e for e in evidence}
    for m in _NUM_BEFORE_REF.finditer(answer):
        raw_num, raw_ref = m.group(1), m.group(2)
        num = float(raw_num.replace(",", ""))
        ref = int(raw_ref)
        ev = by_ref.get(ref)
        if ev is None:
            return f"数字 {raw_num}[{ref}] 的引用不在证据中"
        ev_nums = _nums_of(ev)
        if not ev_nums:
            return f"证据 [{ref}] 中无可对账数字，答案写了 {raw_num}"
        if any(abs(num - e) <= TOLERANCE for e in ev_nums):
            continue
        hits = [
            str(e.get("ref"))
            for e in evidence
            if any(abs(num - x) <= TOLERANCE for x in _nums_of(e))
        ]
        where = f"（实际在 [{']['.join(hits)}]）" if hits else ""
        return f"数字 {raw_num} 与证据 [{ref}] 不一致{where}"
    return None


def step_3_check_orphans(answer: str, citations: list[dict]) -> str | None:
    """孤立数字：含数字却零引用 → 拒（04k §3.3）。

    不逐个查「数字无 [n]」：年份/页码是合法无标记数字。
    """
    if _HAS_DIGIT.search(answer or "") and not (citations or []):
        return "含数字但无引用"
    return None


def _fail_state(msg: str) -> dict:
    return {
        "error": msg,
        "error_code": "INSUFFICIENT_EVIDENCE",
        "error_stage": "validate",
    }


async def node_validate(
    state: QueryState,
    runtime: Runtime[QueryAgentContext],
    writer: StreamWriter = None,
) -> dict:
    answer = state.get("answer") or ""
    citations = state.get("citations") or []
    evidence = state.get("evidence") or []

    reason: str | None = None
    # evidence 空时跳过孤儿数字检查：回答是系统话术（含股票代码等合法数字），
    # 不是从证据编的，拦它等于把「已识别好当家（600467）」也拒掉
    checks = [
        lambda: step_1_check_citations(citations, evidence),
        lambda: step_2_reconcile_numbers(answer, evidence),
    ]
    if evidence:
        checks.append(lambda: step_3_check_orphans(answer, citations))
    for check in checks:
        reason = check()
        if reason is not None:
            break

    if reason is not None:
        logger.warning(f"[node_validate] 拒绝: {reason}")
        if writer is not None:
            writer(
                {
                    "type": "error",
                    "code": "INSUFFICIENT_EVIDENCE",
                    "message": reason,
                    "stage": "validate",
                }
            )
        return _fail_state(reason)

    # 全过 → done 终态；validate 是唯一发 done 的节点（04k §3.4）
    if writer is not None:
        writer({"type": "done", "request_id": state.get("request_id") or ""})
    logger.info(f"[node_validate] request_id={state.get('request_id')} 通过")
    return {"error": None, "error_code": None, "error_stage": None}


if __name__ == "__main__":
    # 纯函数冒烟：不依赖 LLM/DB，直接喂假 answer/citations/evidence。
    # 覆盖：数字一致过、数字不一致拒、页码漂移拒、空证据合法、孤立数字拒。
    import asyncio

    from app.query_graph.state import default_state

    def _ev(ref, content="毛利率 34.2%", page_no=11, **over):
        r = {
            "ref": ref,
            "type": "narrative",
            "content": content,
            "snippet": content[:120],
            "page_no": page_no,
            "document_id": "doc1",
            "stock_code": "000008",
            "report_period": "2024",
            "source": "dense",
            "score": 0.04,
            "sub_id": "sq1",
            "metric_code": None,
            "status": None,
            "metric_value": None,
        }
        r.update(over)
        return r

    class _Writer:
        def __init__(self):
            self.events = []

        def __call__(self, e):
            self.events.append(e)

    async def main():
        rt = Runtime(context=QueryAgentContext())

        def _state(answer, citations, evidence):
            s = default_state(query="q", request_id="smoke")
            s["answer"] = answer
            s["citations"] = citations
            s["evidence"] = evidence
            return s

        cite = lambda ref, page_no=11: {"ref": ref, "page_no": page_no, "type": "narrative"}  # noqa: E731

        print("== sq1 数字一致 → done ==")
        w = _Writer()
        out = await node_validate(
            _state("毛利率为 34.2%[1]", [cite(1)], [_ev(1)]),
            rt,
            w,
        )
        print(f"  error={out['error']} events={[e['type'] for e in w.events]}")

        print("== sq2 数字不一致 → 拒 ==")
        w2 = _Writer()
        out2 = await node_validate(
            _state("毛利率为 34.3%[1]", [cite(1)], [_ev(1)]),
            rt,
            w2,
        )
        print(f"  error={out2['error']} code={out2['error_code']} events={[e['type'] for e in w2.events]}")

        print("== sq3 页码漂移 → 拒 ==")
        out3 = await node_validate(
            _state("34.2%[1]", [cite(1, page_no=45)], [_ev(1, page_no=46)]),
            rt,
            None,
        )
        print(f"  error={out3['error']}")

        print("== sq4 空证据合法 → done ==")
        w4 = _Writer()
        out4 = await node_validate(
            _state("未找到相关年报证据，无法回答该问题。", [], []),
            rt,
            w4,
        )
        print(f"  error={out4['error']} events={[e['type'] for e in w4.events]}")

        print("== sq5 孤立数字 → 拒 ==")
        out5 = await node_validate(
            _state("毛利率为 34.2%", [], [_ev(1)]),
            rt,
            None,
        )
        print(f"  error={out5['error']}")

    asyncio.run(main())
