"""Q1 node_intent（docs/04a）。

与 node_entity 共享一次 LLM 调用——prompt 上下文相同（query+history），
分两次调用纯浪费往返；图上仍拆两节点：intent 先判 out_of_scope 短路。

为什么意图只分类不回答：防数字幻觉，回答内容不进 intent；out_of_scope 例外，发固定模板拒答（§5.3）。
为什么实体只抄原词：stock_dict 是名称→代码真相源，LLM 猜码必错。
"""
from __future__ import annotations

import json

from langgraph.runtime import Runtime
from langgraph.types import StreamWriter

from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.query_graph.context import QueryAgentContext
from app.query_graph.state import QueryState

# compare | metric_calc | trend | simple_qa | out_of_scope（04a §5.1）
INTENT_ENUM = {"compare", "metric_calc", "trend", "simple_qa", "out_of_scope"}
HISTORY_TURNS = 6
LLM_MAX_RETRIES = 1
INTENT_CONFIDENCE_WARN = 0.6
DEFAULT_CONFIDENCE = 0.5

# 固定模板拒答（04a §5.3）：不调 LLM——越界句不该再花生成往返，措辞失控还可能顺手回答
OUT_OF_SCOPE_MESSAGE = (
    "我是股票年报问答助手，只能回答上市公司年报相关的问题"
    "（如财务指标、经营情况、风险因素等）。您可以换一个与年报相关的问题。"
)


def step_1_build_prompt(query: str, history: list[dict]) -> str:
    """拼 prompt；history 取近 N 条，空历史拼空串不报错。"""
    items = (history or [])[-HISTORY_TURNS:]
    lines = []
    for m in items:
        role = "用户" if m.get("role") == "user" else "AI"
        lines.append(f"{role}: {m.get('content') or ''}")
    return load_prompt(
        "entity_extract",
        query=query or "",
        history="\n".join(lines),
    )


def step_2_llm_json(llm, prompt: str) -> dict:
    """json_mode 调用 + 解析；超时/非 JSON/枚举越界都算失败，共重试 1 次。

    为什么 4xx 不单独跳过重试：LangChain 异常不带 HTTP 状态，
    重试 1 次成本可接受；真正请求不合法时第二次仍会立刻失败。
    """
    last_err: Exception | None = None
    for _attempt in range(LLM_MAX_RETRIES + 1):
        try:
            resp = llm.invoke(prompt)
            data = json.loads(resp.content)
            if not isinstance(data, dict):
                raise ValueError(f"LLM 输出不是对象: {type(data)}")
            if (data.get("intent") or "").strip() not in INTENT_ENUM:
                raise ValueError(f"intent 越出枚举: {data.get('intent')!r}")
            return data
        except Exception as e:
            last_err = e
            logger.warning(f"[node_intent] LLM 调用/解析失败: {e}")
    raise last_err if last_err else RuntimeError("LLM 调用失败")


def step_3_normalize(raw: dict, query: str) -> tuple[str, dict, float, str]:
    """宽松字段补全；intent 枚举已在 step_2 校验过。

    缺数组补 []、缺 rewritten 回退原 query：这些是模型偷懒不是失败，
    没必要把整次请求打死。
    """
    intent = (raw.get("intent") or "").strip()
    stock_names = list(raw.get("stock_names") or [])
    periods = list(raw.get("periods") or [])
    metrics = list(raw.get("metrics") or [])
    rewritten = (raw.get("rewritten_query") or "").strip() or (query or "")

    confidence = raw.get("confidence")
    if isinstance(confidence, (int, float)) and 0 <= float(confidence) <= 1:
        confidence = float(confidence)
    else:
        confidence = DEFAULT_CONFIDENCE

    normalized = {
        "intent": intent,
        "confidence": confidence,
        "stock_names": stock_names,
        "periods": periods,
        "metrics": metrics,
        "rewritten_query": rewritten,
    }
    return intent, normalized, confidence, rewritten


async def node_intent(
        state: QueryState,
        runtime: Runtime[QueryAgentContext],
        writer: StreamWriter = None,
) -> dict:
    llm = runtime.context.json_llm or runtime.context.llm
    if llm is None:
        logger.error("[node_intent] llm 未注入 QueryAgentContext")
        return {
            "error": "llm 未注入",
            "error_code": "INTERNAL",
            "error_stage": "intent",
        }

    query = state.get("query") or ""
    try:
        prompt = step_1_build_prompt(query, state.get("history") or [])
        raw = step_2_llm_json(llm, prompt)
        intent, normalized, confidence, rewritten = step_3_normalize(raw, query)
    except Exception as e:
        logger.error(f"[node_intent] request_id={state.get('request_id')} 失败: {e}")
        return {
            "error": str(e),
            "error_code": "INTERNAL",
            "error_stage": "intent",
        }

    if confidence < INTENT_CONFIDENCE_WARN:
        logger.warning(
            f"[node_intent] request_id={state.get('request_id')} confidence={confidence} 偏低"
        )

    if writer is not None:
        writer({"type": "intent", "intent": intent, "confidence": confidence})

    logger.info(
        f"[node_intent] request_id={state.get('request_id')} "
        f"intent={intent} confidence={confidence} names={normalized['stock_names']}"
    )

    if intent == "out_of_scope":
        # 只发专用终态事件不发 error：用户看到提示不是故障（对齐 clarify，04a §5.3）
        if writer is not None:
            writer({"type": "out_of_scope", "message": OUT_OF_SCOPE_MESSAGE})
        return {
            "intent": intent,
            "rewritten_query": rewritten,
            "error": OUT_OF_SCOPE_MESSAGE,
            "error_code": "OUT_OF_SCOPE",
            "error_stage": "intent",
        }

    return {
        "intent": intent,
        "intent_raw": normalized,
        "rewritten_query": rewritten,
        "error": None,
        "error_code": None,
        "error_stage": None,
    }


if __name__ == "__main__":
    # 手工冒烟：真 LLM 跑 3 条典型句（指代/反问/对比），打印改写结果
    # 全量回归跑 tests/test_node_intent_llm.py，这里只看单条链路通不通
    import asyncio

    from langgraph.runtime import Runtime

    from app.lm.llm_utils import get_llm_client
    from app.query_graph.state import default_state

    CASES = [
        {
            "query": "它的毛利率是多少",
            "history": [
                {"role": "user", "content": "贵州茅台2024年经营情况怎么样？"},
                {"role": "assistant", "content": "营收1741亿元，毛利率约91%。"},
            ],
        },
        {
            "query": "这不会是越亏越多吧？",
            "history": [
                {"role": "user", "content": "我看神州高铁2024年亏了5.45亿。"},
                {"role": "assistant", "content": "是的，归母净利润-5.45亿元，同比减亏34.16%。"},
            ],
        },
        {"query": "神州高铁和深振业A 2024年哪家营收更高，亏损更大？", "history": []},
    ]

    async def _main() -> None:
        runtime = Runtime(context=QueryAgentContext(llm=get_llm_client(json_mode=True)))
        for i, case in enumerate(CASES, 1):
            state = default_state(
                query=case["query"],
                request_id=f"smoke-{i}",
                session_id="smoke",
                history=case["history"],
            )
            patch = await node_intent(state, runtime, None)
            logger.info(f"[smoke {i}] {patch}")
            print(f"\n--- case {i} ---")
            print(f"Q   : {case['query']}")
            print(f"got : {patch.get('rewritten_query')}")
            print(f"meta: intent={patch.get('intent')} names={patch.get('intent_raw', {}).get('stock_names')}")

    asyncio.run(_main())
