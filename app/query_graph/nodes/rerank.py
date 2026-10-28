"""Q9 node_rerank（docs/04i）。

bge-reranker-large 成对打分 + 断崖 TopK。
为什么 fuse 之后还独立成节点：RRF 是无监督共识，rerank 是有监督相关性，
二目不同——fuse 保多路信号不丢，rerank 保最终上下文最相关（04i §1.2）。
为什么只重排向量类：metric/calc 无正文可成对打分，数字不该被语义相似度挤走（04i §4.2）。
为什么 ref 保留不重编：04h §5.2 契约；截断后缺口由 generate/validate 消化（04i §3.4）。
为什么不发 SSE：候选已由 fuse 推送，rerank 只影响 generate 上下文（04i §3.5）。
"""
from __future__ import annotations

from langgraph.runtime import Runtime
from langgraph.types import StreamWriter

from app.core.logger import logger
from app.lm.reranker_utils import rerank_pairs
from app.query_graph.context import QueryAgentContext
from app.query_graph.matching.cliff_topk import (
    RERANK_MAX_TOPK,
    RERANK_MIN_TOPK,
    cliff_topk,
)
from app.query_graph.state import QueryState

__all__ = ["RERANK_MAX_TOPK", "RERANK_MIN_TOPK", "node_rerank", "rerank_pairs"]

_VECTOR_TYPES = ("narrative", "table")


def step_1_split(evidence: list[dict]) -> tuple[list[dict], list[dict]]:
    """向量类可成对打分；结构化类位次不动拼回末尾（04i §3.1/§4.2）。"""
    vector = [e for e in evidence if e.get("type") in _VECTOR_TYPES]
    struct = [e for e in evidence if e.get("type") not in _VECTOR_TYPES]
    return vector, struct


def step_2_score(query: str, vector: list[dict]) -> None:
    """(query, content) 成对打分写回 score；空 content 不进模型给 0 分。

    问句优先 rewritten_query（指代已消解）；顺序问前答后，颠倒分数失真（04i §3.2）。
    """
    pending: list[tuple[dict, str]] = []
    for item in vector:
        content = item.get("content") or ""
        if content:
            pending.append((item, content))
        else:
            item["score"] = 0.0
            item["ranked"] = False
            logger.warning(f"[node_rerank] 跳过空 content ref={item.get('ref')}")
    if not pending:
        return
    scores = rerank_pairs(query, [c for _, c in pending])
    for (item, _), score in zip(pending, scores):
        item["score"] = score
        item["ranked"] = True


def step_3_degrade_rrf(vector: list[dict]) -> list[dict]:
    """降级：RRF 原序取前 MAX_TOPK，不跑断崖。

    断崖阈值作用在 sigmoid 分上；RRF 分量级 ~0.01，套用会永不截断或随机误杀（04i §8）。
    """
    out = vector[:RERANK_MAX_TOPK]
    for item in out:
        item["ranked"] = False
    return out


async def node_rerank(
    state: QueryState,
    runtime: Runtime[QueryAgentContext],
    writer: StreamWriter = None,
) -> dict:
    try:
        evidence = state.get("evidence") or []
        if not evidence:
            return {"evidence": [], "error": None, "error_code": None, "error_stage": None}

        vector, struct = step_1_split(evidence)

        # 纯指标题：向量类空，不调模型，结构化原样返回（04i §8）
        if not vector:
            for item in struct:
                item["ranked"] = False
            return {"evidence": struct, "error": None, "error_code": None, "error_stage": None}

        query = state.get("rewritten_query") or state.get("query") or ""
        try:
            step_2_score(query, vector)
        except Exception as ex:
            logger.warning(f"[node_rerank] 打分失败降级 RRF 原序: {ex}")
            vector = step_3_degrade_rrf(vector)
            return {
                "evidence": vector + struct,
                "error": None,
                "error_code": None,
                "error_stage": None,
            }

        vector = cliff_topk(vector)
        for item in struct:
            item["ranked"] = False
        merged = vector + struct

        logger.info(
            f"[node_rerank] request_id={state.get('request_id')} "
            f"vector={len(vector)} struct={len(struct)}"
        )
        return {"evidence": merged, "error": None, "error_code": None, "error_stage": None}
    except Exception as e:
        logger.error(f"[node_rerank] 内部异常: {e}")
        return {
            "evidence": [],
            "error": str(e),
            "error_code": "INTERNAL",
            "error_stage": "rerank",
        }


if __name__ == "__main__":
    # 纯函数冒烟：不加载真模型，mock rerank_pairs 走完整节点。
    # 覆盖：重排、断崖截断、metric/calc 不动、降级原序、纯指标题跳过。
    import asyncio

    from unittest.mock import patch

    from app.query_graph.state import default_state

    def _ev(ref, type="narrative", score=0.04, content="正文", **over):
        r = {
            "ref": ref,
            "type": type,
            "content": content,
            "snippet": content[:120],
            "page_no": 10,
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

    async def main():
        rt = Runtime(context=QueryAgentContext())

        print("== sq1 重排 + 断崖 ==")
        with patch(
            f"{__name__}.rerank_pairs",
            side_effect=lambda q, ps: [0.93, 0.91, 0.90, 0.39, 0.38][: len(ps)],
        ):
            patch_out = await node_rerank(
                {
                    **default_state(query="q", request_id="smoke"),
                    "evidence": [_ev(i + 1, content=f"V{i}") for i in range(5)]
                    + [_ev(99, type="metric", content="营收 2024：20亿", score=0.0)],
                },
                rt,
                None,
            )
        for e in patch_out["evidence"]:
            print(f"  ref={e['ref']} type={e['type']} score={e['score']:.3f} ranked={e.get('ranked')}")

        print("== sq2 降级原序 ==")
        with patch(
            f"{__name__}.rerank_pairs",
            side_effect=RuntimeError("model missing"),
        ):
            patch_out2 = await node_rerank(
                {
                    **default_state(query="q", request_id="smoke"),
                    "evidence": [_ev(i + 1, content=f"V{i}") for i in range(12)],
                },
                rt,
                None,
            )
        print(f"  n={len(patch_out2['evidence'])} refs={[e['ref'] for e in patch_out2['evidence']]}")
        print(f"  ranked={ {e.get('ranked') for e in patch_out2['evidence']} }")

        print("== sq3 纯指标题不调模型 ==")
        with patch(f"{__name__}.rerank_pairs") as m:
            patch_out3 = await node_rerank(
                {
                    **default_state(query="q", request_id="smoke"),
                    "evidence": [_ev(1, type="metric", content="毛利率 2024：34%", score=0.0)],
                },
                rt,
                None,
            )
            print(f"  called={m.called} n={len(patch_out3['evidence'])}")

        print("== sq4 空合法 ==")
        patch_out4 = await node_rerank(
            {**default_state(query="q", request_id="smoke"), "evidence": []},
            rt,
            None,
        )
        print(f"  evidence={patch_out4['evidence']} error={patch_out4['error']}")

    asyncio.run(main())
