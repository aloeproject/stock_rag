"""断崖 TopK 截断（docs/04i §3.3）。

gap_abs=0.5 或 rel≥0.25 触发截断；动态 topK max 10 / min 1。
为什么 rel=gap/|s1| 而不是 s2/s1：对齐 dataset_rag 的落差语义；s1 为负时比值语义崩溃。
为什么先 cap 再找断崖：max=10 是硬上限，第 10 位之后的断崖不该影响结果。
为什么 min 从 min_topk-1 起扫：min 是结果下限不是扫描下限——第 1 名即断崖也保住 1 条。
"""
from __future__ import annotations

RERANK_MAX_TOPK = 10
RERANK_MIN_TOPK = 1
RERANK_GAP_ABS = 0.5
RERANK_GAP_RATIO = 0.25


def cliff_topk(scored: list[dict]) -> list[dict]:
    """按 score 降序（平分按 ref 升序）截断到 1~10 条。

    自带排序，入参乱序也能正确找断崖；必须 return（缺 return 下游拿空）。
    """
    if not scored:
        return []
    ordered = sorted(scored, key=lambda x: (-x.get("score", 0.0), x.get("ref", 0)))
    topk = min(RERANK_MAX_TOPK, len(ordered))
    if topk > RERANK_MIN_TOPK:
        for i in range(RERANK_MIN_TOPK - 1, topk - 1):
            s1 = ordered[i].get("score", 0.0)
            s2 = ordered[i + 1].get("score", 0.0)
            gap = s1 - s2
            # 分母不加 1e-6：0.1/0.4 要精确命中 0.25，容差 fudge 会把阈值挤偏
            rel = gap / abs(s1) if s1 != 0.0 else (float("inf") if gap > 0 else 0.0)
            if gap >= RERANK_GAP_ABS or rel >= RERANK_GAP_RATIO:
                topk = i + 1
                break
    return ordered[:topk]
