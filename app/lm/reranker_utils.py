"""bge-reranker-large 成对打分（docs/04i §5.3）。

为什么本地 FlagReranker：rerank 调用在查询热路径，远程延迟不可控；
不可用/超时由调用侧（node_rerank）降级向量原序，本模块只管抛错。
为什么 normalize=True（sigmoid）：gap_abs=0.5 在无界 logits 空间无意义（04i §4.3）。
为什么 (query, passage) 顺序不可颠倒：Cross-Encoder 对输入顺序敏感，颠倒分数失真（04i §3.2）。
为什么惰性 import FlagEmbedding：其顶层 import 拖入 datasets/pyarrow，
单测 mock rerank_pairs 不该为打分库的重依赖买单。
"""
from __future__ import annotations

from app.conf.reranker_config import reranker_config

_reranker_model = None


def get_reranker_model():
    """模块级缓存，禁止每次调用重新加载（热路径）。"""
    global _reranker_model
    if _reranker_model is None:
        from FlagEmbedding import FlagReranker

        _reranker_model = FlagReranker(
            model_name_or_path=reranker_config.bge_reranker_large,
            device=reranker_config.bge_reranker_device,
            use_fp16=reranker_config.bge_reranker_fp16,
        )
    return _reranker_model


def rerank_pairs(query: str, passages: list[str]) -> list[float]:
    """(query, passage) 成对打分，返回与 passages 等长的 sigmoid 分 (0,1)。

    失败抛异常不吞错，由 node_rerank 降级。
    """
    if not passages:
        return []
    pairs = [[query, p] for p in passages]
    scores = get_reranker_model().compute_score(pairs, normalize=True)
    if isinstance(scores, (int, float)):
        scores = [scores]
    scores = [float(s) for s in scores]
    if len(scores) != len(passages):
        raise RuntimeError(f"reranker 返回条数不符：got={len(scores)} want={len(passages)}")
    return scores
