"""向量生成：DashScope text-embedding-v4 远程调用。

为什么不用本地 BGE-M3：4 核 Intel Mac 无 GPU，单条 ~2.2s，年报 500+ chunk 要 20+ 分钟；
远程 API 并发后吞吐 100×+。为什么 output_type="dense&sparse"：旧实现「字符下标→1.0」
是假稀疏，lexical 匹配完全失效；API 的 sparse_embedding 才是真实词项权重。
"""
from __future__ import annotations

from dashscope import TextEmbedding

from app.conf.embedding_config import embedding_config
from app.core.logger import logger

# DashScope text-embedding 单次 input 上限 10 条，超限直接 400
API_MAX_BATCH = 10


def _parse_sparse(sparse_embedding: list | None) -> dict[int, float]:
    """sparse_embedding=[{index,token,value}] → {index: value}，对齐 Qdrant SparseVector。"""
    if not sparse_embedding:
        return {}
    return {int(item["index"]): float(item["value"]) for item in sparse_embedding}


def generate_embeddings(texts: list[str]) -> dict:
    """为文本列表生成稠密+稀疏混合向量嵌入。

    :param texts: 要生成嵌入的文本列表，单文本也需封装为列表
    :return: {"dense": [[float]...], "sparse": [{int: float}...]}
    :raise ValueError: 入参非非空列表
    :raise Exception: API 失败，由调用方做重试/降级处理
    """
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("生成向量入参不合法，texts必须为非空列表")
        raise ValueError("参数texts必须是包含文本的非空列表")
    if not embedding_config.api_key:
        raise ValueError("配置缺失：请在.env中配置 EMBEDDING_API_KEY 或 LLM_API_KEY")

    logger.info(f"开始为{len(texts)}条文本生成混合向量嵌入")
    dense_all: list[list[float]] = []
    sparse_all: list[dict[int, float]] = []

    for start in range(0, len(texts), API_MAX_BATCH):
        batch = texts[start : start + API_MAX_BATCH]
        resp = TextEmbedding.call(
            model=embedding_config.model,
            input=batch,
            dimension=embedding_config.dimension,
            output_type="dense&sparse",  # 混合向量
            api_key=embedding_config.api_key,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"embedding API 失败 status={resp.status_code} "
                f"code={getattr(resp, 'code', None)} msg={getattr(resp, 'message', None)}"
            )
        # text_index 是批内下标，排序后才与入参顺序一致
        embs = sorted(resp.output["embeddings"], key=lambda e: e["text_index"])
        if len(embs) != len(batch):
            raise RuntimeError(f"embedding 返回条数不符：got={len(embs)} want={len(batch)}")
        for e in embs:
            dense_all.append([float(x) for x in e["embedding"]])
            sparse_all.append(_parse_sparse(e.get("sparse_embedding")))

    logger.success(f"{len(texts)}条文本向量生成完成")
    return {"dense": dense_all, "sparse": sparse_all}
