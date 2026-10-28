"""S5a 向量化入库（docs/02c §4.3）。

为什么编码前缀而不是 LLM 指代消解：指代词模板化，改写有幻觉风险（docs/02c §2）。
为什么 payload 存原文：检索命中后要原样引用/溯源（docs/02c §1.2）。
为什么先删后写：重导时 chunk 可能变少，只 upsert 会留残点（docs/02c §4.3）。
"""
from __future__ import annotations

import asyncio
import uuid

from langgraph.runtime import Runtime
from langgraph.types import StreamWriter
from qdrant_client import models

from app.clients.qdrant_client import QdrantClientManager
from app.conf.embedding_config import embedding_config
from app.core.logger import logger
from app.import_graph.context import DataAgentContext
from app.import_graph.encode import build_encode_text
from app.import_graph.nodes import node_metadata, node_sectionize, node_split
from app.import_graph.nodes.progress import progress_step
from app.import_graph.state import ImportState
from app.lm.embedding_utils import generate_embeddings
from app.utils.file import sha1_file
from app.utils.path_util import PROJECT_ROOT

# docs/02 §S5a：batch=8，平衡单次编码吞吐与失败爆炸半径
BATCH_SIZE = 8


def _point_id(chunk_id: str) -> str:
    """Qdrant point id 只接受 UUID/整数；chunk_id 是 document_id:seq 字符串。

    用 uuid5 而不是 uuid4：同 chunk_id 重跑必须得到同一 id，否则先删后写后
    检索侧缓存/引用会对不上。
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _to_sparse(sparse_dict: dict) -> models.SparseVector | None:
    if not sparse_dict:
        return None
    indices = sorted(sparse_dict.keys())
    return models.SparseVector(indices=indices, values=[sparse_dict[i] for i in indices])


def _get_qdrant(runtime: Runtime[DataAgentContext]) -> QdrantClientManager:
    """只从 DataAgentContext 取 qdrant，禁止回落全局单例（DI 规约）。

    为什么强制注入：单例兜底会让单测/多连接场景静默打到真实库，且与
    api/dependencies 的注入路径不一致。
    """
    qdrant = runtime.context.qdrant
    if qdrant is None:
        raise RuntimeError("qdrant 未注入 DataAgentContext，节点禁止自建客户端")
    return qdrant


def _build_point(chunk: dict, meta: dict, dense: list, sparse: dict) -> models.PointStruct:
    """payload.content 必须是原文；编码串只进 embedding（docs/02c §1.2/§5.1）。"""
    vector: dict = {"dense": dense}
    sv = _to_sparse(sparse)
    if sv is not None:
        vector["sparse"] = sv
    return models.PointStruct(
        id=_point_id(chunk["chunk_id"]),
        vector=vector,
        payload={
            "document_id": meta["document_id"],
            "stock_code": meta["stock_code"],
            "report_period": meta["report_period"],
            "report_type": meta["report_type"],
            "chunk_type": chunk.get("chunk_type", "narrative"),
            "section": chunk.get("section", "md_narrative"),
            "page_no": chunk.get("page_no", 0),
            "title": chunk.get("title", ""),
            "content": chunk.get("content", ""),
            "chunk_id": chunk["chunk_id"],
        },
    )


async def node_embed_store(
    state: ImportState,
    runtime: Runtime[DataAgentContext],
    writer: StreamWriter | None = None,
) -> dict:
    """S5a: BGE-M3 向量化并写入 Qdrant（分支 A）。

    失败不污染 State：单 batch 重试 1 次后跳过并计 failed_chunks，节点不整体失败
    （docs/02c §4.3/§6）。
    """
    logger.info(f"[node_embed_store] 开始处理文档: {state.get('document_id')}")
    with progress_step(writer, "向量入库") as outcome:
        chunks = state.get("chunks") or []
        meta = {
            "document_id": state.get("document_id") or "",
            "stock_code": state.get("stock_code") or "",
            "report_period": state.get("report_period") or "",
            "report_type": state.get("report_type") or "",
        }
        stock_name = state.get("stock_name") or ""
        stock_code = meta["stock_code"]
        resolve = embedding_config.resolve_pronoun
        prev_failed = state.get("failed_chunks") or 0

        qdrant = _get_qdrant(runtime)
        try:
            qdrant.ensure_collection()
            # 先删后写：重导幂等，chunk 变少也不留残点
            qdrant.delete_by_document(meta["document_id"])
        except Exception as e:
            logger.error(f"[node_embed_store] collection 准备失败: {e}", exc_info=True)
            outcome["ok"] = False
            return {"embedded_chunks": 0, "failed_chunks": prev_failed, "qdrant_ok": False}

        # 空 content 直接跳过，不进 batch 也不重试（docs/02c §6）
        valid: list[dict] = []
        failed = 0
        for c in chunks:
            if not (c.get("content") or "").strip():
                failed += 1
                continue
            valid.append(c)

        embedded = 0
        for i in range(0, len(valid), BATCH_SIZE):
            batch = valid[i : i + BATCH_SIZE]
            points: list[models.PointStruct] | None = None
            for attempt in range(2):
                try:
                    encode_texts = [
                        build_encode_text(c, stock_name, stock_code, resolve=resolve) for c in batch
                    ]
                    vectors = generate_embeddings(encode_texts)
                    points = [
                        _build_point(c, meta, vectors["dense"][j], vectors["sparse"][j])
                        for j, c in enumerate(batch)
                    ]
                    break
                except Exception as e:
                    if attempt == 0:
                        logger.warning(f"[node_embed_store] batch 编码失败，重试: {e}")
                        continue
                    logger.error(f"[node_embed_store] batch 编码失败，跳过 {len(batch)} 条: {e}")
                    failed += len(batch)

            if points is not None:
                try:
                    qdrant.client.upsert(collection_name=qdrant.config.collection, points=points)
                    embedded += len(points)
                except Exception as e:
                    logger.error(f"[node_embed_store] upsert 失败，跳过 {len(points)} 条: {e}", exc_info=True)
                    failed += len(points)

        return {
            "embedded_chunks": embedded,
            "failed_chunks": prev_failed + failed,
            "qdrant_ok": True,
        }


if __name__ == "__main__":
    # 真实入库冒烟：真实 parse 产物 → split → sectionize → metadata → embed_store → 真实 Qdrant。
    # 写 report_chunks 集合（非 :memory:）；节点按 document_id 先删后写，重跑幂等。
    # 只跑前 3 条 chunk：冒烟看链路与 payload，不做全量压测。
    pdf_path = PROJECT_ROOT / "report" / "000008_2024_神州高铁_2024年年度报告_2025-04-19.pdf"
    document_id = sha1_file(pdf_path)
    path_root = PROJECT_ROOT / "data" / "parse" / "013d1633320c62a34fd359a6fa5bb27f1e765355"
    state: ImportState = {
        "pdf_path": str(pdf_path),
        "markdown_path": str(path_root / "content.md"),
        "pages_map_path": str(path_root / "pages.json"),
        "document_id": document_id,
    }  # type: ignore[typeddict-item]

    state.update(node_split(state, None, None))  # type: ignore[arg-type]
    state.update(node_metadata(state, None, None))  # type: ignore[arg-type]
    state.update(node_sectionize(state, None, None))  # type: ignore[arg-type]
    state["chunks"] = (state.get("chunks") or [])[:3]

    # 真实 Qdrant：init 读 QDRANT_HOST/PORT，经 DataAgentContext 注入节点
    mgr = QdrantClientManager()
    mgr.init()
    print(f"== Qdrant {mgr.build_url(mgr.config)} collection={mgr.config.collection} ==")

    stock_name = state.get("stock_name") or ""
    stock_code = state.get("stock_code") or ""
    print("== 编码串样例（只进 embedding，不进 payload）==")
    for c in state["chunks"]:
        print(" ", build_encode_text(c, stock_name, stock_code, resolve=False)[:80])

    runtime = Runtime(context=DataAgentContext(qdrant=mgr))
    patch = asyncio.run(node_embed_store(state, runtime))  # type: ignore[arg-type]
    print("== state patch ==")
    print({k: patch[k] for k in ("embedded_chunks", "failed_chunks", "qdrant_ok")})
    assert patch["qdrant_ok"] is True and patch["embedded_chunks"] > 0

    n = mgr.count_by_document(document_id)
    print(f"== 真实库 count_by_document == {n}")
    assert n == patch["embedded_chunks"], "入库点数须与 embedded_chunks 一致"

    points = mgr.client.scroll(mgr.config.collection, with_payload=True, limit=3)[0]
    print("== payload 抽查（content 必须是原文）==")
    for p in points:
        content = p.payload.get("content", "")
        print(f"  id={str(p.id)[:8]}… section={p.payload.get('section')} content={content[:40]!r}")
        assert "神州高铁·" not in content, "payload.content 不得带编码前缀"