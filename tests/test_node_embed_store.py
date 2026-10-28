"""node_embed_store 测试：编码入库、payload 原文、幂等、失败计数（docs/02c §4.3）。"""
import asyncio
import uuid

import pytest
from langgraph.runtime import Runtime
from qdrant_client import QdrantClient, models

from app.clients.qdrant_client import QdrantClientManager
from app.conf.qdrant_config import QdrantConfig
from app.import_graph.context import DataAgentContext
from app.import_graph.nodes.embed_store import node_embed_store


@pytest.fixture()
def qdrant_manager():
    m = QdrantClientManager()
    m.client = QdrantClient(":memory:")
    m.config = QdrantConfig(host="localhost", port=6333, embedding_size=4, collection="test_chunks")
    m.ensure_collection()
    yield m
    m.close()
    QdrantClientManager._instance = None


@pytest.fixture()
def ctx_runtime(qdrant_manager, runtime: Runtime[DataAgentContext]):
    runtime.context.qdrant = qdrant_manager
    return runtime


def _state(chunks):
    return {
        "document_id": "doc1",
        "stock_name": "神州高铁",
        "stock_code": "000008",
        "report_type": "annual",
        "report_period": "2024",
        "chunks": chunks,
        "failed_chunks": 0,
    }


def _chunk(i: int, content: str, section: str = "management_discussion") -> dict:
    return {
        "chunk_id": f"doc1:{i:04d}",
        "document_id": "doc1",
        "chunk_type": "narrative",
        "content": content,
        "section": section,
        "page_no": 10 + i,
        "title": f"标题{i}",
    }


def test_embed_store_requires_injected_qdrant(runtime):
    """qdrant 必须从 DataAgentContext 注入，禁止回落全局单例（DI 规约）。"""
    with pytest.raises(RuntimeError, match="qdrant"):
        asyncio.run(node_embed_store(_state([_chunk(0, "正文")]), runtime))


def test_embed_store_writes_points_with_raw_content(ctx_runtime, monkeypatch):
    captured = []

    def fake_embed(texts):
        captured.extend(texts)
        return {
            "dense": [[0.1, 0.2, 0.3, 0.4]] * len(texts),
            "sparse": [{0: 1.0}] * len(texts),
        }

    monkeypatch.setattr("app.import_graph.nodes.embed_store.generate_embeddings", fake_embed)

    state = _state([
        _chunk(0, "本集团2024年度实现营业收入100元"),
        _chunk(1, "公司可能面临的风险如下", section="risk"),
    ])
    patch = asyncio.run(node_embed_store(state, ctx_runtime))

    assert patch["qdrant_ok"] is True
    assert patch["embedded_chunks"] == 2
    assert patch["failed_chunks"] == 0

    # 编码串带主体与段落类型，payload.content 仍是原文
    assert captured[0] == "神州高铁·管理层讨论：本集团2024年度实现营业收入100元"
    assert captured[1] == "神州高铁·风险因素：公司可能面临的风险如下"

    points = ctx_runtime.context.qdrant.client.scroll("test_chunks", with_payload=True)[0]
    assert len(points) == 2
    by_title = {p.payload["title"]: p for p in points}
    raw = by_title["标题0"]
    assert raw.payload["content"] == "本集团2024年度实现营业收入100元"
    assert "神州高铁·" not in raw.payload["content"]
    assert raw.payload["document_id"] == "doc1"
    assert raw.payload["stock_code"] == "000008"
    assert raw.payload["report_period"] == "2024"
    assert raw.payload["report_type"] == "annual"
    assert raw.payload["chunk_type"] == "narrative"
    assert raw.payload["section"] == "management_discussion"
    assert raw.payload["page_no"] == 10
    # point id 由 chunk_id 稳定派生
    assert raw.id == str(uuid.uuid5(uuid.NAMESPACE_URL, "doc1:0000"))


def test_embed_store_skips_empty_content(ctx_runtime, monkeypatch):
    monkeypatch.setattr(
        "app.import_graph.nodes.embed_store.generate_embeddings",
        lambda texts: {"dense": [[0.1] * 4] * len(texts), "sparse": [{}] * len(texts)},
    )
    state = _state([_chunk(0, "有效内容"), _chunk(1, "")])
    patch = asyncio.run(node_embed_store(state, ctx_runtime))
    assert patch["embedded_chunks"] == 1
    assert patch["failed_chunks"] == 1
    points = ctx_runtime.context.qdrant.client.scroll("test_chunks")[0]
    assert len(points) == 1


def test_embed_store_retries_batch_once_then_counts_failed(ctx_runtime, monkeypatch):
    calls = {"n": 0}

    def flaky(texts):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("encode boom")
        return {"dense": [[0.1] * 4] * len(texts), "sparse": [{}] * len(texts)}

    monkeypatch.setattr("app.import_graph.nodes.embed_store.generate_embeddings", flaky)
    state = _state([_chunk(0, "内容A")])
    patch = asyncio.run(node_embed_store(state, ctx_runtime))
    assert calls["n"] == 2
    assert patch["embedded_chunks"] == 1
    assert patch["failed_chunks"] == 0


def test_embed_store_gives_up_after_retry(ctx_runtime, monkeypatch):
    calls = {"n": 0}

    def always_fail(texts):
        calls["n"] += 1
        raise RuntimeError("encode boom")

    monkeypatch.setattr("app.import_graph.nodes.embed_store.generate_embeddings", always_fail)
    state = _state([_chunk(0, "内容A"), _chunk(1, "内容B")])
    patch = asyncio.run(node_embed_store(state, ctx_runtime))
    assert calls["n"] == 2
    assert patch["embedded_chunks"] == 0
    assert patch["failed_chunks"] == 2
    assert patch["qdrant_ok"] is True
    assert ctx_runtime.context.qdrant.client.scroll("test_chunks")[0] == []


def test_embed_store_rerun_replaces_not_duplicates(ctx_runtime, monkeypatch):
    monkeypatch.setattr(
        "app.import_graph.nodes.embed_store.generate_embeddings",
        lambda texts: {"dense": [[0.1] * 4] * len(texts), "sparse": [{}] * len(texts)},
    )
    state = _state([_chunk(0, "内容A"), _chunk(1, "内容B")])
    asyncio.run(node_embed_store(state, ctx_runtime))
    # 重导只剩 1 条 chunk：旧 2 条必须被清掉
    state["chunks"] = [_chunk(0, "内容A改")]
    asyncio.run(node_embed_store(state, ctx_runtime))
    points = ctx_runtime.context.qdrant.client.scroll("test_chunks", with_payload=True)[0]
    assert len(points) == 1
    assert points[0].payload["content"] == "内容A改"


def test_embed_store_resolve_pronoun_only_on_encode_text(ctx_runtime, monkeypatch):
    captured = []

    def fake_embed(texts):
        captured.extend(texts)
        return {"dense": [[0.1] * 4] * len(texts), "sparse": [{}] * len(texts)}

    monkeypatch.setattr("app.import_graph.nodes.embed_store.generate_embeddings", fake_embed)
    monkeypatch.setattr("app.conf.embedding_config.embedding_config.resolve_pronoun", True)

    state = _state([_chunk(0, "本集团实现营收，子公司不变")])
    asyncio.run(node_embed_store(state, ctx_runtime))
    assert captured[0] == "神州高铁·管理层讨论：神州高铁集团实现营收，子公司不变"
    points = ctx_runtime.context.qdrant.client.scroll("test_chunks", with_payload=True)[0]
    assert points[0].payload["content"] == "本集团实现营收，子公司不变"
