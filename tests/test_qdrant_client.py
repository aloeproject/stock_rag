"""qdrant_client 测试：URL、单例、ensure_collection、delete_by_document。

用 QdrantClient(':memory:') 跑真实 collection 操作，不 mock。
"""
import pytest
from qdrant_client import QdrantClient, models

from app.clients.qdrant_client import QdrantClientManager
from app.conf.qdrant_config import QdrantConfig


@pytest.fixture()
def manager():
    m = QdrantClientManager()
    m.client = QdrantClient(":memory:")
    m.config = QdrantConfig(host="localhost", port=6333, embedding_size=8, collection="test_chunks")
    yield m
    m.close()
    QdrantClientManager._instance = None


def test_build_url():
    cfg = QdrantConfig(host="127.0.0.1", port=6333, embedding_size=1024, collection="report_chunks")
    assert QdrantClientManager.build_url(cfg) == "http://127.0.0.1:6333"


def test_manager_is_singleton():
    m1 = QdrantClientManager()
    m2 = QdrantClientManager()
    assert m1 is m2


def test_ensure_collection_creates_dense_and_sparse(manager: QdrantClientManager):
    manager.ensure_collection()
    name = manager.config.collection
    assert manager.client.collection_exists(name)

    info = manager.client.get_collection(name)
    assert "dense" in info.config.params.vectors
    assert info.config.params.vectors["dense"].size == 8
    assert info.config.params.vectors["dense"].distance == models.Distance.COSINE
    assert "sparse" in info.config.params.sparse_vectors


def test_ensure_collection_idempotent(manager: QdrantClientManager):
    manager.ensure_collection()
    manager.ensure_collection()
    assert manager.client.collection_exists(manager.config.collection)


def test_delete_by_document(manager: QdrantClientManager):
    manager.ensure_collection()
    name = manager.config.collection
    manager.client.upsert(
        collection_name=name,
        points=[
            models.PointStruct(
                id=1,
                vector={"dense": [0.1] * 8, "sparse": models.SparseVector(indices=[0], values=[1.0])},
                payload={"document_id": "doc1"},
            ),
            models.PointStruct(
                id=2,
                vector={"dense": [0.2] * 8, "sparse": models.SparseVector(indices=[1], values=[1.0])},
                payload={"document_id": "doc2"},
            ),
        ],
    )

    manager.delete_by_document("doc1")

    remaining = manager.client.scroll(collection_name=name, with_payload=True)[0]
    assert len(remaining) == 1
    assert remaining[0].payload["document_id"] == "doc2"


def test_count_by_document(manager: QdrantClientManager):
    """S7 单侧重跑靠 count 兜底（docs/02f §6.2）；filter 须与 delete_by_document 一致。"""
    manager.ensure_collection()
    name = manager.config.collection
    manager.client.upsert(
        collection_name=name,
        points=[
            models.PointStruct(
                id=1,
                vector={"dense": [0.1] * 8, "sparse": models.SparseVector(indices=[0], values=[1.0])},
                payload={"document_id": "doc1"},
            ),
            models.PointStruct(
                id=2,
                vector={"dense": [0.2] * 8, "sparse": models.SparseVector(indices=[1], values=[1.0])},
                payload={"document_id": "doc1"},
            ),
            models.PointStruct(
                id=3,
                vector={"dense": [0.3] * 8, "sparse": models.SparseVector(indices=[2], values=[1.0])},
                payload={"document_id": "doc2"},
            ),
        ],
    )

    assert manager.count_by_document("doc1") == 2
    assert manager.count_by_document("doc2") == 1
    assert manager.count_by_document("missing") == 0


def test_count_by_document_missing_collection_is_zero(manager: QdrantClientManager):
    """collection 未建 = 尚无向量，S7 不能当 Qdrant 故障（docs/02f §7）。"""
    manager.config.collection = "not_created_collection"
    assert manager.count_by_document("doc1") == 0


def _seed(manager: QdrantClientManager):
    manager.ensure_collection()
    name = manager.config.collection
    manager.client.upsert(
        collection_name=name,
        points=[
            models.PointStruct(
                id=1,
                vector={"dense": [1.0] * 8, "sparse": models.SparseVector(indices=[0], values=[1.0])},
                payload={
                    "chunk_id": "c1",
                    "stock_code": "000008",
                    "report_period": "2024",
                    "section": "risk",
                    "content": "风险一",
                    "page_no": 1,
                },
            ),
            models.PointStruct(
                id=2,
                vector={"dense": [0.0] * 8, "sparse": models.SparseVector(indices=[1], values=[1.0])},
                payload={
                    "chunk_id": "c2",
                    "stock_code": "000007",
                    "report_period": "2024",
                    "section": "risk",
                    "content": "他司风险",
                    "page_no": 2,
                },
            ),
        ],
    )


def test_search_dense_respects_filter(manager: QdrantClientManager):
    """filter 先过滤后向量（docs/04e §3.3）：他司 chunk 不得混入。"""
    _seed(manager)
    hits = manager.search_dense([1.0] * 8, {"stock_code": ["000008"]}, limit=10)
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "c1"
    assert "score" in hits[0]


def test_search_sparse_returns_payload_plus_score(manager: QdrantClientManager):
    _seed(manager)
    hits = manager.search_sparse({0: 1.0}, {"stock_code": ["000008"]}, limit=10)
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "c1"
    assert hits[0]["score"] > 0


def test_search_dense_empty_collection_is_empty(manager: QdrantClientManager):
    """collection 不存在 = 空结果不是异常（单路空合法，docs/04 §5.5）。"""
    manager.config.collection = "not_created_collection"
    assert manager.search_dense([1.0] * 8, {}, limit=5) == []
