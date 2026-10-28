from __future__ import annotations

from qdrant_client import QdrantClient, models

from app.conf.qdrant_config import QdrantConfig, qdrant_config


class QdrantClientManager:
    """Qdrant 连接单例：全局共享 client，禁止各处散建连接。

    为什么单例：与 MySQLClientManager 同理——import_graph 多节点并发读写
    同一 collection，连接必须共享。用同步 QdrantClient 与 mysql_client 一致；
    embed_store 虽是 async 节点，但 Qdrant 本地/远程 IO 短，不值得双套接口。
    """

    _instance: QdrantClientManager | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.client = None
            cls._instance.config = qdrant_config
        return cls._instance

    @staticmethod
    def build_url(cfg: QdrantConfig) -> str:
        return f"http://{cfg.host}:{cfg.port}"

    def init(self, cfg: QdrantConfig | None = None) -> None:
        self.config = cfg or self.config or qdrant_config
        self.client = QdrantClient(url=self.build_url(self.config))

    def _ensure_client(self) -> QdrantClient:
        if self.client is None:
            self.init()
        return self.client

    def ensure_collection(self, recreate: bool = False) -> None:
        """建 report_chunks：dense(1024,Cosine) + sparse（docs/01 §6.1）。

        recreate=True 先删后建，供全量重导；默认幂等不重复建。
        """
        client = self._ensure_client()
        name = self.config.collection
        if recreate and client.collection_exists(name):
            client.delete_collection(name)
        if client.collection_exists(name):
            return
        client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.config.embedding_size,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(),
            },
        )

    def delete_by_document(self, document_id: str) -> None:
        """按 document_id 先删后写，保证重导幂等（docs/02 §S5a）。"""
        client = self._ensure_client()
        client.delete(
            collection_name=self.config.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    def count_by_document(self, document_id: str) -> int:
        """按 document_id 计点——S7 单侧重跑时 State 向量为空，靠库兜底（docs/02f §6.2）。

        filter 与 delete_by_document 保持一致，避免「删得掉但数不到」的口径分裂。
        collection 不存在 = 尚无向量，返回 0 而不是抛错，否则 S7 会误判 Qdrant 不可达。
        """
        client = self._ensure_client()
        if not client.collection_exists(self.config.collection):
            return 0
        result = client.count(
            collection_name=self.config.collection,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
        )
        return int(result.count)

    def _build_filter(self, query_filter: dict | None) -> models.Filter | None:
        """dict[str, list[str]] → Qdrant Filter；空/None 不加条件（04e §5.1）。"""
        if not query_filter:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(key=key, match=models.MatchAny(any=list(values)))
                for key, values in query_filter.items()
                if values
            ]
        )

    @staticmethod
    def _point_to_hit(point) -> dict:
        hit = dict(point.payload or {})
        hit["score"] = float(point.score) if point.score is not None else 0.0
        return hit

    def search_dense(self, vector: list[float], query_filter: dict | None, limit: int) -> list[dict]:
        """dense 检索：filter 先过滤后向量（04e §3.3）；collection 不存在返回空（单路空合法）。"""
        client = self._ensure_client()
        if not client.collection_exists(self.config.collection):
            return []
        result = client.query_points(
            collection_name=self.config.collection,
            query=vector,
            using="dense",  # collection 是命名向量（dense/sparse），不指定会报 not found
            query_filter=self._build_filter(query_filter),
            limit=limit,
            with_payload=True,
        )
        return [self._point_to_hit(p) for p in result.points]

    def search_sparse(self, sparse: dict[int, float], query_filter: dict | None, limit: int) -> list[dict]:
        """sparse 检索：同 dense 口径，MatchAny filter + payload+score。"""
        client = self._ensure_client()
        if not client.collection_exists(self.config.collection):
            return []
        indices = sorted(sparse)
        result = client.query_points(
            collection_name=self.config.collection,
            query=models.SparseVector(
                indices=indices,
                values=[sparse[i] for i in indices],
            ),
            using="sparse",
            query_filter=self._build_filter(query_filter),
            limit=limit,
            with_payload=True,
        )
        return [self._point_to_hit(p) for p in result.points]

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
