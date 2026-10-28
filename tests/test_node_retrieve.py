"""node_retrieve 节点测试（docs/04e §8/§9）。

假 Qdrant + 假 LLM + monkeypatch embedding；断言三路 hits、HyDE 触发/降级、
fan-out、filter 映射、SSE done 形状。不打真 Qdrant/模型。
"""
from __future__ import annotations

import asyncio

from langgraph.runtime import Runtime

from app.query_graph.context import QueryAgentContext
from app.query_graph.nodes import retrieve as retrieve_mod
from app.query_graph.nodes.retrieve import node_retrieve
from app.query_graph.state import default_state


class FakeQdrant:
    """按 (dense/sparse) 队列吐结果；errors 优先抛出。"""

    def __init__(self, dense_hits=None, sparse_hits=None, dense_error=None, sparse_error=None):
        self.dense_hits = list(dense_hits or [])
        self.sparse_hits = list(sparse_hits or [])
        self.dense_error = dense_error
        self.sparse_error = sparse_error
        self.dense_calls: list[dict] = []
        self.sparse_calls: list[dict] = []

    def search_dense(self, vector, query_filter, limit):
        self.dense_calls.append({"vector": vector, "query_filter": query_filter, "limit": limit})
        if self.dense_error:
            raise self.dense_error
        return list(self.dense_hits)

    def search_sparse(self, sparse, query_filter, limit):
        self.sparse_calls.append({"sparse": sparse, "query_filter": query_filter, "limit": limit})
        if self.sparse_error:
            raise self.sparse_error
        return list(self.sparse_hits)


class FakeLLM:
    def __init__(self, payloads=None, errors=None):
        self.calls: list[str] = []
        self.payloads = list(payloads or [])
        self.errors = list(errors or [])

    def invoke(self, prompt):
        self.calls.append(prompt)
        if self.errors:
            raise self.errors.pop(0)
        payload = self.payloads.pop(0) if self.payloads else "假设文档"
        return type("R", (), {"content": payload})()


class _Writer:
    def __init__(self):
        self.events: list[dict] = []

    def __call__(self, item: dict) -> None:
        self.events.append(item)


def _hit(chunk_id, score=0.9, **over):
    h = {
        "chunk_id": chunk_id,
        "content": f"content-{chunk_id}",
        "page_no": 1,
        "section": "risk",
        "score": score,
        "title": "t",
        "document_id": "doc1",
        "stock_code": "000008",
        "report_period": "2024",
    }
    h.update(over)
    return h


def _embedding_stub(monkeypatch, **kw):
    calls: list[list[str]] = []

    def _gen(texts):
        calls.append(list(texts))
        return {
            "dense": [[0.1] * 4 for _ in texts],
            "sparse": [{1: 1.0} for _ in texts],
        }

    monkeypatch.setattr(retrieve_mod, "generate_embeddings", _gen)
    return calls


def _runtime(qdrant=None, llm=None) -> Runtime:
    ctx = QueryAgentContext(qdrant=qdrant, llm=llm)
    return Runtime(context=ctx)  # type: ignore[arg-type]


def _sq(sub_id="sq1", question="公司面临哪些风险？", route="vector", **over):
    f = {"stock_codes": ["000008"], "report_periods": ["2024"], "report_types": ["annual"], "sections": ["risk"]}
    f.update(over.pop("filters", {}))
    item = {"sub_id": sub_id, "question": question, "route": route, "filters": f, "hyde_used": False}
    item.update(over)
    return item


def _state(sub_queries=None, **kw):
    s = default_state(query="q", request_id="q1", session_id="s1")
    s["sub_queries"] = sub_queries if sub_queries is not None else [_sq()]
    s.update(kw)
    return s


# ---------- 成功路径 ----------


def test_success_writes_three_hit_lists(monkeypatch):
    _embedding_stub(monkeypatch)
    q = FakeQdrant(dense_hits=[_hit("d1")], sparse_hits=[_hit("s1", score=0.5)])
    llm = FakeLLM(["公司面临战略、市场与运营等方面的风险。"])
    events = _Writer()
    patch = asyncio.run(node_retrieve(_state(), _runtime(q, llm), events))

    assert patch["error"] is None
    assert patch["vector_hits"][0]["chunk_id"] == "d1"
    assert patch["vector_hits"][0]["source"] == "dense"
    assert patch["sparse_hits"][0]["chunk_id"] == "s1"
    assert patch["sparse_hits"][0]["source"] == "sparse"
    assert patch["hyde_hits"][0]["chunk_id"] == "d1"  # 假设文档复用 dense 结果
    assert patch["hyde_hits"][0]["source"] == "hyde"
    assert patch["sub_queries"][0]["hyde_used"] is True


def test_sse_done_event_with_hyde_used(monkeypatch):
    _embedding_stub(monkeypatch)
    q = FakeQdrant(dense_hits=[_hit("d1")])
    llm = FakeLLM(["假设答案"])
    events = _Writer()
    asyncio.run(node_retrieve(_state(), _runtime(q, llm), events))
    assert events.events == [
        {"type": "sub_query", "sub_id": "sq1", "status": "done", "hyde_used": True}
    ]


# ---------- 路由筛选 / fan-out / filter ----------


def test_skips_non_vector_routes(monkeypatch):
    _embedding_stub(monkeypatch)
    q = FakeQdrant(dense_hits=[_hit("d1")])
    llm = FakeLLM([])
    state = _state(sub_queries=[_sq(route="metric_sql"), _sq(sub_id="sq2", route="calc")])
    patch = asyncio.run(node_retrieve(state, _runtime(q, llm), None))
    assert patch["vector_hits"] == []
    assert patch["sparse_hits"] == []
    assert q.dense_calls == []


def test_fanout_multi_period_queries_each_unit(monkeypatch):
    """只测 fan-out：用含指标词+notes+高分压掉 HyDE，dense/sparse 调用数=期数。"""
    _embedding_stub(monkeypatch)
    q = FakeQdrant(dense_hits=[_hit("d1", score=0.9)])
    sq = _sq(
        question="毛利率下降的经营原因是什么情况呢",
        filters={
            "stock_codes": ["000008"],
            "report_periods": ["2023", "2024"],
            "sections": ["notes"],
        },
    )
    state = _state(sub_queries=[sq])
    asyncio.run(node_retrieve(state, _runtime(q, FakeLLM([])), None))
    # 2 期 × 1 子问题 = 2 次 dense + 2 次 sparse
    assert len(q.dense_calls) == 2
    assert len(q.sparse_calls) == 2
    periods = sorted(c["query_filter"]["report_period"][0] for c in q.dense_calls)
    assert periods == ["2023", "2024"]


def test_filter_maps_contract_keys_to_payload(monkeypatch):
    _embedding_stub(monkeypatch)
    q = FakeQdrant(dense_hits=[])
    sq = _sq(
        filters={
            "stock_codes": ["000008"],
            "report_periods": ["2024"],
            "report_types": ["annual"],
            "sections": ["risk"],
        }
    )
    asyncio.run(node_retrieve(_state(sub_queries=[sq]), _runtime(q, FakeLLM([])), None))
    f = q.dense_calls[0]["query_filter"]
    assert f == {
        "stock_code": ["000008"],
        "report_period": ["2024"],
        "report_type": ["annual"],
        "section": ["risk"],
    }


# ---------- HyDE ----------


def test_hyde_not_triggered_when_no_condition(monkeypatch):
    """含指标词 + 高分 + 非目标 section → 不跑 HyDE。"""
    _embedding_stub(monkeypatch)
    q = FakeQdrant(dense_hits=[_hit("d1", score=0.9)], sparse_hits=[_hit("s1", score=0.8)])
    llm = FakeLLM([])
    sq = _sq(
        question="毛利率下降的经营原因是什么情况呢",
        filters={"stock_codes": ["000008"], "report_periods": ["2024"], "sections": ["notes"]},
    )
    patch = asyncio.run(node_retrieve(_state(sub_queries=[sq]), _runtime(q, llm), None))
    assert patch["hyde_hits"] == []
    assert patch["sub_queries"][0]["hyde_used"] is False
    assert llm.calls == []


def test_hyde_used_true_even_when_empty_hits(monkeypatch):
    """hyde_used=「跑过 HyDE」不是「有命中」——空结果也要让 fuse 知道补过路。"""
    _embedding_stub(monkeypatch)
    q = FakeQdrant(dense_hits=[_hit("d1")], sparse_hits=[])
    llm = FakeLLM(["公司面临战略、市场与运营等方面的风险。"])

    class _NoHydeQdrant(FakeQdrant):
        def search_dense(self, vector, query_filter, limit):
            hits = super().search_dense(vector, query_filter, limit)
            # 首轮有命中，HyDE 二次检索返回空
            self.dense_calls  # 保持计数
            if len(self.dense_calls) > 1:
                return []
            return hits

    q2 = _NoHydeQdrant(dense_hits=[_hit("d1")])
    patch = asyncio.run(node_retrieve(_state(), _runtime(q2, llm), None))
    assert patch["hyde_hits"] == []
    assert patch["sub_queries"][0]["hyde_used"] is True


def test_hyde_llm_failure_skips_hyde_path(monkeypatch):
    _embedding_stub(monkeypatch)
    q = FakeQdrant(dense_hits=[_hit("d1")])
    llm = FakeLLM(errors=[TimeoutError("t1"), TimeoutError("t2")])
    patch = asyncio.run(node_retrieve(_state(), _runtime(q, llm), None))
    assert len(llm.calls) == 2  # 重试 1 次
    assert patch["hyde_hits"] == []
    assert patch["sub_queries"][0]["hyde_used"] is False
    assert patch["error"] is None  # 跳过 hyde 不整节点失败
    assert patch["vector_hits"]  # 另路照常


# ---------- 失败降级 ----------


def test_dense_fail_sparse_still_runs(monkeypatch):
    _embedding_stub(monkeypatch)
    q = FakeQdrant(sparse_hits=[_hit("s1")], dense_error=RuntimeError("dense down"))
    llm = FakeLLM([])
    patch = asyncio.run(node_retrieve(_state(), _runtime(q, llm), None))
    assert patch["vector_hits"] == []
    assert patch["sparse_hits"][0]["chunk_id"] == "s1"
    assert patch["error"] is None


def test_sparse_fail_dense_still_runs(monkeypatch):
    _embedding_stub(monkeypatch)
    q = FakeQdrant(dense_hits=[_hit("d1")], sparse_error=RuntimeError("sparse down"))
    patch = asyncio.run(node_retrieve(_state(), _runtime(q, FakeLLM([])), None))
    assert patch["sparse_hits"] == []
    assert patch["vector_hits"][0]["chunk_id"] == "d1"
    assert patch["error"] is None


def test_qdrant_unavailable_returns_dependency(monkeypatch):
    _embedding_stub(monkeypatch)

    class _DeadQdrant:
        def search_dense(self, *a, **k):
            raise ConnectionError("qdrant down")

        def search_sparse(self, *a, **k):
            raise ConnectionError("qdrant down")

    patch = asyncio.run(node_retrieve(_state(), _runtime(_DeadQdrant(), FakeLLM([])), None))
    assert patch["error_code"] == "DEPENDENCY_UNAVAILABLE"
    assert patch["error_stage"] == "retrieve"


def test_hits_carry_source_and_sub_id(monkeypatch):
    _embedding_stub(monkeypatch)
    q = FakeQdrant(dense_hits=[_hit("d1")], sparse_hits=[_hit("s1")])
    patch = asyncio.run(node_retrieve(_state(), _runtime(q, FakeLLM([])), None))
    for hit in patch["vector_hits"] + patch["sparse_hits"]:
        assert hit["sub_id"] == "sq1"
        assert hit["source"] in {"dense", "sparse", "hyde"}
