"""node_fuse 节点测试（docs/04h §3/§9）。

纯函数层：假 hits / metric_rows / calc_results 注入 State，断言 RRF 累加、
chunk_id 去重、截断次序、结构化归一、sub_id 对齐、SSE 纪律。不碰 DB / LLM。
"""
from __future__ import annotations

import asyncio

from langgraph.runtime import Runtime

from app.query_graph.context import QueryAgentContext
from app.query_graph.nodes.fuse import (
    FUSE_TOP_K,
    RRF_K,
    RRF_WEIGHTS,
    node_fuse,
    rrf_score,
)
from app.query_graph.state import default_state


class _Writer:
    def __init__(self):
        self.events: list[dict] = []

    def __call__(self, item: dict) -> None:
        self.events.append(item)


def _runtime() -> Runtime:
    return Runtime(context=QueryAgentContext())  # type: ignore[arg-type]


def _hit(chunk_id, source="dense", rank_score=0.9, sub_id="sq1", **over):
    r = {
        "chunk_id": chunk_id,
        "content": f"正文 {chunk_id}",
        "page_no": 10,
        "section": "management_discussion",
        "chunk_type": "narrative",
        "score": rank_score,
        "title": f"标题 {chunk_id}",
        "document_id": "doc1",
        "stock_code": "000008",
        "report_period": "2024",
        "sub_id": sub_id,
        "source": source,
    }
    r.update(over)
    return r


def _mrow(code="revenue", period="2024", value=20.82, sub_id="sq1", **over):
    r = {
        "stock_code": "000008",
        "metric_code": code,
        "metric_name": "营业收入",
        "metric_period": period,
        "metric_period_type": "current",
        "metric_value": value,
        "unit": "亿元",
        "statement_scope": "consolidated",
        "page_no": 45,
        "sub_id": sub_id,
    }
    r.update(over)
    return r


def _crow(code="revenue_yoy", kind="yoy", value=10.0, status="ok", sub_id="sq1", **over):
    r = {
        "stock_code": "000008",
        "metric_code": code,
        "metric_name": "营业收入",
        "kind": kind,
        "periods": ["2024", "2023"],
        "value": value,
        "values": None,
        "unit": "%",
        "formula": "(current-prior)/abs(prior)*100",
        "status": status,
        "reason": None,
        "sub_id": sub_id,
    }
    r.update(over)
    return r


def _state(hits=None, sparse=None, hyde=None, metrics=None, calcs=None, **kw):
    s = default_state(query="q", request_id="q1", session_id="s1")
    s["vector_hits"] = hits if hits is not None else []
    s["sparse_hits"] = sparse if sparse is not None else []
    s["hyde_hits"] = hyde if hyde is not None else []
    s["metric_rows"] = metrics if metrics is not None else []
    s["calc_results"] = calcs if calcs is not None else []
    s.update(kw)
    return s


def _run(state, writer=None):
    return asyncio.run(node_fuse(state, _runtime(), writer))


# ---------- RRF 纯函数 ----------


def test_rrf_score_formula():
    # k=60：rank1 → 1/61
    assert abs(rrf_score({"dense": 1}) - 1.0 / 61) < 1e-9


def test_rrf_weights_applied():
    s = rrf_score({"dense": 1, "sparse": 1, "hyde": 1})
    expect = 1.0 / 61 + 0.9 / 61 + 0.8 / 61
    assert abs(s - expect) < 1e-9


def test_rrf_constants():
    assert RRF_K == 60
    assert RRF_WEIGHTS == {"dense": 1.0, "sparse": 0.9, "hyde": 0.8}


# ---------- 去重与累加 ----------


def test_same_chunk_multi_route_merged():
    """同 chunk_id 多路命中 → 1 条，score 加权累加，source 含全部命中路（04h §3.1）。"""
    patch = _run(
        _state(
            hits=[_hit("A", "dense")],
            sparse=[_hit("A", "sparse")],
            hyde=[_hit("A", "hyde")],
        )
    )
    [e] = patch["evidence"]
    expect = 1.0 / 61 + 0.9 / 61 + 0.8 / 61
    assert abs(e["score"] - expect) < 1e-9
    assert set(e["source"].split("+")) == {"dense", "sparse", "hyde"}
    assert e["ref"] == 1


def test_raw_score_discarded():
    """各路原始 score 不可比，evidence 只留 RRF 分（04h §3.1）。"""
    patch = _run(_state(hits=[_hit("A", "dense", rank_score=0.99)]))
    [e] = patch["evidence"]
    assert abs(e["score"] - 1.0 / 61) < 1e-9


def test_source_single_route():
    patch = _run(_state(sparse=[_hit("B", "sparse")]))
    [e] = patch["evidence"]
    assert e["source"] == "sparse"


# ---------- 排序与截断 ----------


def test_rank_order_dense_beats_hyde_same_rank():
    patch = _run(_state(hits=[_hit("A", "dense")], hyde=[_hit("B", "hyde")]))
    a, b = patch["evidence"]
    assert a["chunk_id"] == "A"
    assert b["chunk_id"] == "B"


def test_full_merge_before_truncate():
    """共识 chunk 各路 rank 20~30 也能进——先截断会丢（04h §4.1）。"""
    # dense 只有 X 在第 25 位，sparse 只有 X 在第 25 位：累加后应排最前
    dense = [_hit(f"d{i}", "dense") for i in range(24)] + [_hit("X", "dense")]
    sparse = [_hit(f"s{i}", "sparse") for i in range(24)] + [_hit("X", "sparse")]
    patch = _run(_state(hits=dense, sparse=sparse))
    assert patch["evidence"][0]["chunk_id"] == "X"


def test_truncate_to_fuse_top_k():
    hits = [_hit(f"C{i:03d}", "dense") for i in range(FUSE_TOP_K + 5)]
    patch = _run(_state(hits=hits))
    assert len(patch["evidence"]) == FUSE_TOP_K


def test_tie_break_by_chunk_id():
    """平分按 chunk_id 字典序——RRF 真平分难构造，直接喂同分项（04h §3.2）。"""
    from app.query_graph.nodes.fuse import step_2_sort_truncate

    out = step_2_sort_truncate(
        [{"chunk_id": "B", "score": 1.0}, {"chunk_id": "A", "score": 1.0}]
    )
    assert [e["chunk_id"] for e in out] == ["A", "B"]


# ---------- metric / calc 归一 ----------


def test_metric_row_becomes_evidence():
    patch = _run(_state(metrics=[_mrow()]))
    [e] = patch["evidence"]
    assert e["type"] == "metric"
    assert e["ref"] == 1
    assert "营业收入" in e["content"]
    assert "2024" in e["content"]
    assert e["page_no"] == 45
    assert e["metric_code"] == "revenue"
    assert e["source"] == "metric"
    assert e["score"] == 0.0


def test_metric_content_carries_stock_identity():
    """compare 题两家同码：content 必须带 名称(代码)，否则 LLM 串引用（04h §3.3）。"""
    patch = _run(
        _state(
            metrics=[
                _mrow(code="net_profit_attributable", value=43427553.57, unit="元", stock_code="600467"),
                _mrow(
                    code="net_profit_attributable",
                    value=22359786600.0,
                    unit="元",
                    stock_code="601225",
                ),
            ],
            resolved_names={"好当家": "600467", "陕西煤业": "601225"},
        )
    )
    c0, c1 = patch["evidence"][0]["content"], patch["evidence"][1]["content"]
    assert "好当家(600467)" in c0
    assert "陕西煤业(601225)" in c1
    assert "好当家" not in c1


def test_metric_content_carries_dim_name():
    """切片行 content 必须带 dim_name：不带维度串引用更狠（04h §3.3 / 02g §8.2）。"""
    patch = _run(
        _state(
            metrics=[
                _mrow(code="revenue", value=300.0, dim_type="industry", dim_name="煤炭采掘业"),
                _mrow(code="revenue", value=200.0, dim_type="product", dim_name="自产煤"),
            ]
        )
    )
    c0, c1 = patch["evidence"][0]["content"], patch["evidence"][1]["content"]
    assert "煤炭采掘业" in c0
    assert "自产煤" in c1
    assert "自产煤" not in c0


def test_calc_result_becomes_evidence():
    patch = _run(_state(calcs=[_crow()]))
    [e] = patch["evidence"]
    assert e["type"] == "calc"
    assert e["metric_code"] == "revenue_yoy"
    assert "yoy" in e["content"]
    assert e["status"] == "ok"


def test_calc_unsupported_still_included():
    """status≠ok 仍进 evidence：generate 要说「不支持」而不是装没查（04h §3.3）。"""
    patch = _run(
        _state(calcs=[_crow(code="margin_vol", kind="unsupported", value=None, status="unsupported", reason="暂不支持波动率")])
    )
    [e] = patch["evidence"]
    assert e["status"] == "unsupported"
    assert "unsupported" in e["content"]
    assert "暂不支持波动率" in e["content"]


def test_multi_period_values_in_content():
    patch = _run(
        _state(
            calcs=[
                _crow(
                    code="revenue_multi_period",
                    kind="multi_period",
                    value=None,
                    values=[100.0, 150.0, 200.0],
                    unit="",
                )
            ]
        )
    )
    [e] = patch["evidence"]
    assert "100.0" in e["content"]
    assert "200.0" in e["content"]


def test_pure_metric_question_no_vector_ok():
    """纯指标题：三路空 + metric 有 → evidence 非空（04h §9）。"""
    patch = _run(_state(metrics=[_mrow()]))
    assert len(patch["evidence"]) == 1


def test_struct_not_ranked_by_rrf():
    """metric/calc 不进 RRF：score=0.0，不与向量分混排（04h §4.2）。"""
    patch = _run(_state(hits=[_hit("A", "dense")], metrics=[_mrow()]))
    vec = [e for e in patch["evidence"] if e["type"] != "metric"]
    st = [e for e in patch["evidence"] if e["type"] == "metric"]
    assert vec[0]["score"] > 0
    assert st[0]["score"] == 0.0


# ---------- sub_id 对齐 / ref ----------


def test_ref_continuous_vectors_first():
    patch = _run(_state(hits=[_hit("A", "dense")], metrics=[_mrow()], calcs=[_crow()]))
    refs = [e["ref"] for e in patch["evidence"]]
    assert refs == [1, 2, 3]
    types = [e["type"] for e in patch["evidence"]]
    assert types == ["narrative", "metric", "calc"]


def test_sub_id_carried():
    patch = _run(_state(hits=[_hit("A", "dense", sub_id="sq9")], metrics=[_mrow(sub_id="sq9")]))
    assert all(e["sub_id"] == "sq9" for e in patch["evidence"])


# ---------- SSE ----------


def test_sse_only_vector_and_ok_struct():
    w = _Writer()
    _run(
        _state(
            hits=[_hit("A", "dense")],
            metrics=[_mrow()],
            calcs=[
                _crow(),
                _crow(code="x_ins", status="insufficient_periods", reason="缺期", value=None),
            ],
        ),
        w,
    )
    assert len(w.events) == 1
    rows = w.events[0]["rows"]
    assert w.events[0]["type"] == "evidence"
    codes = [r.get("metric_code") for r in rows]
    assert "x_ins" not in [c for c in codes if c]
    assert all(r["status"] in (None, "ok") for r in rows)


def test_sse_snippet_truncated():
    w = _Writer()
    long_hit = _hit("A", "dense", content="长" * 500)
    _run(_state(hits=[long_hit]), w)
    assert len(w.events[0]["rows"][0]["snippet"]) <= 120


def test_sse_no_event_when_all_insufficient():
    w = _Writer()
    _run(_state(calcs=[_crow(status="unsupported", reason="不支持", value=None)]), w)
    assert w.events == []


# ---------- 空合法 ----------


def test_all_empty_legal():
    patch = _run(_state())
    assert patch["evidence"] == []
    assert patch["error"] is None


def test_hyde_empty_not_error():
    patch = _run(_state(hits=[_hit("A", "dense")], hyde=[]))
    assert patch["error"] is None
    assert len(patch["evidence"]) == 1


def test_must_return_evidence():
    """避坑清单：缺 return 下游 rerank 拿空（04h §4.1）。"""
    patch = _run(_state(hits=[_hit("A", "dense")]))
    assert "evidence" in patch
    assert isinstance(patch["evidence"], list)
