"""node_rerank 节点测试（docs/04i §3/§9）。

假 evidence + mock rerank_pairs 注入，断言只重排向量类、断崖截断、ref 保留、
降级原序、纯指标题跳过、ranked 语义。不碰真实模型 / DB / LLM。
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from langgraph.runtime import Runtime

from app.query_graph.context import QueryAgentContext
from app.query_graph.nodes.rerank import RERANK_MAX_TOPK, node_rerank
from app.query_graph.state import default_state


class _Writer:
    def __init__(self):
        self.events: list[dict] = []

    def __call__(self, item: dict) -> None:
        self.events.append(item)


def _runtime() -> Runtime:
    return Runtime(context=QueryAgentContext())  # type: ignore[arg-type]


def _ev(ref, type="narrative", score=0.04, content="正文段落", **over):
    r = {
        "ref": ref,
        "type": type,
        "content": content,
        "snippet": content[:120],
        "page_no": 10 + ref,
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


def _state(evidence=None, **kw):
    s = default_state(query="q", request_id="q1", session_id="s1")
    s["evidence"] = evidence if evidence is not None else []
    s.update(kw)
    return s


def _run(state, writer=None):
    return asyncio.run(node_rerank(state, _runtime(), writer))


def _mock_scores(*scores):
    return patch(
        "app.query_graph.nodes.rerank.rerank_pairs",
        side_effect=lambda q, passages: list(scores[: len(passages)]),
    )


# ---------- 重排与覆盖 ----------


def test_rerank_reorders_vectors():
    """mock 分 [0.55,0.85] → 原序 A,B 变 B,A；分差避开断崖（04i §3.4）。"""
    with _mock_scores(0.80, 0.82):
        out = _run(_state([_ev(1, content="A"), _ev(2, content="B")]))
    assert [e["content"] for e in out["evidence"]] == ["B", "A"]


def test_score_overwritten_ranked_true():
    with _mock_scores(0.77):
        out = _run(_state([_ev(1, score=0.0164)]))
    [e] = out["evidence"]
    assert e["score"] == 0.77
    assert e["ranked"] is True


def test_ref_preserved_not_renumbered():
    """ref 保留 fuse 编号不重编（04h §5.2 / 04i §3.4）。"""
    with _mock_scores(0.80, 0.82):
        out = _run(_state([_ev(7), _ev(12)]))
    assert sorted(e["ref"] for e in out["evidence"]) == [7, 12]


def test_rewritten_query_preferred():
    """问句优先 rewritten_query（04i §3.2）。"""
    seen = {}

    def _fake(q, passages):
        seen["q"] = q
        return [0.5] * len(passages)

    with patch("app.query_graph.nodes.rerank.rerank_pairs", side_effect=_fake):
        _run(_state([_ev(1)], query="原话", rewritten_query="补全句"))
    assert seen["q"] == "补全句"


def test_fallback_to_query_when_rewrite_empty():
    seen = {}

    def _fake(q, passages):
        seen["q"] = q
        return [0.5] * len(passages)

    with patch("app.query_graph.nodes.rerank.rerank_pairs", side_effect=_fake):
        _run(_state([_ev(1)], query="原话", rewritten_query=""))
    assert seen["q"] == "原话"


# ---------- metric / calc 不动 ----------


def test_struct_untouched_at_tail():
    """metric/calc 位次不动，拼回向量类之后（04i §4.2）。"""
    with _mock_scores(0.80, 0.82):
        out = _run(
            _state(
                [
                    _ev(1, content="V1"),
                    _ev(2, content="V2"),
                    _ev(3, type="metric", metric_code="revenue", score=0.0),
                    _ev(4, type="calc", metric_code="revenue_yoy", score=0.0, status="ok"),
                ]
            )
        )
    types = [e["type"] for e in out["evidence"]]
    assert types == ["narrative", "narrative", "metric", "calc"]
    # 向量重排后 V2 分 0.85 在前
    assert out["evidence"][0]["content"] == "V2"
    assert out["evidence"][2]["ref"] == 3
    assert out["evidence"][3]["ref"] == 4


def test_struct_ranked_false_score_zero():
    with _mock_scores(0.9):
        out = _run(
            _state([_ev(1), _ev(2, type="metric", score=0.0, metric_code="revenue")])
        )
    metric = [e for e in out["evidence"] if e["type"] == "metric"][0]
    assert metric["ranked"] is False
    assert metric["score"] == 0.0


# ---------- 断崖截断 ----------


def test_cliff_truncates_vectors():
    """5 条向量在第 3 名断崖 → 向量类只剩 3，struct 不受影响。"""
    scores = [0.93, 0.91, 0.90, 0.39, 0.38]
    evs = [_ev(i + 1, content=f"V{i}") for i in range(5)]
    evs.append(_ev(99, type="metric", score=0.0, metric_code="revenue"))
    with _mock_scores(*scores):
        out = _run(_state(evs))
    vec = [e for e in out["evidence"] if e["type"] != "metric"]
    st = [e for e in out["evidence"] if e["type"] == "metric"]
    assert len(vec) == 3
    assert len(st) == 1


def test_max_topk_bounds():
    """15 条向量无断崖 → 10 条（04i §4.1）。"""
    evs = [_ev(i + 1, content=f"V{i}") for i in range(15)]
    scores = [0.5] * 15
    with _mock_scores(*scores):
        out = _run(_state(evs))
    assert len(out["evidence"]) == RERANK_MAX_TOPK


# ---------- 空 / 纯指标题 ----------


def test_empty_evidence_legal():
    out = _run(_state([]))
    assert out["evidence"] == []
    assert out["error"] is None


def test_pure_metric_skips_model():
    """纯指标题：向量类空 → 不调模型，结构化原样返回（04i §8）。"""
    with patch("app.query_graph.nodes.rerank.rerank_pairs") as mock:
        out = _run(
            _state([_ev(1, type="metric", score=0.0, metric_code="revenue")])
        )
        mock.assert_not_called()
    assert len(out["evidence"]) == 1
    assert out["evidence"][0]["ranked"] is False


def test_empty_content_scores_zero():
    """空 content 不进模型，给 0 分 + ranked=False（04i §8）。

    0.8→0.0 会触发 gap_abs 断崖，空条目被截掉是合法结果——单独验证给分语义。
    """
    with _mock_scores():
        out = _run(_state([_ev(1, content="")]))
    [e] = out["evidence"]
    assert e["score"] == 0.0
    assert e["ranked"] is False


# ---------- 降级 ----------


def test_degrade_keeps_rrf_order_top_max():
    """模型抛错 → 降级 RRF 原序取前 10，ranked=False（04i §8）。"""
    evs = [_ev(i + 1, content=f"V{i}", score=0.04) for i in range(15)]
    with patch(
        "app.query_graph.nodes.rerank.rerank_pairs",
        side_effect=RuntimeError("model missing"),
    ):
        out = _run(_state(evs))
    assert out["error"] is None
    assert len(out["evidence"]) == RERANK_MAX_TOPK
    assert [e["ref"] for e in out["evidence"]] == list(range(1, RERANK_MAX_TOPK + 1))
    assert all(e["ranked"] is False for e in out["evidence"])


def test_degrade_keeps_rrf_score():
    evs = [_ev(1, score=0.0164)]
    with patch(
        "app.query_graph.nodes.rerank.rerank_pairs",
        side_effect=RuntimeError("boom"),
    ):
        out = _run(_state(evs))
    assert out["evidence"][0]["score"] == 0.0164


def test_degrade_must_return_evidence():
    """降级后仍必须 return evidence（04i §8 避坑）。"""
    with patch(
        "app.query_graph.nodes.rerank.rerank_pairs",
        side_effect=RuntimeError("x"),
    ):
        out = _run(_state([_ev(1)]))
    assert isinstance(out["evidence"], list)
    assert len(out["evidence"]) == 1


# ---------- 边界 ----------


def test_must_return_evidence():
    with _mock_scores(0.5):
        out = _run(_state([_ev(1)]))
    assert "evidence" in out
    assert isinstance(out["evidence"], list)


def test_metadata_preserved():
    with _mock_scores(0.9):
        out = _run(
            _state([_ev(3, sub_id="sq9", page_no=45, document_id="docX")])
        )
    [e] = out["evidence"]
    assert e["sub_id"] == "sq9"
    assert e["page_no"] == 45
    assert e["document_id"] == "docX"
