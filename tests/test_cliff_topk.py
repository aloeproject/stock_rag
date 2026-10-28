"""cliff_topk 纯函数测试（docs/04i §3.3/§9）。

断崖双阈值、min/max、平分稳定序、空合法。不碰模型 / 节点。
"""
from __future__ import annotations

from app.query_graph.matching.cliff_topk import (
    RERANK_GAP_ABS,
    RERANK_GAP_RATIO,
    RERANK_MAX_TOPK,
    RERANK_MIN_TOPK,
    cliff_topk,
)


def _scored(*scores, start_ref=1):
    return [{"ref": start_ref + i, "score": s} for i, s in enumerate(scores)]


def test_constants():
    assert RERANK_MAX_TOPK == 10
    assert RERANK_MIN_TOPK == 1
    assert RERANK_GAP_ABS == 0.5
    assert RERANK_GAP_RATIO == 0.25


def test_gap_abs_triggers():
    """0.90→0.39 绝对差 0.51 ≥ 0.5 → 截到 3（04i §7.3）。"""
    out = cliff_topk(_scored(0.93, 0.91, 0.90, 0.39, 0.38))
    assert [e["score"] for e in out] == [0.93, 0.91, 0.90]


def test_rel_triggers():
    """0.40→0.30 相对落差 0.25 ≥ 0.25 → 截到 1（软滑坡，04i §7.3）。"""
    out = cliff_topk(_scored(0.40, 0.30, 0.20))
    assert [e["score"] for e in out] == [0.40]


def test_rel_below_threshold_no_cut():
    out = cliff_topk(_scored(0.40, 0.35, 0.32))
    assert len(out) == 3


def test_no_cliff_takes_max():
    """无断崖顶格 MAX_TOPK，不看列表多长（04i §3.3）。"""
    scores = [0.60 - i * 0.001 for i in range(RERANK_MAX_TOPK + 5)]
    out = cliff_topk(_scored(*scores))
    assert len(out) == RERANK_MAX_TOPK


def test_max_cap_before_cliff_search():
    """先 cap 到 10 再找断崖：第 15 位的断崖看不见（04i §4.3）。"""
    scores = [0.9] * 15 + [0.1] * 5
    out = cliff_topk(_scored(*scores))
    assert len(out) == RERANK_MAX_TOPK


def test_min_one_when_single():
    out = cliff_topk(_scored(0.9))
    assert len(out) == 1


def test_min_one_on_immediate_cliff():
    """第 1 名与第 2 名即断崖也保住 1 条（04i §4.3）。"""
    out = cliff_topk(_scored(0.9, 0.1, 0.05))
    assert [e["score"] for e in out] == [0.9]


def test_tie_keeps_ref_order():
    """平分不断崖，次序按 ref 升序稳定（04i §3.3）。"""
    items = [
        {"ref": 3, "score": 0.5},
        {"ref": 1, "score": 0.5},
        {"ref": 2, "score": 0.5},
    ]
    out = cliff_topk(items)
    assert [e["ref"] for e in out] == [1, 2, 3]


def test_sorts_descending_before_cut():
    """入参乱序也能按分降序找断崖（cliff 自带排序）。"""
    out = cliff_topk(_scored(0.39, 0.93, 0.91, 0.90, 0.38))
    assert [e["score"] for e in out] == [0.93, 0.91, 0.90]


def test_empty_legal():
    assert cliff_topk([]) == []


def test_negative_score_rel_uses_abs():
    """s1 为负时 rel=gap/|s1| 仍是正落差（04i §3.3）。"""
    out = cliff_topk(_scored(-0.1, -0.5))
    assert len(out) == 1
