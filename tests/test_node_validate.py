"""node_validate 节点测试（docs/04k §3/§9）。

假 evidence/citations + 假 answer 注入，断言页码对账、数字对账（一致/不一致/容差边界）、
孤立数字、空证据合法、段落引用、metric_value 对账、SSE 形状、宁拒勿猜。
不碰真实模型 / DB / LLM。
"""
from __future__ import annotations

import asyncio

from langgraph.runtime import Runtime

from app.query_graph.context import QueryAgentContext
from app.query_graph.nodes.validate import node_validate
from app.query_graph.state import default_state


class _Writer:
    def __init__(self):
        self.events: list[dict] = []

    def __call__(self, item: dict) -> None:
        self.events.append(item)


def _runtime() -> Runtime:
    return Runtime(context=QueryAgentContext())  # type: ignore[arg-type]


def _ev(ref, type="narrative", content="正文", page_no=10, **over):
    r = {
        "ref": ref,
        "type": type,
        "content": content,
        "snippet": content[:120],
        "page_no": page_no,
        "document_id": "doc1",
        "stock_code": "000008",
        "report_period": "2024",
        "source": "dense",
        "score": 0.04,
        "sub_id": "sq1",
        "metric_code": None,
        "status": None,
        "metric_value": None,
    }
    r.update(over)
    return r


def _state(answer="", citations=None, evidence=None, **kw):
    s = default_state(query="q", request_id="q1", session_id="s1")
    s["answer"] = answer
    s["citations"] = citations if citations is not None else []
    s["evidence"] = evidence if evidence is not None else []
    s.update(kw)
    return s


def _run(state, writer=None):
    return asyncio.run(node_validate(state, _runtime(), writer))


def _cite(ref, page_no=10, **over):
    c = {
        "ref": ref,
        "document_id": "doc1",
        "stock_code": "000008",
        "report_period": "2024",
        "page_no": page_no,
        "snippet": "片段",
        "type": "narrative",
    }
    c.update(over)
    return c


# ---------- 通过基线 ----------


def test_pass_emits_done():
    """通过 → done 事件 + error 三字段 None（04k §3.4）。"""
    w = _Writer()
    out = _run(
        _state(
            answer="毛利率为 34.2%[1]",
            citations=[_cite(1, page_no=11)],
            evidence=[_ev(1, content="毛利率 34.2%", page_no=11)],
        ),
        w,
    )
    assert out["error"] is None
    assert out["error_code"] is None
    assert out["error_stage"] is None
    done = [e for e in w.events if e["type"] == "done"]
    assert len(done) == 1
    assert done[0]["request_id"] == "q1"


def test_pass_no_token_citation_events():
    w = _Writer()
    _run(
        _state(
            answer="34.2%[1]",
            citations=[_cite(1)],
            evidence=[_ev(1, content="34.2%")],
        ),
        w,
    )
    assert all(e["type"] not in ("token", "citation") for e in w.events)


# ---------- 页码对账 ----------


def test_ref_not_in_evidence_rejected():
    out = _run(
        _state(
            answer="34.2%[99]",
            citations=[_cite(99)],
            evidence=[_ev(1, content="34.2%")],
        )
    )
    assert out["error_code"] == "INSUFFICIENT_EVIDENCE"
    assert out["error_stage"] == "validate"


def test_page_no_mismatch_rejected():
    """citations[n].page_no ≠ evidence[n].page_no → 拒（04k §3.1）。"""
    out = _run(
        _state(
            answer="34.2%[1]",
            citations=[_cite(1, page_no=45)],
            evidence=[_ev(1, content="34.2%", page_no=46)],
        )
    )
    assert out["error_code"] == "INSUFFICIENT_EVIDENCE"
    assert "页码" in out["error"] or "page" in out["error"].lower()


def test_page_no_both_none_ok():
    out = _run(
        _state(
            answer="毛利率 34.2%[1]",
            citations=[_cite(1, page_no=None)],
            evidence=[_ev(1, content="毛利率 34.2%", page_no=None)],
        )
    )
    assert out["error"] is None


def test_page_no_one_none_rejected():
    """一侧 None 一侧有值 → 不一致（04k §3.1）。"""
    out = _run(
        _state(
            answer="34.2%[1]",
            citations=[_cite(1, page_no=None)],
            evidence=[_ev(1, content="34.2%", page_no=45)],
        )
    )
    assert out["error_code"] == "INSUFFICIENT_EVIDENCE"


# ---------- 数字对账 ----------


def test_number_match_passes():
    out = _run(
        _state(
            answer="毛利率为 34.2%[1]",
            citations=[_cite(1)],
            evidence=[_ev(1, content="毛利率 34.2%")],
        )
    )
    assert out["error"] is None


def test_number_mismatch_rejected():
    """差 0.1 > 0.01 → 拒（04k §3.2）。"""
    out = _run(
        _state(
            answer="毛利率为 34.3%[1]",
            citations=[_cite(1)],
            evidence=[_ev(1, content="毛利率 34.2%")],
        )
    )
    assert out["error_code"] == "INSUFFICIENT_EVIDENCE"


def test_number_mismatch_points_to_real_ref():
    """串号拒答要指出数字实际在哪条证据——比只报不一致好排查。"""
    out = _run(
        _state(
            answer="好当家归母净利润 43,427,553.57[2]",
            citations=[_cite(1), _cite(2)],
            evidence=[
                _ev(1, type="metric", content="好当家(600467) 归母净利润 2024：43427553.57元"),
                _ev(2, type="metric", content="好当家(600467) 扣非净利润 2024：12895999.88元"),
            ],
        )
    )
    assert out["error_code"] == "INSUFFICIENT_EVIDENCE"
    assert "实际在 [1]" in out["error"]


def test_number_tolerance_boundary():
    """差 0.01 ≤ 0.01 → 通过（04k §3.2）。"""
    out = _run(
        _state(
            answer="毛利率为 34.21%[1]",
            citations=[_cite(1)],
            evidence=[_ev(1, content="毛利率 34.2%")],
        )
    )
    assert out["error"] is None


def test_number_match_metric_value():
    """兼容 metric_value 字段对账（04k §3.2）。"""
    out = _run(
        _state(
            answer="营收为 20.8 亿[1]",
            citations=[_cite(1)],
            evidence=[_ev(1, type="metric", content="营收 2024：20.8 亿", metric_value=20.8)],
        )
    )
    assert out["error"] is None


def test_number_comma_thousands():
    """千分位数字 1,241 → 1241.0 对账（04k §3.2）。"""
    out = _run(
        _state(
            answer="营收 1,241 亿[1]",
            citations=[_cite(1)],
            evidence=[_ev(1, content="营收 1241 亿")],
        )
    )
    assert out["error"] is None


def test_number_signed():
    """带符号数字 +1.1pct → 1.1 对账（04k §3.2）。"""
    out = _run(
        _state(
            answer="同比 +1.1pct[1]",
            citations=[_cite(1)],
            evidence=[_ev(1, content="同比 1.1pct")],
        )
    )
    assert out["error"] is None


def test_cot_number_mismatch_rejected():
    """推理段错数也拦（04k §1.2 全文对账）。"""
    out = _run(
        _state(
            answer="【思考】证据 [1] 给出 34.3%[1]。\n【答案】毛利率为 34.2%[1]",
            citations=[_cite(1)],
            evidence=[_ev(1, content="毛利率 34.2%")],
        )
    )
    assert out["error_code"] == "INSUFFICIENT_EVIDENCE"


def test_multiple_refs_one_bad_rejected():
    """一对不过整条拒（04k §8 宁拒勿猜）。"""
    out = _run(
        _state(
            answer="A 20 亿[1] 与 B 30 亿[2]",
            citations=[_cite(1), _cite(2, page_no=12)],
            evidence=[
                _ev(1, content="A 20 亿", page_no=11),
                _ev(2, content="B 25 亿", page_no=12),
            ],
        )
    )
    assert out["error_code"] == "INSUFFICIENT_EVIDENCE"


# ---------- 孤立数字 ----------


def test_orphan_numbers_rejected():
    """含数字却零引用 → 拒（04k §3.3）。"""
    out = _run(
        _state(
            answer="毛利率为 34.2%",
            citations=[],
            evidence=[_ev(1, content="毛利率 34.2%")],
        )
    )
    assert out["error_code"] == "INSUFFICIENT_EVIDENCE"


def test_no_digits_no_citations_passes():
    """段落引用/定性转述：无数字无引用 → 过（04k §4）。"""
    out = _run(
        _state(
            answer="该公司主营轨道交通维护。",
            citations=[],
            evidence=[_ev(1, content="公司主营轨道交通维护")],
        )
    )
    assert out["error"] is None


# ---------- 边界 ----------


def test_empty_evidence_fixed_message_passes():
    """evidence=[] + 固定话术 + citations=[] → done（04k §4）。"""
    w = _Writer()
    out = _run(
        _state(
            answer="未找到相关年报证据，无法回答该问题。",
            citations=[],
            evidence=[],
        ),
        w,
    )
    assert out["error"] is None
    assert any(e["type"] == "done" for e in w.events)


def test_paragraph_citation_no_number_ok():
    """段落引用（无数字）：只查页码，不做数字对账（04k §4）。"""
    out = _run(
        _state(
            answer="公司主营轨道交通维护[1]",
            citations=[_cite(1)],
            evidence=[_ev(1, content="公司主营轨道交通维护")],
        )
    )
    assert out["error"] is None


def test_status_unsupported_qualitative_passes():
    """status≠ok 定性转述无数+标对，vacuously 过（04k §5.2）。"""
    out = _run(
        _state(
            answer="该指标暂不支持[1]",
            citations=[_cite(1)],
            evidence=[
                _ev(1, type="calc", content="毛利率 CAGR：status=unsupported，暂不支持", status="unsupported"),
            ],
        )
    )
    assert out["error"] is None


def test_fail_keeps_answer_citations():
    """失败不覆盖 answer/citations（04k §7.2）。"""
    answer = "毛利率为 34.3%[1]"
    citations = [_cite(1)]
    out = _run(
        _state(
            answer=answer,
            citations=citations,
            evidence=[_ev(1, content="毛利率 34.2%")],
        )
    )
    assert out.get("answer", answer) == answer  # 不覆盖则不在 patch 中
    assert out["error_code"] == "INSUFFICIENT_EVIDENCE"


def test_fail_emits_error_sse():
    w = _Writer()
    _run(
        _state(
            answer="34.3%[1]",
            citations=[_cite(1)],
            evidence=[_ev(1, content="34.2%")],
        ),
        w,
    )
    errors = [e for e in w.events if e["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["code"] == "INSUFFICIENT_EVIDENCE"
    assert errors[0]["stage"] == "validate"
    assert errors[0]["message"]


def test_must_return_error_fields():
    out = _run(
        _state(answer="34.2%[1]", citations=[_cite(1)], evidence=[_ev(1, content="34.2%")])
    )
    assert "error" in out
    assert "error_code" in out
    assert "error_stage" in out
