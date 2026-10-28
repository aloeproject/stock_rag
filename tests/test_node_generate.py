"""node_generate 节点测试（docs/04j §3/§9）。

假 evidence + FakeLLM（stream）注入，断言 context 拼接/截断、[n] 抽取
（保序/去重/越界）、空证据话术、token 推送、重试路径、Redis 回写、citation SSE。
不碰真实模型 / DB / LLM。
"""
from __future__ import annotations

import asyncio
import re
from unittest.mock import MagicMock, patch

from langgraph.runtime import Runtime

from app.query_graph.context import QueryAgentContext
from app.query_graph.nodes.generate import (
    MAX_CONTEXT_CHARS,
    HISTORY_TURNS,
    node_generate,
)
from app.query_graph.state import default_state


class FakeLLM:
    """stream 接口同构 ChatOpenAI：yield chunk.content 增量。

    payloads 项支持 str / list[str] / Exception，分别表示单 chunk、多 chunk、抛错。
    """

    def __init__(self, payloads=None, errors=None):
        self.calls: list[str] = []
        self.payloads = list(payloads or [])
        self.errors = list(errors or [])

    def stream(self, prompt):
        self.calls.append(prompt)
        if self.errors:
            raise self.errors.pop(0)
        item = self.payloads.pop(0) if self.payloads else "【思考】x\n【答案】y"
        if isinstance(item, Exception):
            raise item
        chunks = [item] if isinstance(item, str) else list(item)
        for delta in chunks:
            yield type("Chunk", (), {"content": delta})()


class _Writer:
    def __init__(self):
        self.events: list[dict] = []

    def __call__(self, item: dict) -> None:
        self.events.append(item)


def _runtime(llm=None) -> Runtime:
    return Runtime(context=QueryAgentContext(llm=llm))  # type: ignore[arg-type]


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
    s = default_state(query="毛利率是多少", request_id="q1", session_id="s1")
    s["evidence"] = evidence if evidence is not None else []
    s.update(kw)
    return s


def _run(state, llm=None, writer=None):
    return asyncio.run(node_generate(state, _runtime(llm), writer))


def _ok_answer(text="【思考】证据 [1] 给出 34.2%。\n【答案】毛利率为 34.2%[1]"):
    return text


# ---------- 依赖 / 空证据 ----------


def test_llm_missing_dependency_unavailable():
    out = _run(_state([_ev(1)]), llm=None)
    assert out["error_code"] == "DEPENDENCY_UNAVAILABLE"
    assert out["error_stage"] == "generate"
    assert out["answer"] == ""


def test_empty_evidence_fixed_message_no_llm():
    """evidence=[] → 固定话术 + citations=[]，不调 LLM（04j §8）。"""
    llm = FakeLLM()
    out = _run(_state([]), llm=llm)
    assert "未找到" in out["answer"]
    assert out["citations"] == []
    assert llm.calls == []
    assert out["error"] is None


# ---------- context 拼接 ----------


def test_context_entry_format():
    """每条带 [ref][type=][source=][title=][score=] + content（04j §3.1）。"""
    llm = FakeLLM(payloads=[_ok_answer()])
    out = _run(_state([_ev(1, content="营收 20 亿")]), llm=llm)
    prompt = llm.calls[0]
    assert "[1][type=narrative][source=dense]" in prompt
    assert "title=000008 2024" in prompt
    assert "score=0.04" in prompt
    assert "营收 20 亿" in prompt


def test_context_title_metric_uses_metric_code():
    llm = FakeLLM(payloads=[_ok_answer()])
    _run(
        _state(
            [_ev(3, type="metric", metric_code="gross_margin", content="毛利率 2024：34.2%", score=0.0)]
        ),
        llm=llm,
    )
    prompt = llm.calls[0]
    assert "[3][type=metric]" in prompt
    assert "title=000008 gross_margin" in prompt


def test_context_ref_not_renumbered():
    """ref 保留 fuse 编号，缺口合法（04j §3.1）。"""
    llm = FakeLLM(payloads=[_ok_answer()])
    _run(_state([_ev(2), _ev(7)]), llm=llm)
    prompt = llm.calls[0]
    assert "[2][type=narrative]" in prompt
    assert "[7][type=narrative]" in prompt


def test_context_truncates_whole_entries():
    """超 12000 整条丢弃不切半条（04j §3.1）。"""
    big1 = "A" * 8000
    big2 = "B" * 8000
    llm = FakeLLM(payloads=[_ok_answer()])
    out = _run(_state([_ev(1, content=big1), _ev(2, content=big2)]), llm=llm)
    prompt = llm.calls[0]
    assert big1 in prompt
    assert big2 not in prompt
    assert out["answer"] != ""  # 截断不阻断


# ---------- prompt 铁律 ----------


def test_prompt_contains_discipline_rules():
    llm = FakeLLM(payloads=[_ok_answer()])
    _run(_state([_ev(1)]), llm=llm)
    prompt = llm.calls[0]
    assert "禁止心算" in prompt
    assert "34.2%[12]" in prompt  # 数字必带 [n] 格式样例
    assert "禁止编造引用" in prompt
    assert "unsupported" in prompt  # status≠ok 转述
    assert "insufficient_periods" in prompt
    assert "【思考】" in prompt
    assert "【答案】" in prompt


def test_prompt_uses_rewritten_query():
    llm = FakeLLM(payloads=[_ok_answer()])
    _run(
        _state([_ev(1)], query="原话", rewritten_query="贵州茅台 2024 年毛利率"),
        llm=llm,
    )
    assert "贵州茅台 2024 年毛利率" in llm.calls[0]


# ---------- 流式 token ----------


def test_token_streamed_per_chunk():
    llm = FakeLLM(payloads=[["【思考】", "34.2%[1]", "【答案】", "毛利率 34.2%[1]"]])
    w = _Writer()
    out = _run(_state([_ev(1)]), llm=llm, writer=w)
    tokens = [e for e in w.events if e["type"] == "token"]
    assert [t["content"] for t in tokens] == ["【思考】", "34.2%[1]", "【答案】", "毛利率 34.2%[1]"]
    assert out["answer"] == "【思考】34.2%[1]【答案】毛利率 34.2%[1]"


def test_answer_keeps_cot():
    """answer 保留完整 CoT 供 validate 全文对账（04j §1.2）。"""
    text = "【思考】证据 [1] 给出 34.2%。\n【答案】毛利率为 34.2%[1]"
    llm = FakeLLM(payloads=[text])
    out = _run(_state([_ev(1)]), llm=llm)
    assert out["answer"] == text
    assert "【思考】" in out["answer"]
    assert "【答案】" in out["answer"]


def test_no_done_event():
    """不发 done，终态归 validate（04j §3.5）。"""
    llm = FakeLLM(payloads=[_ok_answer()])
    w = _Writer()
    _run(_state([_ev(1)]), llm=llm, writer=w)
    assert all(e["type"] != "done" for e in w.events)


# ---------- citations 抽取 ----------


def test_citations_first_occurrence_order_dedup():
    """按 answer 首现序去重（04j §3.4）。"""
    text = "【答案】A[3] B[1] C[3]"
    llm = FakeLLM(payloads=[text])
    out = _run(_state([_ev(1), _ev(3)]), llm=llm)
    assert [c["ref"] for c in out["citations"]] == [3, 1]


def test_citations_drops_out_of_range_ref():
    """越界 [n] 丢弃 + WARN，不进 citations（04j §3.4）。"""
    text = "【答案】34.2%[1] 和 20.8%[99]"
    llm = FakeLLM(payloads=[text])
    out = _run(_state([_ev(1)]), llm=llm)
    assert [c["ref"] for c in out["citations"]] == [1]


def test_citation_fields_copied():
    text = "【答案】34.2%[2]"
    llm = FakeLLM(payloads=[text])
    out = _run(
        _state(
            [
                _ev(
                    2,
                    page_no=45,
                    document_id="docX",
                    snippet="毛利率片段",
                    type="table",
                )
            ]
        ),
        llm=llm,
    )
    [c] = out["citations"]
    assert c["ref"] == 2
    assert c["page_no"] == 45
    assert c["document_id"] == "docX"
    assert c["snippet"] == "毛利率片段"
    assert c["type"] == "table"


def test_citation_sse_pushed_in_order():
    llm = FakeLLM(payloads=["【答案】A[2] B[1]"])
    w = _Writer()
    _run(_state([_ev(1), _ev(2)]), llm=llm, writer=w)
    cites = [e for e in w.events if e["type"] == "citation"]
    assert [c["ref"] for c in cites] == [2, 1]
    assert cites[0]["page_no"] == 12


# ---------- 含数字零引用重试 ----------


def test_retry_when_digits_without_citations():
    """含数字却零引用 → 重试 1 次（04j §3.4）。"""
    llm = FakeLLM(
        payloads=[
            "【答案】毛利率为 34.2%",  # 有数字无引用
            "【答案】毛利率为 34.2%[1]",
        ]
    )
    out = _run(_state([_ev(1)]), llm=llm)
    assert len(llm.calls) == 2
    assert out["answer"] == "【答案】毛利率为 34.2%[1]"
    assert [c["ref"] for c in out["citations"]] == [1]


def test_retry_once_only_warn_and_fallthrough():
    """重试 1 次后仍无引用 → WARN 落盘，交 validate 兜底（04j §8）。"""
    llm = FakeLLM(
        payloads=[
            "【答案】毛利率为 34.2%",
            "【答案】毛利率为 34.2%",  # 重试仍无引用
        ]
    )
    out = _run(_state([_ev(1)]), llm=llm)
    assert len(llm.calls) == 2
    assert out["answer"] == "【答案】毛利率为 34.2%"
    assert out["citations"] == []
    assert out["error"] is None  # 不算失败


def test_no_retry_when_citations_present():
    llm = FakeLLM(payloads=["【答案】34.2%[1]"])
    _run(_state([_ev(1)]), llm=llm)
    assert len(llm.calls) == 1


def test_no_retry_when_no_digits():
    llm = FakeLLM(payloads=["【答案】该指标暂不支持。"])
    out = _run(_state([_ev(1, type="calc", status="unsupported")]), llm=llm)
    assert len(llm.calls) == 1
    assert out["citations"] == []


# ---------- LLM 失败重试 ----------


def test_llm_retry_recovers():
    llm = FakeLLM(errors=[TimeoutError("t1")], payloads=[_ok_answer()])
    out = _run(_state([_ev(1)]), llm=llm)
    assert len(llm.calls) == 2
    assert out["error"] is None
    assert out["answer"] == _ok_answer()


def test_llm_retry_exhausted_internal():
    llm = FakeLLM(errors=[TimeoutError("t1"), TimeoutError("t2")])
    out = _run(_state([_ev(1)]), llm=llm)
    assert out["error_code"] == "INTERNAL"
    assert out["error_stage"] == "generate"
    assert len(llm.calls) == 2  # 初次 + 重试 1 次


# ---------- 多轮历史 ----------


def test_history_recent_turns():
    history = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "u4"},
        {"role": "assistant", "content": "a4"},
        {"role": "user", "content": "u5"},
        {"role": "assistant", "content": "a5"},
    ]
    llm = FakeLLM(payloads=[_ok_answer()])
    _run(_state([_ev(1)], history=history), llm=llm)
    prompt = llm.calls[0]
    assert "u5" in prompt
    assert "a4" in prompt  # 近 4 条：u2/a2/u3/a3/u4/a4/u5/a5 的窗口起点附近
    assert "u1" not in prompt


def test_history_empty_placeholder():
    llm = FakeLLM(payloads=[_ok_answer()])
    _run(_state([_ev(1)], history=[]), llm=llm)
    assert "无历史对话" in llm.calls[0]


# ---------- Redis 回写 ----------


def test_redis_assistant_written():
    llm = FakeLLM(payloads=[_ok_answer()])
    mock_hist = MagicMock()
    with patch(
        "app.query_graph.nodes.generate.get_redis_chat_history",
        return_value=mock_hist,
    ):
        out = _run(
            _state([_ev(1)], rewritten_query="茅台 2024 毛利率"),
            llm=llm,
        )
    mock_hist.add_ai_message.assert_called_once()
    args = mock_hist.add_ai_message.call_args
    assert args[0][0] == out["answer"]
    assert args[0][1] == "茅台 2024 毛利率"


def test_redis_failure_warns_not_blocking():
    llm = FakeLLM(payloads=[_ok_answer()])
    with patch(
        "app.query_graph.nodes.generate.get_redis_chat_history",
        side_effect=RuntimeError("redis down"),
    ):
        out = _run(_state([_ev(1)]), llm=llm)
    assert out["error"] is None
    assert out["answer"] == _ok_answer()


# ---------- 边界 ----------


def test_pure_metric_evidence_in_context():
    """纯指标题：metric/calc 进 context，编号不丢（04j §9）。"""
    llm = FakeLLM(payloads=["【答案】营收为 20 亿[5]"])
    out = _run(
        _state(
            [
                _ev(5, type="metric", metric_code="revenue", content="营收 2024：20 亿", score=0.0),
                _ev(6, type="calc", metric_code="revenue_yoy", content="营收同比：+10%[计算]", score=0.0, status="ok"),
            ]
        ),
        llm=llm,
    )
    prompt = llm.calls[0]
    assert "[5][type=metric]" in prompt
    assert "[6][type=calc]" in prompt
    assert [c["ref"] for c in out["citations"]] == [5]


def test_status_unsupported_passthrough_in_context():
    """status≠ok 的 calc content 原样进 context（04h §3.3 已写进 content）。"""
    llm = FakeLLM(payloads=["【答案】该指标暂不支持[1]"])
    _run(
        _state([_ev(1, type="calc", status="unsupported", content="毛利率 CAGR：status=unsupported，暂不支持")]),
        llm=llm,
    )
    assert "暂不支持" in llm.calls[0]


def test_must_return_answer_and_citations():
    llm = FakeLLM(payloads=[_ok_answer()])
    out = _run(_state([_ev(1)]), llm=llm)
    assert "answer" in out
    assert "citations" in out
    assert isinstance(out["citations"], list)
