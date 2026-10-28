"""node_plan 测试（docs/04d §9）。

假 LLM 注入 QueryAgentContext；断言拆条落盘、route 校验、filters 继承、
sub_id 重编、截尾、重试与 SSE 形状。不打真模型。
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from langgraph.runtime import Runtime

from app.query_graph.context import QueryAgentContext
from app.query_graph.nodes.plan import node_plan
from app.query_graph.state import default_state


class FakeLLM:
    """按序返回 content；errors 优先抛出，模拟超时/5xx。"""

    def __init__(self, payloads=None, errors=None):
        self.calls: list[str] = []
        self.payloads = list(payloads or [])
        self.errors = list(errors or [])

    def invoke(self, prompt):
        self.calls.append(prompt)
        if self.errors:
            raise self.errors.pop(0)
        payload = self.payloads.pop(0) if self.payloads else {}
        content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return SimpleNamespace(content=content)


class _Writer:
    def __init__(self):
        self.events: list[dict] = []

    def __call__(self, item: dict) -> None:
        self.events.append(item)


def _runtime(llm) -> Runtime:
    return Runtime(context=QueryAgentContext(llm=llm))  # type: ignore[arg-type]


def _state(
    intent="metric_calc",
    rewritten="神州高铁2024年营业收入是多少？",
    entities=None,
    resolved_names=None,
    **kw,
):
    s = default_state(query="q", request_id="q1", session_id="s1")
    s["intent"] = intent
    s["rewritten_query"] = rewritten
    ent = {
        "stock_names": ["神州高铁"],
        "stock_codes": ["000008"],
        "unresolved_names": [],
        "periods": ["2024"],
        "report_types": ["annual"],
        "metrics": ["营业收入"],
    }
    if entities:
        ent.update(entities)
    s["entities"] = ent
    s["resolved_names"] = resolved_names if resolved_names is not None else {"神州高铁": "000008"}
    s.update(kw)
    return s


def _item(question="神州高铁2024年营业收入是多少？", route="metric_sql", **over):
    p = {"question": question, "route": route, "sections": []}
    p.update(over)
    return p


def _run(llm, state=None, w=None):
    writer = w if w is not None else _Writer()
    patch = asyncio.run(node_plan(state or _state(), _runtime(llm), writer))
    return patch, writer.events


# ---------- 成功路径 ----------


def test_success_single_subquery_writes_state_and_sse():
    llm = FakeLLM([[_item()]])
    patch, events = _run(llm)

    assert patch["error"] is None
    sqs = patch["sub_queries"]
    assert len(sqs) == 1
    assert sqs[0]["sub_id"] == "sq1"
    assert sqs[0]["question"] == "神州高铁2024年营业收入是多少？"
    assert sqs[0]["route"] == "metric_sql"
    assert sqs[0]["hyde_used"] is False
    assert events == [
        {
            "type": "sub_query",
            "sub_id": "sq1",
            "question": "神州高铁2024年营业收入是多少？",
            "status": "running",
        }
    ]


def test_multi_intent_questions_preserved_in_order():
    """multi_intent 拆条按 LLM 顺序落盘，sub_id 重编 sq1..sn。"""
    qs = [
        "神州高铁2024年营业收入是多少？",
        "神州高铁2024年归母净利润是多少？",
        "神州高铁2024年的利润分配方案是什么？",
    ]
    llm = FakeLLM([[
        _item(qs[0], "metric_sql"),
        _item(qs[1], "metric_sql"),
        _item(qs[2], "vector"),
    ]])
    patch, _ = _run(llm, _state(intent="multi_intent"))
    assert [s["question"] for s in patch["sub_queries"]] == qs
    assert [s["sub_id"] for s in patch["sub_queries"]] == ["sq1", "sq2", "sq3"]


def test_sub_id_renumbered_discards_llm_ids():
    llm = FakeLLM([[
        _item("问1", "metric_sql", sub_id="zzz"),
        _item("问2", "vector", sub_id="a1"),
    ]])
    patch, events = _run(llm)
    assert [s["sub_id"] for s in patch["sub_queries"]] == ["sq1", "sq2"]
    assert [e["sub_id"] for e in events] == ["sq1", "sq2"]


# ---------- filters 继承 ----------


def test_filters_inherit_entities_subject_keys():
    llm = FakeLLM([[_item()]])
    patch, _ = _run(
        llm,
        _state(
            entities={
                "stock_codes": ["000008"],
                "periods": ["2023", "2024"],
                "report_types": ["annual"],
            },
        ),
    )
    f = patch["sub_queries"][0]["filters"]
    assert f["stock_codes"] == ["000008"]
    assert f["report_periods"] == ["2023", "2024"]
    assert f["report_types"] == ["annual"]
    assert f["sections"] == []


def test_filters_ignore_llm_subject_keys():
    """LLM 偷写 filters 主体键必须丢弃——猜码必错，一律继承 entities。"""
    llm = FakeLLM([[
        _item(filters={"stock_codes": ["999999"], "report_periods": ["1999"]}),
    ]])
    patch, _ = _run(llm)
    f = patch["sub_queries"][0]["filters"]
    assert f["stock_codes"] == ["000008"]
    assert f["report_periods"] == ["2024"]
    assert "1999" not in f["report_periods"]


def test_filters_sections_from_llm():
    llm = FakeLLM([[_item(route="vector", sections=["management_discussion"])]])
    patch, _ = _run(llm)
    assert patch["sub_queries"][0]["filters"]["sections"] == ["management_discussion"]


def test_filters_sections_bad_type_defaults_empty():
    llm = FakeLLM([[_item(sections="risk")]])
    patch, _ = _run(llm)
    assert patch["sub_queries"][0]["filters"]["sections"] == []


# ---------- compare 收窄 ----------


def test_compare_narrows_stock_codes_per_company():
    llm = FakeLLM([[
        _item("神州高铁2024年营业收入是多少？", "metric_sql"),
        _item("深振业A 2024年营业收入是多少？", "metric_sql"),
    ]])
    patch, _ = _run(
        llm,
        _state(
            intent="compare",
            rewritten="神州高铁和深振业A 2024年哪家营收更高？",
            entities={
                "stock_names": ["神州高铁", "深振业A"],
                "stock_codes": ["000008", "000006"],
            },
            resolved_names={"神州高铁": "000008", "深振业A": "000006"},
        ),
    )
    assert patch["sub_queries"][0]["filters"]["stock_codes"] == ["000008"]
    assert patch["sub_queries"][1]["filters"]["stock_codes"] == ["000006"]


def test_narrow_skipped_when_question_matches_both_companies():
    llm = FakeLLM([[_item("神州高铁和深振业A 2024年营业收入对比？", "metric_sql")]])
    patch, _ = _run(
        llm,
        _state(
            intent="compare",
            entities={
                "stock_names": ["神州高铁", "深振业A"],
                "stock_codes": ["000008", "000006"],
            },
            resolved_names={"神州高铁": "000008", "深振业A": "000006"},
        ),
    )
    assert patch["sub_queries"][0]["filters"]["stock_codes"] == ["000008", "000006"]


def test_single_code_never_narrows():
    llm = FakeLLM([[_item("神州高铁2024年营业收入是多少？")]])
    patch, _ = _run(llm)
    assert patch["sub_queries"][0]["filters"]["stock_codes"] == ["000008"]


# ---------- route 校验 / 重试 ----------


def test_invalid_route_retries_then_internal():
    llm = FakeLLM([
        [_item(route="sql")],
        [_item(route="查数")],
    ])
    patch, _ = _run(llm)
    assert len(llm.calls) == 2
    assert patch["error_code"] == "INTERNAL"
    assert patch["error_stage"] == "plan"


def test_invalid_route_retry_recovers():
    llm = FakeLLM([
        [_item(route="sql")],
        [_item(route="vector")],
    ])
    patch, _ = _run(llm)
    assert len(llm.calls) == 2
    assert patch["sub_queries"][0]["route"] == "vector"
    assert patch["error"] is None


def test_non_json_retries_then_internal():
    llm = FakeLLM(["not json", "[]oops"])
    patch, _ = _run(llm)
    assert len(llm.calls) == 2
    assert patch["error_code"] == "INTERNAL"
    assert patch["error_stage"] == "plan"


def test_empty_list_retries_then_internal():
    llm = FakeLLM([[], []])
    patch, _ = _run(llm)
    assert len(llm.calls) == 2
    assert patch["error_code"] == "INTERNAL"
    assert patch["error_stage"] == "plan"


def test_non_list_retries_then_internal():
    llm = FakeLLM([{"question": "x"}, {"question": "y"}])
    patch, _ = _run(llm)
    assert len(llm.calls) == 2
    assert patch["error_code"] == "INTERNAL"


def test_timeout_retries_then_internal():
    llm = FakeLLM(errors=[TimeoutError("t1"), TimeoutError("t2")])
    patch, _ = _run(llm)
    assert len(llm.calls) == 2
    assert patch["error_code"] == "INTERNAL"
    assert patch["error_stage"] == "plan"


def test_retry_recovers_on_second_call_after_timeout():
    llm = FakeLLM(payloads=[[_item(route="calc")]], errors=[TimeoutError("t1")])
    patch, _ = _run(llm)
    assert len(llm.calls) == 2
    assert patch["sub_queries"][0]["route"] == "calc"


# ---------- 单条丢弃 / 截尾 ----------


def test_empty_question_dropped_others_kept():
    llm = FakeLLM([[
        _item("有效问句", "metric_sql"),
        _item("", "vector"),
        _item("   ", "calc"),
    ]])
    patch, _ = _run(llm)
    assert [s["question"] for s in patch["sub_queries"]] == ["有效问句"]
    assert patch["sub_queries"][0]["sub_id"] == "sq1"


def test_all_questions_empty_internal():
    llm = FakeLLM([[_item(""), _item("  ")]])
    patch, _ = _run(llm)
    assert patch["error_code"] == "INTERNAL"
    assert patch["error_stage"] == "plan"


def test_truncate_over_max_sub_queries():
    items = [_item(f"问句{i}", "metric_sql") for i in range(1, 12)]
    llm = FakeLLM([items])
    patch, _ = _run(llm)
    assert len(patch["sub_queries"]) == 8
    assert patch["sub_queries"][-1]["sub_id"] == "sq8"
    assert patch["sub_queries"][-1]["question"] == "问句8"


# ---------- 边界 / prompt ----------


def test_empty_stock_codes_internal():
    llm = FakeLLM([[_item()]])
    patch, _ = _run(llm, _state(entities={"stock_codes": []}))
    assert patch["error_code"] == "INTERNAL"
    assert patch["error_stage"] == "plan"
    assert llm.calls == []  # 不白调 LLM


def test_llm_missing_returns_internal():
    async def _go():
        return await node_plan(_state(), Runtime(context=QueryAgentContext()), _Writer())

    patch = asyncio.run(_go())
    assert patch["error_code"] == "INTERNAL"
    assert patch["error_stage"] == "plan"


def test_prompt_includes_rewrite_entities_intent():
    llm = FakeLLM([[_item()]])
    _run(
        llm,
        _state(
            intent="compare",
            rewritten="神州高铁和深振业A 2024年哪家营收更高？",
            entities={
                "stock_names": ["神州高铁", "深振业A"],
                "stock_codes": ["000008", "000006"],
                "periods": ["2024"],
                "metrics": ["营业收入"],
            },
            resolved_names={"神州高铁": "000008", "深振业A": "000006"},
        ),
    )
    prompt = llm.calls[0]
    assert "神州高铁和深振业A 2024年哪家营收更高？" in prompt
    assert "compare" in prompt
    assert "000008" in prompt
    assert "000006" in prompt
    assert "2024" in prompt
    assert "营业收入" in prompt


def test_prompt_omits_unresolved_names():
    llm = FakeLLM([[_item()]])
    _run(
        llm,
        _state(entities={"unresolved_names": ["神州"], "stock_names": ["神州高铁"]}),
    )
    prompt = llm.calls[0]
    assert "unresolved" not in prompt.lower()


def test_prompt_simple_qa_rule_scoped_to_single_question():
    """simple_qa 禁止写死「1 条」——多问句要按句拆，否则与独立疑问数总规则打架。"""
    llm = FakeLLM([[_item()]])
    _run(llm, _state(intent="simple_qa"))
    prompt = llm.calls[0]
    assert "simple_qa" in prompt
    assert "simple_qa：1 条" not in prompt
    assert "单疑问" in prompt
    # 多问句按句拆是结构规则，不挂在 multi_intent 意图下（intent 枚举无此值）
    assert "多个独立疑问" in prompt
