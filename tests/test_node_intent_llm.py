"""node_intent 真实 LLM 抽样评测（docs/04a §9 / 04 §11.1）。

每类取 problem/test_queries.json 2 条，共 10 条；对照 expected_rewrite 看改写。
跑法：PYTHONPATH=. uv run pytest tests/test_node_intent_llm.py -s
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest
from langgraph.runtime import Runtime

from app.lm.llm_utils import get_llm_client
from app.query_graph.context import QueryAgentContext
from app.query_graph.nodes.intent import node_intent
from app.query_graph.state import default_state

QUERIES_PATH = Path(__file__).resolve().parents[1] / "problem" / "test_queries.json"
SAMPLES_PER_TYPE = 5


def _load_samples() -> list[dict]:
    data = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    by_type: dict[str, list[dict]] = {}
    for q in data["queries"]:
        by_type.setdefault(q["type"], []).append(q)
    samples: list[dict] = []
    for items in by_type.values():
        samples.extend(items[:SAMPLES_PER_TYPE])
    return samples


def _parse_history(raw: str) -> list[dict]:
    """'用户: xxx\\nAI: yyy' → [{role, content}]；空串返 []。"""
    history: list[dict] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if line.startswith("用户:"):
            history.append({"role": "user", "content": line[len("用户:"):].strip()})
        elif line.startswith("AI:"):
            history.append({"role": "assistant", "content": line[len("AI:"):].strip()})
    return history


def _companies(raw: str) -> list[str]:
    return [c.strip() for c in (raw or "").split(",") if c.strip()]


def _expected_text(expected) -> str:
    if isinstance(expected, list):
        return " ".join(expected)
    return expected or ""


def _run_one(sample: dict) -> dict:
    llm = get_llm_client(json_mode=True)
    runtime = Runtime(context=QueryAgentContext(llm=llm))  # type: ignore[arg-type]
    state = default_state(
        query=sample["current_query"],
        request_id=f"eval-{sample['id']}",
        session_id=f"eval-{sample['type']}",
        history=_parse_history(sample.get("conversation_history") or ""),
    )
    patch = asyncio.run(node_intent(state, runtime, None))
    return {**sample, "patch": patch}


@pytest.fixture(scope="module")
def results() -> list[dict]:
    if not QUERIES_PATH.exists():
        pytest.skip(f"缺少评测集 {QUERIES_PATH}")
    return [_run_one(s) for s in _load_samples()]


def test_llm_eval_all_rows_produce_rewrite(results):
    """硬断言：不崩溃、intent 合法、改写非空、主体名进产物。"""
    assert len(results) == 25
    for row in results:
        patch = row["patch"]
        assert patch.get("error") is None, f"id={row['id']} {patch.get('error')}"
        assert patch["intent"] in {
            "compare", "metric_calc", "trend", "simple_qa", "out_of_scope",
        }
        rewrite = patch["rewritten_query"]
        assert rewrite.strip(), f"id={row['id']} 改写为空"

        expected = _expected_text(row["expected_rewrite"])
        companies = _companies(row["company"])
        names = patch["intent_raw"]["stock_names"]
        blob = rewrite + " ".join(names)
        # 历史为空且问句用「四家/两家/这」等模糊指代时，金标默认了样例股宇宙，
        # 模型无法凭空解析——跳过主体断言，只保留改写非空
        vague = not row.get("conversation_history") and not any(
            c in row["current_query"] for c in companies
        )
        if companies and not vague:
            assert any(c in blob for c in companies), (
                f"id={row['id']} 主体缺失 company={companies} names={names} rewrite={rewrite}"
            )
        assert expected, f"id={row['id']} 金标为空"


def test_llm_eval_report(results, capsys):
    """软报告：对照金标打印改写，看语义是否贴合（不断言相似度）。"""
    lines = []
    ok = 0
    for row in results:
        patch = row["patch"]
        expected = row["expected_rewrite"]
        rewrite = patch["rewritten_query"]
        exp_text = _expected_text(expected)
        # 粗粒度重合：金标中的 2-gram 有多少落进改写（仅诊断，不判生死）
        grams = {exp_text[i : i + 2] for i in range(len(exp_text) - 1)}
        grams = {g for g in grams if not g.isspace()}
        if grams:
            hit = sum(1 for g in grams if g in rewrite)
            ratio = hit / len(grams)
        else:
            ratio = 0.0
        if ratio >= 0.35:
            ok += 1
        lines.append(
            f"\n[{row['type']}#{row['id']}] intent={patch['intent']} "
            f"conf={patch['intent_raw'].get('confidence')} 2gram={ratio:.2f}\n"
            f"  Q    : {row['current_query']}\n"
            f"  gold : {expected}\n"
            f"  got  : {rewrite}\n"
            f"  names: {patch['intent_raw']['stock_names']}"
        )
    lines.append(f"\n=== 2gram≥0.35 命中 {ok}/{len(results)} ===")
    with capsys.disabled():
        print("\n".join(lines))
    # 报告型用例始终过，详情看 stdout
    assert results
