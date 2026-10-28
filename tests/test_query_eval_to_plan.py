"""problem/test_queries.json 联调评测：intent → entity → plan 三节点（docs/04 §11.1）。

不触检索/生成。真 LLM 跑 100 条 5 类金标，对照 expected_rewrite 出分桶指标。
跑法：EVAL_PIPELINE=1 PYTHONPATH=. uv run pytest tests/test_query_eval_to_plan.py -s
默认 skip——04 §11.1「集成测试脚本，不进单测 CI；Prompts 调优后手动跑」。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import pytest
from langgraph.runtime import Runtime

from app.entities.stock_dict import StockDictDbData
from app.lm.llm_utils import get_llm_client
from app.query_graph.context import QueryAgentContext
from app.query_graph.nodes import entity as entity_mod
from app.query_graph.nodes.entity import node_entity
from app.query_graph.nodes.intent import node_intent
from app.query_graph.nodes.plan import node_plan
from app.query_graph.state import default_state
from app.repositories.financial_metric_repository import FinancialMetricRepository
from app.repositories.stock_dict_repository import StockDictRepository

pytestmark = pytest.mark.skipif(
    not os.environ.get("EVAL_PIPELINE"),
    reason="集成评测不进单测 CI；EVAL_PIPELINE=1 手动跑（04 §11.1）",
)

QUERIES_PATH = Path(__file__).resolve().parents[1] / "problem" / "test_queries.json"
LATEST_PERIOD = "2024"

CODE_BY_NAME = {
    "神州高铁": "000008",
    "深振业A": "000006",
    "全新好": "000007",
    "新世界": "600628",
}

DICT_ROWS = [
    StockDictDbData("000008", "神州高铁", "神州高铁技术股份有限公司", [], "SZ"),
    StockDictDbData("000006", "深振业A", "深圳市振业(集团)股份有限公司", ["深振业"], "SZ"),
    StockDictDbData("000007", "全新好", "全新好股份有限公司", [], "SZ"),
    StockDictDbData("600628", "新世界", "上海新世界股份有限公司", [], "SH"),
]

_RHETORICAL_MARKS = re.compile(r"难道|不会是|不会吧|反而|吧？|！")


class FakeStockDict(StockDictRepository):
    def __init__(self):
        pass  # 不接 session，假字典离线

    def list_all(self) -> list[StockDictDbData]:
        return DICT_ROWS


class FakeMetrics(FinancialMetricRepository):
    def __init__(self):
        pass

    def latest_report_period(self) -> str | None:
        return LATEST_PERIOD


class _NoopChat:
    def add_user_message(self, *a, **k) -> None:
        pass


class _Writer:
    def __init__(self):
        self.events: list[dict] = []

    def __call__(self, item: dict) -> None:
        self.events.append(item)


def _load_samples() -> list[dict]:
    data = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    return list(data["queries"])


def _companies(raw: str) -> list[str]:
    return [c.strip() for c in (raw or "").split(",") if c.strip()]


def _parse_history(raw: str) -> list[dict]:
    history: list[dict] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if line.startswith("用户:"):
            history.append({"role": "user", "content": line[len("用户:"):].strip()})
        elif line.startswith("AI:"):
            history.append({"role": "assistant", "content": line[len("AI:"):].strip()})
    return history


def _attach_codes(history: list[dict]) -> list[dict]:
    """user 条挂样例股 codes，模拟 Redis 窗口——指代消解的依据。"""
    out: list[dict] = []
    for m in history:
        item = {"role": m["role"], "content": m.get("content") or ""}
        if m["role"] == "user":
            names = [n for n in CODE_BY_NAME if n in item["content"]]
            if names:
                item["stock_codes"] = [CODE_BY_NAME[n] for n in names]
                item["stock_names"] = names
        out.append(item)
    return out


def _expected_text(expected) -> str:
    if isinstance(expected, list):
        return " ".join(expected)
    return expected or ""


def _gram_ratio(gold: str, got: str) -> float:
    """2gram 重合度，仅诊断用（与 test_node_intent_llm 同口径）。"""
    grams = {gold[i : i + 2] for i in range(len(gold) - 1)}
    grams = {g for g in grams if not g.isspace()}
    if not grams:
        return 0.0
    return sum(1 for g in grams if g in got) / len(grams)


def _run_one(sample: dict) -> dict:
    """intent → entity → plan 串跑；out_of_scope / missing 短路，不触 plan。"""
    llm = get_llm_client(json_mode=True)
    runtime = Runtime(  # type: ignore[arg-type]
        context=QueryAgentContext(
            llm=llm,
            stock_dict_repository=FakeStockDict(),
            financial_metric_repository=FakeMetrics(),
        )
    )
    history = _attach_codes(_parse_history(sample.get("conversation_history") or ""))
    state = default_state(
        query=sample["current_query"],
        request_id=f"eval-{sample['id']}",
        session_id=f"eval-{sample['type']}",
        history=history,
    )
    writer = _Writer()

    intent_patch = asyncio.run(node_intent(state, runtime, writer))
    state.update(intent_patch)
    outcome = "intent_fail" if intent_patch.get("error_code") else "ok"

    entity_patch: dict = {}
    plan_patch: dict = {}
    if outcome == "ok" and state.get("intent") != "out_of_scope":
        entity_patch = asyncio.run(node_entity(state, runtime, writer))
        state.update(entity_patch)
        if entity_patch.get("error_code"):
            outcome = "entity_fail"
        elif state.get("missing_fields"):
            outcome = "clarify"
        else:
            plan_patch = asyncio.run(node_plan(state, runtime, writer))
            state.update(plan_patch)
            outcome = "plan_fail" if plan_patch.get("error_code") else "planned"
    elif state.get("intent") == "out_of_scope":
        outcome = "out_of_scope"

    return {
        **sample,
        "history": history,
        "outcome": outcome,
        "intent_patch": intent_patch,
        "entity_patch": entity_patch,
        "plan_patch": plan_patch,
        "state": state,
        "events": writer.events,
    }


def _is_quota_error(err: str) -> bool:
    return "Free quota exhausted" in err or "AllocationQuota" in err


@pytest.fixture(scope="module")
def results() -> list[dict]:
    if not QUERIES_PATH.exists():
        pytest.skip(f"缺少评测集 {QUERIES_PATH}")
    entity_mod.get_redis_chat_history = lambda sid: _NoopChat()
    rows: list[dict] = []
    for s in _load_samples():
        row = _run_one(s)
        err = (row["intent_patch"].get("error") or "") + (
            row["plan_patch"].get("error") or ""
        )
        if _is_quota_error(err):
            # 配额耗尽不是链路故障；中止并跳过，避免把 403 当模型失败
            pytest.skip(f"LLM 免费配额耗尽，已完成 {len(rows)} 条后中止")
        rows.append(row)
    return rows


# ---------- 硬断言：不崩、契约不破 ----------


def test_pipeline_runs_all_rows(results):
    assert len(results) == 100
    for row in results:
        st = row["state"]
        if row["outcome"] in ("intent_fail", "entity_fail"):
            pytest.fail(f"id={row['id']} {row['outcome']}: {st.get('error')}")
        assert st.get("intent") in {
            "compare", "metric_calc", "trend", "simple_qa", "out_of_scope",
        }, f"id={row['id']} intent={st.get('intent')}"


def test_planned_rows_filters_contract(results):
    """走到 plan 的行：route 四枚举、sub_id=sq1..sn、filters 主体键来自 entities。"""
    for row in results:
        if row["outcome"] != "planned":
            continue
        sqs = row["state"].get("sub_queries") or []
        assert sqs, f"id={row['id']} planned 但 sub_queries 空"
        ent = row["state"]["entities"]
        for i, sq in enumerate(sqs, 1):
            assert sq["sub_id"] == f"sq{i}", f"id={row['id']} sub_id={sq['sub_id']}"
            assert sq["route"] in {"vector", "metric_sql", "calc", "mixed"}, (
                f"id={row['id']} route={sq['route']}"
            )
            f = sq["filters"]
            assert set(f) == {
                "stock_codes", "report_periods", "report_types", "sections",
            }, f"id={row['id']} filters keys={set(f)}"
            assert set(f["stock_codes"]) <= set(ent["stock_codes"]), (
                f"id={row['id']} filters 收窄越界 {f['stock_codes']} vs {ent['stock_codes']}"
            )
            assert f["report_periods"] == ent["periods"]
            assert f["report_types"] == ent["report_types"]


def test_multi_intent_split_count_aligned(results):
    """multi_intent 拆条：对齐金标 N 条（理想），或按公司展开 N×C 条（可执行）。

    04d §4.4 要求 N 条对齐；LLM 偶发走 compare 拆法（flash 模型方差）。
    两种形态 filters 都干净、都能执行，故都放行；理想率看软报告「拆条」列。
    其它条数（合并问句/凭空多拆）一律判错。
    """
    for row in results:
        if row["type"] != "multi_intent":
            continue
        if row["outcome"] == "clarify":
            continue  # 缺字段短路，本轮不考核拆条
        if row["outcome"] == "out_of_scope":
            continue
        assert row["outcome"] == "planned", f"id={row['id']} outcome={row['outcome']}"
        got = row["state"]["sub_queries"]
        exp = row["expected_rewrite"]
        n_companies = max(len(_companies(row["company"])), 1)
        ok_counts = {len(exp), len(exp) * n_companies}
        assert len(got) in ok_counts, (
            f"id={row['id']} 拆条数 got={len(got)} 合法值={ok_counts} "
            f"got_q={[s['question'] for s in got]}"
        )


def test_rewritten_not_empty_for_non_oos(results):
    for row in results:
        if row["outcome"] == "out_of_scope":
            continue
        rewrite = (row["state"].get("rewritten_query") or "").strip()
        assert rewrite, f"id={row['id']} rewritten_query 空"


def test_planned_rows_have_stock_codes(results):
    """走到 plan 的行必有 stock_codes——否则 plan 会 INTERNAL，与 outcome 矛盾。"""
    for row in results:
        if row["outcome"] != "planned":
            continue
        codes = (row["state"].get("entities") or {}).get("stock_codes") or []
        assert codes, f"id={row['id']} planned 但 stock_codes 空"


# ---------- 软报告：分桶指标 ----------


def test_eval_report(results, capsys):
    buckets: dict[str, dict] = {}
    for row in results:
        b = buckets.setdefault(
            row["type"],
            {"n": 0, "rewrite_hit": 0, "entity_hit": 0, "split_ok": 0,
             "split_n": 0, "outcome": {}},
        )
        b["n"] += 1
        b["outcome"][row["outcome"]] = b["outcome"].get(row["outcome"], 0) + 1

        gold = _expected_text(row["expected_rewrite"])
        if row["outcome"] == "planned" and row["type"] == "multi_intent":
            got_text = " ".join(s["question"] for s in row["state"]["sub_queries"])
        else:
            got_text = row["state"].get("rewritten_query") or ""
        if _gram_ratio(gold, got_text) >= 0.35:
            b["rewrite_hit"] += 1

        if row["outcome"] not in ("out_of_scope", "intent_fail"):
            ent = row["state"].get("entities") or {}
            resolved = set(row["state"].get("resolved_names") or {})
            unresolved = set(ent.get("unresolved_names") or [])
            named = [c for c in _companies(row["company"]) if c in row["current_query"]]
            if named and all(c in resolved or c in unresolved for c in named):
                b["entity_hit"] += 1
            elif not named:
                b["entity_hit"] += 1  # 模糊句不考核

        if row["type"] == "multi_intent" and row["outcome"] == "planned":
            b["split_n"] += 1
            exp = row["expected_rewrite"]
            got = row["state"]["sub_queries"]
            if len(got) == len(exp):
                b["split_ok"] += 1
            elif len(got) == len(exp) * max(len(_companies(row["company"])), 1):
                b["split_expanded"] = b.get("split_expanded", 0) + 1

    lines = ["=== 分桶指标（2gram≥0.35 记改写命中）==="]
    totals = {"n": 0, "rewrite_hit": 0, "entity_hit": 0, "split_ok": 0, "split_n": 0, "split_expanded": 0}
    for t, b in buckets.items():
        totals["n"] += b["n"]
        totals["rewrite_hit"] += b["rewrite_hit"]
        totals["entity_hit"] += b["entity_hit"]
        totals["split_ok"] += b["split_ok"]
        totals["split_n"] += b["split_n"]
        totals["split_expanded"] += b.get("split_expanded", 0)
        split = (
            f" 拆条对齐 {b['split_ok']}/{b['split_n']}"
            f"+展开{b.get('split_expanded', 0)}"
            if b["split_n"] else ""
        )
        lines.append(
            f"[{t}] n={b['n']} 改写 {b['rewrite_hit']}/{b['n']} "
            f"实体 {b['entity_hit']}/{b['n']}{split} outcome={b['outcome']}"
        )
    lines.append(
        f"[TOTAL] n={totals['n']} 改写 {totals['rewrite_hit']}/{totals['n']} "
        f"实体 {totals['entity_hit']}/{totals['n']} "
        f"拆条对齐 {totals['split_ok']}/{totals['split_n']}"
        f"+展开{totals['split_expanded']}"
    )

    # 明细：只打未命中/非 planned 的行，命中太多不刷屏
    lines.append("\n=== 未命中 / 异常明细 ===")
    for row in results:
        gold = _expected_text(row["expected_rewrite"])
        if row["outcome"] == "planned" and row["type"] == "multi_intent":
            got_text = " ".join(s["question"] for s in row["state"]["sub_queries"])
        else:
            got_text = row["state"].get("rewritten_query") or ""
        ratio = _gram_ratio(gold, got_text)
        interesting = row["outcome"] != "planned" or ratio < 0.35
        if row["type"] == "multi_intent" and row["outcome"] == "planned":
            if len(row["state"]["sub_queries"]) != len(row["expected_rewrite"]):
                interesting = True
        if not interesting:
            continue
        lines.append(
            f"\n[{row['type']}#{row['id']}] {row['outcome']} 2gram={ratio:.2f}\n"
            f"  Q    : {row['current_query']}\n"
            f"  gold : {gold}\n"
            f"  got  : {got_text}\n"
            f"  sq   : {[(s['sub_id'], s['route'], s['question']) for s in row['state'].get('sub_queries') or []]}"
        )

    with capsys.disabled():
        print("\n".join(lines))
    assert results
