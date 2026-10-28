"""node_entity 匹配评测（docs/04a §4/§6）。

从 problem/test_queries.json 每类抽 5 条共 25 条。entity 不调 LLM，
intent_raw 用金标构造（问句里出现的公司名 + 期间词），历史 user 条挂
stock_codes 模拟 Redis 窗口；假字典只放 4 家样例股。

判据：问句有公司名 → 必出对码；无名有历史 → 沿用历史码；无名无史 →
缺 stock_codes 追问。periods 相对词绝对化。
跑法：PYTHONPATH=. uv run pytest tests/test_node_entity_eval.py -s
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest
from langgraph.runtime import Runtime

from app.entities.stock_dict import StockDictDbData
from app.query_graph.context import QueryAgentContext
from app.query_graph.nodes import entity as entity_mod
from app.query_graph.nodes.entity import node_entity
from app.query_graph.state import default_state
from app.repositories.financial_metric_repository import FinancialMetricRepository
from app.repositories.stock_dict_repository import StockDictRepository

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

_PERIOD_RE = re.compile(r"近[一二三四五六七八九十0-9]+年|\d{4}年?|去年|前年|今年")


class FakeStockDict(StockDictRepository):
    def __init__(self, rows: list[StockDictDbData]):
        self.rows = rows

    def list_all(self) -> list[StockDictDbData]:
        return self.rows


class FakeMetrics(FinancialMetricRepository):
    def __init__(self, latest: str | None = LATEST_PERIOD):
        self.latest = latest

    def latest_report_period(self) -> str | None:
        return self.latest


class FakeChatHistory:
    def add_user_message(self, content, stock_codes=None, periods=None) -> None:
        pass


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
    """user 条里出现的样例股挂上 codes，模拟 Redis 窗口（04a §6）。"""
    out: list[dict] = []
    for m in history:
        content = m.get("content") or ""
        item: dict = {"role": m["role"], "content": content}
        if m["role"] == "user":
            names = [n for n in CODE_BY_NAME if n in content]
            if names:
                item["stock_codes"] = [CODE_BY_NAME[n] for n in names]
                item["stock_names"] = names
        out.append(item)
    return out


def _build_intent_raw(sample: dict) -> dict:
    """金标构造 LLM 抽取结果：只放问句里出现的公司名，指代句交给 coref。"""
    query = sample["current_query"]
    names = [c for c in _companies(sample["company"]) if c in query]
    return {
        "intent": "metric_calc",
        "stock_names": names,
        "periods": _PERIOD_RE.findall(query),
        "metrics": [],
        "rewritten_query": query,
    }


def _expected_codes(sample: dict, history: list[dict]) -> list[str]:
    """问句有名用问句名；无名有历史沿用历史码；无名无史视为模糊（金标假设宇宙）。"""
    query = sample["current_query"]
    companies = _companies(sample["company"])
    names_in_q = [c for c in companies if c in query]
    if names_in_q:
        return [CODE_BY_NAME[c] for c in names_in_q]
    hist_codes: list[str] = []
    for m in history:
        hist_codes.extend(m.get("stock_codes") or [])
    return list(dict.fromkeys(hist_codes))


def _run_one(sample: dict) -> dict:
    runtime = Runtime(  # type: ignore[arg-type]
        context=QueryAgentContext(
            stock_dict_repository=FakeStockDict(DICT_ROWS),
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
    state["intent_raw"] = _build_intent_raw(sample)
    patch = asyncio.run(node_entity(state, runtime, None))
    return {**sample, "history": history, "expected_codes": _expected_codes(sample, history), "patch": patch}


class _NoopChat:
    def add_user_message(self, *a, **k) -> None:
        pass


@pytest.fixture(scope="module")
def results() -> list[dict]:
    if not QUERIES_PATH.exists():
        pytest.skip(f"缺少评测集 {QUERIES_PATH}")
    # module 级夹具用不了 function 级 monkeypatch，直接换掉（entity 只 WARN）
    entity_mod.get_redis_chat_history = lambda sid: _NoopChat()
    return [_run_one(s) for s in _load_samples()]


def test_eval_all_rows_resolve_codes(results):
    """硬断言：不崩溃、有名出对码、无名无史走追问、期间绝对化。"""
    assert len(results) == 100
    for row in results:
        patch = row["patch"]
        assert patch.get("error") is None, f"id={row['id']} {patch.get('error')}"
        ent = patch["entities"]
        expected = row["expected_codes"]
        got = ent["stock_codes"]

        if expected:
            assert set(got) == set(expected), (
                f"id={row['id']} codes got={got} expected={expected} "
                f"resolved={patch.get('resolved_names')}"
            )
            assert not ent["unresolved_names"], (
                f"id={row['id']} 不应有 unresolved={ent['unresolved_names']}"
            )
        else:
            # 模糊指代且无历史：必须追问而不是瞎猜
            assert got == [], f"id={row['id']} 无依据却给出 codes={got}"
            assert "stock_codes" in patch["missing_fields"], (
                f"id={row['id']} 缺 codes 却未进 missing={patch['missing_fields']}"
            )

        if "2024年" in row["current_query"]:
            assert "2024" in ent["periods"], (
                f"id={row['id']} periods={ent['periods']} 未吸收 2024年"
            )


def test_eval_report(results, capsys):
    """软报告：对照金标打印匹配明细，看各类型命中情况。"""
    lines = []
    hit = 0
    for row in results:
        patch = row["patch"]
        ent = patch["entities"]
        expected = row["expected_codes"]
        got = ent["stock_codes"]
        ok = set(got) == set(expected)
        if ok:
            hit += 1
        lines.append(
            f"\n[{row['type']}#{row['id']}] {'OK' if ok else 'MISS'}\n"
            f"  Q      : {row['current_query']}\n"
            f"  gold   : company={row['company']} -> {expected}\n"
            f"  got    : codes={got} names={ent['stock_names']} "
            f"resolved={patch.get('resolved_names')}\n"
            f"  extra  : unresolved={ent['unresolved_names']} "
            f"periods={ent['periods']} missing={patch.get('missing_fields')}"
        )
    lines.append(f"\n=== codes 精确命中 {hit}/{len(results)} ===")
    with capsys.disabled():
        print("\n".join(lines))
    assert results
