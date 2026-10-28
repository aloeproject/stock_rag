"""node_table_extract 测试：规则过滤后 LLM、落库、失败计数、幂等（docs/02d §5/§6）。"""
import asyncio
import threading
import time

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.import_graph.nodes.table_extract import node_table_extract
from app.models.base import Base
from app.repositories.report_table_repository import ReportTableRepository

HEADLINE_CAP = "六、主要会计数据和财务指标"
HEADLINE_HTML = (
    "<table><tr><td>项目</td><td>2024年</td><td>2023年</td></tr>"
    "<tr><td>营业收入</td><td>100</td><td>90</td></tr>"
    "<tr><td>归属于上市公司股东的净利润</td><td>10</td><td>9</td></tr>"
    "<tr><td>总资产</td><td>500</td><td>450</td></tr></table>"
)
NOTES_CAP = "十一、资产构成情况"
NOTES_HTML = (
    "<table><tr><td>项目</td><td>期末余额</td><td>期初余额</td></tr>"
    "<tr><td>其他货币资金</td><td>3</td><td>4</td></tr>"
    "<tr><td>合计</td><td>3</td><td>4</td></tr></table>"
)
BALANCE_CAP = "（除特别注明外，金额单位均为人民币元）"
BALANCE_HTML = (
    "<table><tr><td>项目</td><td>本年年末数</td><td>上年年末数</td></tr>"
    "<tr><td>货币资金</td><td>100</td><td>90</td></tr>"
    "<tr><td>资产总计</td><td>500</td><td>450</td></tr>"
    "<tr><td>负债合计</td><td>300</td><td>280</td></tr></table>"
)


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    yield s
    s.close()


@pytest.fixture()
def ctx_runtime(runtime, session):
    runtime.context.report_table_repository = ReportTableRepository(session)
    return runtime


def _write_md(tmp_path, sections: list[tuple[str, str]]) -> str:
    parts = []
    for cap, html in sections:
        parts.append(f"# {cap}\n{html}\n")
    p = tmp_path / "content.md"
    p.write_text("\n".join(parts), encoding="utf-8")
    return str(p)


def _state(md_path: str) -> dict:
    return {
        "document_id": "doc1",
        "markdown_path": md_path,
        "table_paths": [],
        "pages_map_path": "",
        "stock_code": "000006",
        "stock_name": "深振业A",
        "report_type": "annual",
        "report_period": "2024",
        "tables_total": 0,
        "tables_ok": 0,
        "tables_skipped": 0,
        "tables_failed": 0,
    }


def _fake_llm(items=None, scope="consolidated", fail_times=0):
    calls = {"n": 0}

    def fn(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise RuntimeError("llm boom")
        return {"items": items if items is not None else [{"metric_name": "营业收入"}], "statement_scope": scope}

    fn.calls = calls
    return fn


def test_skips_non_core_without_llm(tmp_path, ctx_runtime, monkeypatch):
    fake = _fake_llm()
    monkeypatch.setattr("app.import_graph.nodes.table_extract.llm_extract_items", fake)
    md = _write_md(tmp_path, [(NOTES_CAP, NOTES_HTML)])
    patch = asyncio.run(node_table_extract(_state(md), ctx_runtime))
    assert fake.calls["n"] == 0
    assert patch["tables_total"] == 1
    assert patch["tables_skipped"] == 1
    assert patch["tables_ok"] == 0
    assert patch["tables_failed"] == 0

    rows = ctx_runtime.context.report_table_repository.list_by_document("doc1")
    # skip 表也落 raw，规则演进后可补抽（docs/02d §5.1）
    assert len(rows) == 1
    assert rows[0].extract_status.value == "skipped"
    assert "货币资金" in (rows[0].raw_markdown or "") or "其他货币资金" in (rows[0].raw_markdown or "")


def test_core_table_calls_llm_and_stores_items(tmp_path, ctx_runtime, monkeypatch):
    items = [{"metric_name": "营业收入", "metric_period": "2024", "value_raw": "100", "unit": "元"}]
    fake = _fake_llm(items=items)
    monkeypatch.setattr("app.import_graph.nodes.table_extract.llm_extract_items", fake)
    md = _write_md(tmp_path, [(HEADLINE_CAP, HEADLINE_HTML)])
    patch = asyncio.run(node_table_extract(_state(md), ctx_runtime))
    assert fake.calls["n"] == 1
    assert patch["tables_total"] == 1
    assert patch["tables_ok"] == 1
    rows = ctx_runtime.context.report_table_repository.list_by_document("doc1")
    assert rows[0].extract_status.value == "ok"
    assert rows[0].extracted_metrics == items
    assert rows[0].statement_type.value == "headline"
    assert rows[0].statement_scope.value == "consolidated"


def test_segment_table_goes_to_llm_and_keeps_dim(tmp_path, ctx_runtime, monkeypatch):
    """segment 不再 skip：LLM 抽带 dim 的 items 原样落库，S6 再路由（docs/02g §5）。"""
    items = [
        {
            "dim_type": "industry",
            "dim_name": "煤炭采掘业",
            "metric_name": "营业收入",
            "metric_period": "2024",
            "value_raw": "100",
            "unit": "元",
        },
        {
            "dim_type": "product",
            "dim_name": "电力",
            "metric_name": "营业收入",
            "metric_period": "2024",
            "value_raw": "50",
            "unit": "元",
        },
    ]
    fake = _fake_llm(items=items)
    monkeypatch.setattr("app.import_graph.nodes.table_extract.llm_extract_items", fake)
    md = _write_md(
        tmp_path,
        [("(1). 主营业务分行业、分产品、分地区、分销售模式情况", HEADLINE_HTML)],
    )
    patch = asyncio.run(node_table_extract(_state(md), ctx_runtime))
    assert fake.calls["n"] == 1
    assert patch["tables_ok"] == 1
    rows = ctx_runtime.context.report_table_repository.list_by_document("doc1")
    assert rows[0].statement_type.value == "segment"
    assert rows[0].extracted_metrics == items


def test_mixed_tables_only_core_goes_to_llm(tmp_path, ctx_runtime, monkeypatch):
    fake = _fake_llm()
    monkeypatch.setattr("app.import_graph.nodes.table_extract.llm_extract_items", fake)
    md = _write_md(tmp_path, [(HEADLINE_CAP, HEADLINE_HTML), (NOTES_CAP, NOTES_HTML)])
    patch = asyncio.run(node_table_extract(_state(md), ctx_runtime))
    assert fake.calls["n"] == 1
    assert patch == {
        "tables_total": 2,
        "tables_ok": 1,
        "tables_skipped": 1,
        "tables_failed": 0,
    }


def test_llm_failure_retries_once_then_marks_failed(tmp_path, ctx_runtime, monkeypatch):
    fake = _fake_llm(fail_times=99)
    monkeypatch.setattr("app.import_graph.nodes.table_extract.llm_extract_items", fake)
    md = _write_md(tmp_path, [(HEADLINE_CAP, HEADLINE_HTML)])
    patch = asyncio.run(node_table_extract(_state(md), ctx_runtime))
    assert fake.calls["n"] == 2  # 失败重试 1 次
    assert patch["tables_failed"] == 1
    assert patch["tables_ok"] == 0
    rows = ctx_runtime.context.report_table_repository.list_by_document("doc1")
    assert rows[0].extract_status.value == "failed"


def test_empty_items_counts_as_failed(tmp_path, ctx_runtime, monkeypatch):
    fake = _fake_llm(items=[])
    monkeypatch.setattr("app.import_graph.nodes.table_extract.llm_extract_items", fake)
    md = _write_md(tmp_path, [(HEADLINE_CAP, HEADLINE_HTML)])
    patch = asyncio.run(node_table_extract(_state(md), ctx_runtime))
    assert patch["tables_failed"] == 1
    assert patch["tables_ok"] == 0


def test_sig_scope_backfilled_from_llm(tmp_path, ctx_runtime, monkeypatch):
    fake = _fake_llm(scope="parent")
    monkeypatch.setattr("app.import_graph.nodes.table_extract.llm_extract_items", fake)
    md = _write_md(tmp_path, [(BALANCE_CAP, BALANCE_HTML)])
    patch = asyncio.run(node_table_extract(_state(md), ctx_runtime))
    assert patch["tables_ok"] == 1
    rows = ctx_runtime.context.report_table_repository.list_by_document("doc1")
    assert rows[0].statement_type.value == "balance"
    assert rows[0].statement_scope.value == "parent"


def test_rerun_replaces_not_duplicates(tmp_path, ctx_runtime, monkeypatch):
    fake = _fake_llm()
    monkeypatch.setattr("app.import_graph.nodes.table_extract.llm_extract_items", fake)
    md = _write_md(tmp_path, [(HEADLINE_CAP, HEADLINE_HTML), (NOTES_CAP, NOTES_HTML)])
    asyncio.run(node_table_extract(_state(md), ctx_runtime))
    asyncio.run(node_table_extract(_state(md), ctx_runtime))
    rows = ctx_runtime.context.report_table_repository.list_by_document("doc1")
    assert len(rows) == 2  # 先删后写，不叠双份


def test_empty_caption_still_inserts(tmp_path, ctx_runtime, monkeypatch):
    """caption 空串时 splitlines()[0] 会越界——必须落 table_title=None 而不是整节点失败。"""
    fake = _fake_llm()
    monkeypatch.setattr("app.import_graph.nodes.table_extract.llm_extract_items", fake)
    md = _write_md(tmp_path, [("", NOTES_HTML)])
    patch = asyncio.run(node_table_extract(_state(md), ctx_runtime))
    assert patch["tables_total"] == 1
    assert patch["tables_skipped"] == 1
    rows = ctx_runtime.context.report_table_repository.list_by_document("doc1")
    assert rows[0].table_title is None


def test_no_db_transaction_spans_llm(tmp_path, ctx_runtime, monkeypatch):
    """LLM 期间不得持有未提交事务：raw 先提交，每条结果一个短事务（docs/02d §6）。

    长事务跨模型调用会占死 MySQL 连接与行锁，秒级 LLM × 多表直接把库拖死。
    并行后事件交错，只钉不变量：先 commit，再 llm，commit 数 = 1 raw + n 结果。
    """
    events: list[str] = []
    repo = ctx_runtime.context.report_table_repository
    real_commit = repo.session.commit

    def commit_spy():
        events.append("commit")
        return real_commit()

    monkeypatch.setattr(repo.session, "commit", commit_spy)

    def fake_llm(*args, **kwargs):
        events.append("llm")
        return {"items": [{"metric_name": "营业收入"}], "statement_scope": "consolidated"}

    monkeypatch.setattr("app.import_graph.nodes.table_extract.llm_extract_items", fake_llm)
    md = _write_md(tmp_path, [(HEADLINE_CAP, HEADLINE_HTML), (BALANCE_CAP, BALANCE_HTML)])
    asyncio.run(node_table_extract(_state(md), ctx_runtime))
    assert events[0] == "commit"
    assert events.count("llm") == 2
    assert events.count("commit") == 3  # raw 1 次 + 2 条结果各 1 次
    assert events.index("llm") > 0  # 所有 LLM 都在首次 raw commit 之后


def test_core_llm_calls_run_in_parallel(tmp_path, ctx_runtime, monkeypatch):
    """核心表 LLM 并行：大表串行 45s × N 会拖垮整节点，只并行抽取不并行写库。"""
    inflight = {"n": 0, "max": 0}
    lock = threading.Lock()

    def fake_llm(*args, **kwargs):
        with lock:
            inflight["n"] += 1
            inflight["max"] = max(inflight["max"], inflight["n"])
        time.sleep(0.15)
        with lock:
            inflight["n"] -= 1
        return {"items": [{"metric_name": "营业收入"}], "statement_scope": "consolidated"}

    monkeypatch.setattr("app.import_graph.nodes.table_extract.llm_extract_items", fake_llm)
    md = _write_md(tmp_path, [(HEADLINE_CAP, HEADLINE_HTML), (BALANCE_CAP, BALANCE_HTML)])
    patch = asyncio.run(node_table_extract(_state(md), ctx_runtime))
    assert patch["tables_ok"] == 2
    assert inflight["max"] >= 2, f"未并行，峰值并发={inflight['max']}"
