"""node_commit_status 测试：状态判定、库兜底、upsert 幂等（docs/02f）。"""
import asyncio

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.entities.report_document import ParseStatus
from app.entities.report_table import ExtractStatus, ReportTableDbData, StatementScope, StatementType
from app.entities.financial_metric import ExtractSource, FinancialMetricDbData, MetricPeriodType
from app.entities.report_document import ReportType
from app.import_graph.nodes.commit_status import (
    derive_parse_status,
    node_commit_status,
    resolve_metrics_ok,
    resolve_vector_ok,
)
from app.models.base import Base
from app.repositories.financial_metric_repository import FinancialMetricRepository
from app.repositories.report_document_repository import ReportDocumentRepository
from app.repositories.report_table_repository import ReportTableRepository


class FakeQdrant:
    def __init__(self, n: int):
        self.n = n

    def count_by_document(self, document_id: str) -> int:
        return self.n


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    yield s
    s.close()


@pytest.fixture()
def ctx_runtime(runtime, session):
    runtime.context.report_document_repository = ReportDocumentRepository(session)
    runtime.context.report_table_repository = ReportTableRepository(session)
    runtime.context.financial_metric_repository = FinancialMetricRepository(session)
    return runtime


def _state(**over):
    base = {
        "document_id": "doc1",
        "stock_code": "600628",
        "stock_name": "新世界",
        "report_type": "annual",
        "report_period": "2024",
        "fiscal_year": 2024,
        "source_file": "600628.pdf",
        "parse_ok": True,
        "error": None,
        "error_stage": None,
        "chunks": [],
        "failed_chunks": 0,
        "embedded_chunks": 0,
        "qdrant_ok": False,
        "tables_total": 0,
        "tables_ok": 0,
        "tables_skipped": 0,
        "tables_failed": 0,
        "metrics_rows": 0,
        "unmapped_names": [],
    }
    base.update(over)
    return base


def _table(repo, status=ExtractStatus.OK, statement=StatementType.BALANCE):
    return repo.insert(
        ReportTableDbData(
            document_id="doc1",
            page_no=1,
            table_title="合并资产负债表",
            statement_type=statement,
            statement_scope=StatementScope.CONSOLIDATED,
            raw_markdown="raw",
            extracted_metrics=[{"metric_name": "总资产", "value_raw": "1"}],
            extract_status=status,
        )
    )


def _metric(repo, n=1):
    # uk 含 metric_code，必须逐条换 code，否则 upsert 覆盖只剩 1 行
    for i in range(n):
        repo.upsert(
            FinancialMetricDbData(
                document_id="doc1",
                stock_code="600628",
                report_period="2024",
                report_type=ReportType.ANNUAL,
                metric_period="2024",
                metric_period_type=MetricPeriodType.CURRENT,
                statement_type=StatementType.BALANCE,
                statement_scope=StatementScope.CONSOLIDATED,
                metric_code=f"m{i}",
                metric_name=f"科目{i}",
                metric_value=1,
                unit="元",
                value_raw="1",
                is_restated=0,
                page_no=1,
                extract_source=ExtractSource.TABLE,
            )
        )


# ---------- 纯函数 ----------

def test_derive_s1_failure_is_failed():
    st = _state(parse_ok=False, error_stage="parse")
    assert derive_parse_status(st, vector_ok=True, metrics_ok=True) == ParseStatus.FAILED
    assert derive_parse_status(_state(error_stage="metadata"), True, True) == ParseStatus.FAILED


def test_derive_both_ok_is_done():
    assert derive_parse_status(_state(), True, True) == ParseStatus.DONE


def test_derive_one_side_is_failed_part():
    assert derive_parse_status(_state(), True, False) == ParseStatus.FAILED_PART
    assert derive_parse_status(_state(), False, True) == ParseStatus.FAILED_PART


def test_derive_both_fail_is_failed():
    assert derive_parse_status(_state(), False, False) == ParseStatus.FAILED


def test_vector_ok_db_fallback_for_metrics_rerun():
    """只重跑指标时 State 向量为 0，Qdrant 有点则不能判失败（docs/02f §1.2）。"""
    st = _state(qdrant_ok=False, embedded_chunks=0, metrics_rows=133)
    assert resolve_vector_ok(st, qdrant_count=412) is True


def test_vector_all_failed_is_not_ok():
    st = _state(chunks=[{"chunk_id": "c1"}], embedded_chunks=0, failed_chunks=1, qdrant_ok=False)
    assert resolve_vector_ok(st, qdrant_count=0) is False


def test_metrics_pending_veto():
    """pending 表 = S5b 未跑完，即使已有 metrics 也不能算成功（docs/02f §4.2）。"""
    st = _state(metrics_rows=10)
    assert resolve_metrics_ok(st, fm_count=10, tables_pending=1) is False
    assert resolve_metrics_ok(st, fm_count=10, tables_pending=0) is True


# ---------- 节点 ----------

def test_node_creates_document_row_when_missing(ctx_runtime, session):
    """report_documents 空表时必须 upsert 建行——update 会静默丢状态（docs/02f §1.1）。"""
    _metric(ctx_runtime.context.financial_metric_repository, n=3)
    patch = asyncio.run(
        node_commit_status(_state(metrics_rows=3), ctx_runtime)
    )
    doc = ctx_runtime.context.report_document_repository.get_by_id("doc1")
    assert doc is not None
    assert doc.parse_status == ParseStatus.FAILED_PART  # 无向量
    assert patch["parse_status"] == "failed(part)"
    assert patch["stats"]["metrics_ok"] is True
    assert patch["stats"]["vector_ok"] is False


def test_node_xinshijie_real_shape(ctx_runtime):
    """新世界现状：210 表(10 ok/200 skipped) + 133 指标 + 无向量 → failed(part)。"""
    repo = ctx_runtime.context.report_table_repository
    for _ in range(10):
        _table(repo, ExtractStatus.OK)
    for _ in range(200):
        _table(repo, ExtractStatus.SKIPPED, statement=None)
    _metric(ctx_runtime.context.financial_metric_repository, n=133)
    patch = asyncio.run(
        node_commit_status(
            _state(tables_total=210, tables_ok=10, tables_skipped=200, metrics_rows=133),
            ctx_runtime,
        )
    )
    stats = patch["stats"]
    assert stats["tables_total"] == 210
    assert stats["tables_ok"] == 10
    assert stats["tables_skipped"] == 200
    assert stats["metrics_rows"] == 133
    assert patch["parse_status"] == "failed(part)"


def test_node_both_branches_done(ctx_runtime):
    _metric(ctx_runtime.context.financial_metric_repository, n=2)
    ctx_runtime.context.qdrant = FakeQdrant(412)
    patch = asyncio.run(
        node_commit_status(
            _state(qdrant_ok=True, embedded_chunks=412, metrics_rows=2),
            ctx_runtime,
        )
    )
    assert patch["parse_status"] == "done"
    assert patch["stats"]["vector_ok"] is True
    assert patch["stats"]["metrics_ok"] is True


def test_node_metrics_rerun_keeps_done_via_qdrant(ctx_runtime):
    """只重跑 S5b～S7：State 向量空，靠 Qdrant count 兜底（docs/02f §6.3）。"""
    _metric(ctx_runtime.context.financial_metric_repository, n=5)
    ctx_runtime.context.qdrant = FakeQdrant(412)
    patch = asyncio.run(
        node_commit_status(_state(metrics_rows=5), ctx_runtime)
    )
    assert patch["parse_status"] == "done"
    assert patch["stats"]["vector_ok"] is True
    assert patch["stats"]["chunks"] == 412  # stats 信库


def test_node_unmapped_does_not_block_done(ctx_runtime):
    _metric(ctx_runtime.context.financial_metric_repository, n=1)
    ctx_runtime.context.qdrant = FakeQdrant(10)
    patch = asyncio.run(
        node_commit_status(
            _state(unmapped_names=["少数股东权益", "盈余公积"], metrics_rows=1),
            ctx_runtime,
        )
    )
    assert patch["parse_status"] == "done"
    assert patch["stats"]["unmapped_names"] == ["少数股东权益", "盈余公积"]


def test_node_pending_table_blocks_done(ctx_runtime):
    _table(ctx_runtime.context.report_table_repository, ExtractStatus.PENDING)
    _metric(ctx_runtime.context.financial_metric_repository, n=1)
    ctx_runtime.context.qdrant = FakeQdrant(10)
    patch = asyncio.run(
        node_commit_status(_state(metrics_rows=1), ctx_runtime)
    )
    assert patch["parse_status"] == "failed(part)"
    assert patch["stats"]["metrics_ok"] is False


def test_node_s1_failure_still_writes_stats(ctx_runtime):
    patch = asyncio.run(
        node_commit_status(_state(parse_ok=False, error_stage="parse", error="timeout"), ctx_runtime)
    )
    assert patch["parse_status"] == "failed"
    doc = ctx_runtime.context.report_document_repository.get_by_id("doc1")
    assert doc is not None
    assert doc.stats["error_stage"] == "parse"


def test_node_idempotent(ctx_runtime):
    _metric(ctx_runtime.context.financial_metric_repository, n=2)
    ctx_runtime.context.qdrant = FakeQdrant(1)
    st = _state(qdrant_ok=True, embedded_chunks=1, metrics_rows=2)
    p1 = asyncio.run(node_commit_status(st, ctx_runtime))
    p2 = asyncio.run(node_commit_status(st, ctx_runtime))
    assert p1["parse_status"] == p2["parse_status"] == "done"
    session = ctx_runtime.context.report_document_repository.session
    from sqlalchemy import text
    n = session.execute(text("SELECT COUNT(*) FROM report_documents")).scalar()
    assert n == 1
