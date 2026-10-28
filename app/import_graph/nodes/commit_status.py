"""S7 状态提交（docs/02f）。

本节点只写 report_documents.parse_status/stats，不写业务数据——
report_tables / financial_metrics / Qdrant 由 S5a/S5b/S6 各自落。
"""
from __future__ import annotations

from collections import Counter

from langgraph.runtime import Runtime
from langgraph.types import StreamWriter

from app.core.logger import logger
from app.entities.report_document import ParseStatus, ReportDocumentDbData, ReportType
from app.import_graph.context import DataAgentContext
from app.import_graph.nodes.progress import progress_step
from app.import_graph.state import ImportState


def resolve_vector_ok(state: ImportState, qdrant_count: int) -> bool:
    """向量分支是否成功：State 本跑信号 OR 库内点数兜底。

    为什么要库兜底：只重跑 S5b～S7 时 State 的 qdrant_ok/embedded_chunks
    是工厂默认空值，只信 State 会把已在库的向量误判为失败（docs/02f §1.2）。
    """
    chunks = state.get("chunks") or []
    embedded = state.get("embedded_chunks") or 0
    # 本跑全灭否决：尝试过 chunks 但既没嵌入成功，库里也没点
    if chunks and embedded == 0 and qdrant_count == 0:
        return False
    return bool(state.get("qdrant_ok")) or embedded > 0 or qdrant_count > 0


def resolve_metrics_ok(state: ImportState, fm_count: int, tables_pending: int) -> bool:
    """指标分支是否成功：State/库产物计数 + pending 一票否决。

    为什么 pending 不算成功：pending 表示该表既没 ok 也没显式 failed，
    属于 S5b 未跑完；宁可 failed(part) 触发重跑，也不写 done 谎报完成（docs/02f §4.2）。
    """
    if tables_pending > 0:
        return False
    metrics_rows = state.get("metrics_rows") or 0
    tables_total = state.get("tables_total") or 0
    tables_ok = state.get("tables_ok") or 0
    tables_failed = state.get("tables_failed") or 0
    # 本跑全灭：尝试过表但全失败且库无指标
    if tables_total > 0 and tables_ok == 0 and tables_failed > 0 and fm_count == 0:
        return False
    return metrics_rows > 0 or fm_count > 0


def derive_parse_status(
    state: ImportState,
    vector_ok: bool,
    metrics_ok: bool,
) -> ParseStatus:
    """State + 双分支结论 → 终态三值（docs/02f §4.3）。"""
    error_stage = state.get("error_stage")
    if state.get("parse_ok") is False or error_stage in ("parse", "metadata"):
        return ParseStatus.FAILED
    if vector_ok and metrics_ok:
        return ParseStatus.DONE
    if vector_ok or metrics_ok:
        return ParseStatus.FAILED_PART
    return ParseStatus.FAILED


def _load_signals(state: ImportState, runtime: Runtime[DataAgentContext]) -> dict:
    """查库产物计数；库查询异常退回 State 并 WARN（docs/02f §7）。"""
    ctx = runtime.context
    document_id = state["document_id"]
    tables: list = []
    fm_count = 0
    qdrant_count = 0

    table_repo = getattr(ctx, "report_table_repository", None)
    if table_repo is not None:
        try:
            tables = table_repo.list_by_document(document_id)
        except Exception as e:  # noqa: BLE001 —— 库不可用时降级 State，不打断收口
            logger.warning("commit_status 读 report_tables 失败，退回 State: {}", e)
    fm_repo = getattr(ctx, "financial_metric_repository", None)
    if fm_repo is not None:
        try:
            fm_count = len(fm_repo.list_by_document(document_id))
        except Exception as e:  # noqa: BLE001
            logger.warning("commit_status 读 financial_metrics 失败，退回 State: {}", e)
    qdrant = getattr(ctx, "qdrant", None)
    if qdrant is not None:
        try:
            qdrant_count = int(qdrant.count_by_document(document_id))
        except Exception as e:  # noqa: BLE001
            logger.warning("commit_status 读 Qdrant count 失败，退回 State: {}", e)

    by_status = Counter(t.extract_status for t in tables)
    return {
        "tables_total": len(tables),
        "tables_ok": by_status.get("ok", 0),
        "tables_skipped": by_status.get("skipped", 0),
        "tables_failed": by_status.get("failed", 0),
        "tables_pending": by_status.get("pending", 0),
        "fm_count": fm_count,
        "qdrant_count": qdrant_count,
    }


def build_stats(state: ImportState, sig: dict, vector_ok: bool, metrics_ok: bool) -> dict:
    """stats 以库计数为准，State 只补 failed_chunks/unmapped/error（docs/02f §5.3）。

    为什么信库不信 State：State 是本次运行切片，只重跑指标时会把 chunks 抄成 0，
    覆盖上一轮真实值（docs/02f §1.1）。
    """
    qdrant_count = sig["qdrant_count"]
    # chunks/embedded：Qdrant count 优先，否则本跑 State
    if qdrant_count > 0:
        chunks = qdrant_count
        embedded = qdrant_count
    else:
        chunks = len(state.get("chunks") or [])
        embedded = state.get("embedded_chunks") or 0

    unmapped = list(dict.fromkeys(state.get("unmapped_names") or []))
    return {
        "chunks": chunks,
        "failed_chunks": state.get("failed_chunks") or 0,
        "embedded_chunks": embedded,
        "tables_total": sig["tables_total"] or state.get("tables_total") or 0,
        "tables_ok": sig["tables_ok"] or state.get("tables_ok") or 0,
        "tables_skipped": sig["tables_skipped"] or state.get("tables_skipped") or 0,
        "tables_failed": sig["tables_failed"] or state.get("tables_failed") or 0,
        "metrics_rows": sig["fm_count"] or state.get("metrics_rows") or 0,
        "unmapped_names": unmapped,
        "vector_ok": vector_ok,
        "metrics_ok": metrics_ok,
        "error": state.get("error"),
        "error_stage": state.get("error_stage"),
    }


def _upsert_document(
    state: ImportState,
    runtime: Runtime[DataAgentContext],
    status: ParseStatus,
    stats: dict,
) -> None:
    """upsert 而不是 update：report_documents 空表时 update 会静默丢状态（docs/02f §1.1）。"""
    doc_repo = runtime.context.report_document_repository
    if doc_repo is None:
        raise RuntimeError("report_document_repository 未注入，S7 无法写状态")

    document_id = state["document_id"]
    stock_code = state.get("stock_code") or ""
    stock_name = state.get("stock_name")
    report_type_raw = state.get("report_type") or "annual"
    report_period = state.get("report_period") or ""
    fiscal_year = state.get("fiscal_year") or 0
    source_file = state.get("source_file")

    # State 缺元数据时从已有文档行继承（只重跑后半段时 S2 字段不在 State）
    existing = doc_repo.get_by_id(document_id)
    if existing is not None:
        stock_code = stock_code or existing.stock_code
        stock_name = stock_name or existing.stock_name
        report_period = report_period or existing.report_period
        fiscal_year = fiscal_year or existing.fiscal_year
        source_file = source_file or existing.source_file
        try:
            report_type_raw = report_type_raw or existing.report_type.value
        except Exception:  # noqa: BLE001
            pass

    doc_repo.upsert(
        ReportDocumentDbData(
            document_id=document_id,
            stock_code=stock_code,
            stock_name=stock_name,
            report_type=ReportType(report_type_raw),
            report_period=report_period,
            fiscal_year=fiscal_year,
            source_file=source_file,
            parse_status=status,
            stats=stats,
        )
    )
    doc_repo.session.commit()


async def node_commit_status(
    state: ImportState,
    runtime: Runtime[DataAgentContext],
    writer: StreamWriter | None = None,
) -> dict:
    """S7: 汇合双分支，写 parse_status/stats。只改状态列，不写业务数据。"""
    logger.info(f"[node_commit_status] 开始处理文档: {state.get('document_id')}")
    with progress_step(writer, "状态提交") as outcome:
        sig = _load_signals(state, runtime)
        vector_ok = resolve_vector_ok(state, sig["qdrant_count"])
        metrics_ok = resolve_metrics_ok(state, sig["fm_count"], sig["tables_pending"])
        status = derive_parse_status(state, vector_ok, metrics_ok)
        stats = build_stats(state, sig, vector_ok, metrics_ok)

        if len(stats.get("unmapped_names") or []) > 20:
            logger.warning(
                "commit_status unmapped_names 共 {} 个，前 20: {}",
                len(stats["unmapped_names"]),
                stats["unmapped_names"][:20],
            )

        try:
            _upsert_document(state, runtime, status, stats)
        except Exception as e:  # noqa: BLE001 —— 已落库的数据不因收口失败而中断
            logger.error("commit_status upsert 失败: {}", e, exc_info=True)
            stats["error"] = str(e)
            stats["error_stage"] = "commit"
            outcome["ok"] = False
            return {"parse_status": ParseStatus.FAILED_PART.value, "stats": stats}

        logger.info(
            "commit_status {} status={} vector_ok={} metrics_ok={}",
            state.get("document_id"),
            status.value,
            vector_ok,
            metrics_ok,
        )
        return {"parse_status": status.value, "stats": stats}


if __name__ == "__main__":
    # 真实收口：600628 新世界已落库产物（S5b+S6 完成、S5a 未跑）→ S7 → report_documents。
    # 验收对照 docs/02f §5.2/§9：空表要能建行；现状应 failed(part)；重跑幂等。
    import asyncio
    from collections import Counter

    from langgraph.runtime import Runtime
    from sqlalchemy import text

    from app.clients.mysql_client import MySQLClientManager
    from app.clients.qdrant_client import QdrantClientManager
    from app.models.base import Base
    from app.repositories.financial_metric_repository import FinancialMetricRepository
    from app.repositories.report_document_repository import ReportDocumentRepository
    from app.repositories.report_table_repository import ReportTableRepository

    DOC_ID = "4ffe5afef59df4342e46bfd86b41b62992f642c7"  # 600628 新世界 PDF sha1
    SOURCE_FILE = "600628_2024_新世界_新世界2024年年度报告_2025-04-12.pdf"

    mgr = MySQLClientManager()
    mgr.init()
    Base.metadata.create_all(mgr.engine)
    session = mgr.session()

    doc_repo = ReportDocumentRepository(session)
    table_repo = ReportTableRepository(session)
    metric_repo = FinancialMetricRepository(session)

    print("== 库内产物（S7 输入面） ==")
    n_docs = session.execute(text("SELECT COUNT(*) FROM report_documents")).scalar()
    status_rows = session.execute(
        text("SELECT extract_status, COUNT(*) FROM report_tables GROUP BY extract_status")
    ).all()
    n_fm = session.execute(
        text("SELECT COUNT(*) FROM financial_metrics")
    ).scalar()
    print(f"  report_documents={n_docs}（02f §1.1：update 会静默丢状态，必须 upsert）")
    print(f"  report_tables by status={ {r[0]: r[1] for r in status_rows} }")
    src_rows = session.execute(
        text("SELECT extract_source, COUNT(*) FROM financial_metrics GROUP BY extract_source")
    ).all()
    print(f"  financial_metrics={n_fm} (source={ {r[0]: r[1] for r in src_rows} })")

    qdrant = QdrantClientManager()
    try:
        qdrant.init()
        q_count = qdrant.count_by_document(DOC_ID)
    except Exception as e:  # noqa: BLE001
        q_count = -1
        print(f"  Qdrant 不可达: {e}")
    else:
        print(f"  Qdrant count(document_id)={q_count}（S5a 未跑则为 0）")

    # 本跑只跑 S7：State 向量/表计数为工厂空值，产物以库为准（02f §3）
    state: ImportState = {
        "document_id": DOC_ID,
        "stock_code": "600628",
        "stock_name": "新世界",
        "report_type": "annual",
        "report_period": "2024",
        "fiscal_year": 2024,
        "source_file": SOURCE_FILE,
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
    }  # type: ignore[typeddict-item]

    ctx = DataAgentContext(
        report_document_repository=doc_repo,
        report_table_repository=table_repo,
        financial_metric_repository=metric_repo,
        qdrant=qdrant if q_count >= 0 else None,
    )
    patch = asyncio.run(node_commit_status(state, Runtime(context=ctx)))  # type: ignore[arg-type]
    print(f"== state patch == {patch}")
    stats = patch["stats"]

    print("== 02f §9 验收 ==")
    assert patch["parse_status"] == "failed(part)", patch["parse_status"]
    print("  ✓ 新世界现状 → failed(part)")
    assert stats["tables_total"] == 210 and stats["tables_ok"] == 10 and stats["tables_skipped"] == 200, stats
    print(f"  ✓ tables 210/10/200")
    assert stats["metrics_rows"] == 133, stats["metrics_rows"]
    print("  ✓ metrics_rows=133")
    assert stats["vector_ok"] is False and stats["metrics_ok"] is True, stats
    print("  ✓ vector_ok=false, metrics_ok=true")

    doc = doc_repo.get_by_id(DOC_ID)
    assert doc is not None, "空表必须 upsert 建行（02f §1.1）"
    assert doc.parse_status.value == "failed(part)"
    assert doc.stock_code == "600628" and doc.report_period == "2024"
    print(f"  ✓ report_documents 建行 parse_status={doc.parse_status.value}")

    # 幂等：再跑一遍状态一致、不增行
    patch2 = asyncio.run(node_commit_status(state, Runtime(context=ctx)))  # type: ignore[arg-type]
    n_docs2 = session.execute(text("SELECT COUNT(*) FROM report_documents")).scalar()
    print(f"== 幂等重跑 == status {patch['parse_status']} -> {patch2['parse_status']} rows {n_docs} -> {n_docs2}")
    assert patch2["parse_status"] == patch["parse_status"]
    assert patch2["stats"]["metrics_rows"] == stats["metrics_rows"]
    assert n_docs2 == 1, "report_documents 不得增行"

    # 单侧重跑兜底演示（不落 done 到真库）：State 向量空 + Qdrant 有点 → 应判 done
    class _FakeQdrant:
        def count_by_document(self, document_id: str) -> int:
            return 412

    ctx_fallback = DataAgentContext(
        report_document_repository=doc_repo,
        report_table_repository=table_repo,
        financial_metric_repository=metric_repo,
        qdrant=_FakeQdrant(),
    )
    v_ok = resolve_vector_ok(state, 412)
    m_ok = resolve_metrics_ok(state, fm_count=133, tables_pending=0)
    status_fb = derive_parse_status(state, v_ok, m_ok)
    print(f"== 单侧重跑兜底（内存推演，不写库）== vector_ok={v_ok} metrics_ok={m_ok} → {status_fb.value}")
    assert status_fb.value == "done"

    session.close()
    print("== 完成 ==")
