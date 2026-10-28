"""节点进度事件测试：writer 推 {"type":"progress","step","status"}。

SSE 前端靠这些事件渲染进度条；节点入口 running、出口 done/failed。
"""
import asyncio
import json
from pathlib import Path

import pytest
from langgraph.runtime import Runtime

from app.import_graph.context import DataAgentContext
from app.import_graph.nodes.sectionize import node_sectionize
from app.import_graph.nodes.split import node_split
from app.import_graph.nodes.progress import emit_progress, progress_step


class RecordingWriter:
    def __init__(self):
        self.events: list[dict] = []

    def __call__(self, event):
        self.events.append(event)


def test_emit_progress_writes_event():
    w = RecordingWriter()
    emit_progress(w, "解析PDF", "running")
    assert w.events == [{"type": "progress", "step": "解析PDF", "status": "running"}]


def test_emit_progress_none_writer_is_noop():
    emit_progress(None, "解析PDF", "running")  # 不应抛异常


def test_progress_step_context_manager_emits_running_then_done():
    w = RecordingWriter()
    with progress_step(w, "切分"):
        pass
    assert w.events == [
        {"type": "progress", "step": "切分", "status": "running"},
        {"type": "progress", "step": "切分", "status": "done"},
    ]


def test_progress_step_context_manager_emits_failed_on_exception():
    w = RecordingWriter()
    try:
        with progress_step(w, "切分"):
            raise ValueError("boom")
    except ValueError:
        pass
    assert w.events == [
        {"type": "progress", "step": "切分", "status": "running"},
        {"type": "progress", "step": "切分", "status": "failed"},
    ]


def test_progress_step_outcome_false_emits_failed():
    """节点吞异常返回 error dict 时，outcome["ok"]=False 应报 failed 而非 done。"""
    w = RecordingWriter()
    with progress_step(w, "解析PDF") as outcome:
        outcome["ok"] = False
    assert w.events == [
        {"type": "progress", "step": "解析PDF", "status": "running"},
        {"type": "progress", "step": "解析PDF", "status": "failed"},
    ]


def test_node_split_emits_progress(tmp_path, runtime: Runtime[DataAgentContext]):
    md = "# 第十节 财务报告\n前言。\n"
    md_path = tmp_path / "content.md"
    md_path.write_text(md, encoding="utf-8")
    pages_path = tmp_path / "pages.json"
    pages_path.write_text(
        json.dumps([{"char_start": 0, "char_end": len(md), "page_no": 1}]),
        encoding="utf-8",
    )
    state = {
        "document_id": "doc1",
        "markdown_path": str(md_path),
        "pages_map_path": str(pages_path),
    }
    w = RecordingWriter()
    node_split(state, runtime, writer=w)
    steps = [(e["step"], e["status"]) for e in w.events]
    assert ("切分", "running") in steps
    assert ("切分", "done") in steps


def test_node_sectionize_emits_progress(runtime: Runtime[DataAgentContext]):
    chunks = [{
        "chunk_id": "doc1:0000",
        "document_id": "doc1",
        "chunk_type": "narrative",
        "content": "前言说明文字。",
        "section": "",
        "page_no": 1,
        "title": "第十节 财务报告",
    }]
    w = RecordingWriter()
    node_sectionize({"chunks": chunks}, runtime, writer=w)
    steps = [(e["step"], e["status"]) for e in w.events]
    assert ("章节归类", "running") in steps
    assert ("章节归类", "done") in steps


def test_node_parse_emits_failed_when_file_missing(runtime: Runtime[DataAgentContext]):
    from app.import_graph.nodes.parse import node_parse

    w = RecordingWriter()
    patch = node_parse(
        {"document_id": "doc1", "pdf_path": "/nope/missing.pdf"},
        runtime,
        writer=w,
    )
    assert patch["parse_ok"] is False
    steps = [(e["step"], e["status"]) for e in w.events]
    assert ("解析PDF", "running") in steps
    assert ("解析PDF", "failed") in steps


def test_node_metadata_emits_failed_when_invalid(runtime: Runtime[DataAgentContext]):
    from app.import_graph.nodes.metadata import node_metadata

    w = RecordingWriter()
    patch = node_metadata({"pdf_path": "/nope/unknown.pdf"}, runtime, writer=w)
    assert patch["error_stage"] == "metadata"
    steps = [(e["step"], e["status"]) for e in w.events]
    assert ("元数据抽取", "running") in steps
    assert ("元数据抽取", "failed") in steps


def test_node_embed_store_emits_running_before_qdrant_check(runtime: Runtime[DataAgentContext]):
    from app.import_graph.nodes.embed_store import node_embed_store

    w = RecordingWriter()
    with pytest.raises(RuntimeError, match="qdrant"):
        asyncio.run(node_embed_store({"document_id": "doc1", "chunks": []}, runtime, writer=w))
    steps = [(e["step"], e["status"]) for e in w.events]
    assert ("向量入库", "running") in steps


def test_node_commit_status_emits_progress(runtime: Runtime[DataAgentContext]):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.models.base import Base
    from app.repositories.financial_metric_repository import FinancialMetricRepository
    from app.repositories.report_document_repository import ReportDocumentRepository
    from app.repositories.report_table_repository import ReportTableRepository
    from app.import_graph.nodes.commit_status import node_commit_status

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    runtime.context.report_document_repository = ReportDocumentRepository(session)
    runtime.context.report_table_repository = ReportTableRepository(session)
    runtime.context.financial_metric_repository = FinancialMetricRepository(session)

    w = RecordingWriter()
    try:
        asyncio.run(node_commit_status({"document_id": "doc1"}, runtime, writer=w))
    finally:
        session.close()
    steps = [(e["step"], e["status"]) for e in w.events]
    assert ("状态提交", "running") in steps
    assert ("状态提交", "done") in steps
