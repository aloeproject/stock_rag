from __future__ import annotations

from typing import Any, TypedDict


class Chunk(TypedDict):
    # 完整字段语义见 docs/02a-文本切分与Chunk设计.md §5
    chunk_id: str
    document_id: str
    chunk_type: str          # narrative | table
    content: str
    token_count: int
    char_start: int
    char_end: int
    page_no: int
    page_end: int
    title: str
    parent_title: str
    part: int
    section_path: str
    section: str
    table_id: str            # 仅 table
    table_caption: str       # 仅 table


class TableItem(TypedDict):
    metric_name: str
    metric_period: str
    metric_period_type: str
    value_raw: str
    unit: str
    is_restated: bool


class ImportState(TypedDict):
    pdf_path: str
    document_id: str

    parse_ok: bool
    markdown_path: str
    pages_map_path: str
    table_paths: list[str]

    stock_code: str
    stock_name: str
    report_type: str
    report_period: str
    fiscal_year: int
    source_file: str

    chunks: list[Chunk]
    failed_chunks: int

    embedded_chunks: int
    qdrant_ok: bool

    tables_total: int
    tables_ok: int
    tables_skipped: int
    tables_failed: int
    metrics_rows: int
    business_facts_rows: int
    unmapped_names: list[str]

    parse_status: str
    stats: dict[str, Any]
    error: str | None
    error_stage: str | None


def initial_state(pdf_path: str, document_id: str) -> ImportState:
    source_file = pdf_path.replace("\\", "/").rsplit("/", 1)[-1]
    return {
        "pdf_path": pdf_path,
        "document_id": document_id,
        "parse_ok": False,
        "markdown_path": "",
        "pages_map_path": "",
        "table_paths": [],
        "stock_code": "",
        "stock_name": "",
        "report_type": "",
        "report_period": "",
        "fiscal_year": 0,
        "source_file": source_file,
        "chunks": [],
        "failed_chunks": 0,
        "embedded_chunks": 0,
        "qdrant_ok": False,
        "tables_total": 0,
        "tables_ok": 0,
        "tables_skipped": 0,
        "tables_failed": 0,
        "metrics_rows": 0,
        "business_facts_rows": 0,
        "unmapped_names": [],
        "parse_status": "pending",
        "stats": {},
        "error": None,
        "error_stage": None,
    }
