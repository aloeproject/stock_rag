from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class StatementType(StrEnum):
    HEADLINE = "headline"
    INCOME = "income"
    BALANCE = "balance"
    CASHFLOW = "cashflow"
    EQUITY = "equity"
    RATIO = "ratio"
    # 业务切片表：主营业务分行业/分产品/分地区/分销售模式（docs/02g）
    SEGMENT = "segment"
    OTHER = "other"


class StatementScope(StrEnum):
    CONSOLIDATED = "consolidated"
    PARENT = "parent"
    EMPTY = "empty"


class ExtractStatus(StrEnum):
    PENDING = "pending"
    OK = "ok"
    FAILED = "failed"
    # 规则过滤跳过、不调 LLM（docs/02d §5.1）；可强制重跑补抽
    SKIPPED = "skipped"


@dataclass
class ReportTableDbData:
    document_id: str
    page_no: int | None
    table_title: str | None
    statement_type: StatementType | None
    statement_scope: StatementScope | None
    raw_markdown: str | None
    extracted_metrics: list[dict[str, Any]] | None
    extract_status: ExtractStatus
    # uk 裁决用 table_id 排序（docs/02e §6）；insert 时为 None，list 时回填
    id: int | None = None
