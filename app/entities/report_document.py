from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ReportType(StrEnum):
    ANNUAL = "annual"
    Q1 = "q1"
    Q2 = "q2"
    Q3 = "q3"
    INTERIM = "interim"
    PROSPECTUS = "prospectus"


class ParseStatus(StrEnum):
    PENDING = "pending"
    PARSING = "parsing"
    SPLITTING = "splitting"
    EMBEDDING = "embedding"
    EXTRACTING = "extracting"
    DONE = "done"
    FAILED_PART = "failed(part)"
    FAILED = "failed"


@dataclass
class ReportDocumentDbData:
    document_id: str
    stock_code: str
    stock_name: str | None
    report_type: ReportType
    report_period: str
    fiscal_year: int
    source_file: str | None
    parse_status: ParseStatus
    stats: dict[str, Any] | None
