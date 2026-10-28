from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from app.entities.report_document import ReportType
from app.entities.report_table import StatementScope, StatementType


class MetricPeriodType(StrEnum):
    CURRENT = "current"
    COMPARATIVE = "comparative"
    RESTATED = "restated"


class ExtractSource(StrEnum):
    HEADLINE = "headline"
    TABLE = "table"
    DERIVED = "derived"


@dataclass
class FinancialMetricDbData:
    document_id: str
    stock_code: str
    report_period: str
    report_type: ReportType
    metric_period: str
    metric_period_type: MetricPeriodType
    statement_type: StatementType
    statement_scope: StatementScope
    metric_code: str
    metric_name: str | None
    metric_value: Decimal | None
    unit: str | None
    value_raw: str | None
    is_restated: int
    page_no: int | None
    extract_source: ExtractSource
