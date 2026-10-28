"""业务切片事实实体（docs/02g）。

为什么不并进 FinancialMetric：开放维度不该掺进公司级 uk 与 L2 公式
（docs/02g §1.1）；uk 用 metric_name 不用 metric_code，后者可空
MySQL NULL 不去重（docs/02g §3.2）。
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from app.entities.financial_metric import ExtractSource
from app.entities.report_document import ReportType


class DimType(StrEnum):
    """切片类型枚举可扩；过滤要稳定所以不用自由文本（docs/02g §7.3）。"""

    INDUSTRY = "industry"
    PRODUCT = "product"
    REGION = "region"
    CHANNEL = "channel"
    CUSTOMER = "customer"
    OTHER = "other"


@dataclass
class BusinessFactDbData:
    document_id: str
    stock_code: str
    report_period: str
    report_type: ReportType
    dim_type: DimType
    dim_name: str
    # 未映射列头为 NULL：metric_name 才是稳定标识（docs/02g §3.2）
    metric_code: str | None
    metric_name: str
    metric_period: str
    metric_value: Decimal | None
    unit: str | None
    value_raw: str | None
    page_no: int | None
    extract_source: ExtractSource
