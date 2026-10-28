from dataclasses import asdict

from sqlalchemy import BigInteger, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.entities.financial_metric import (
    ExtractSource,
    FinancialMetricDbData,
    MetricPeriodType,
)
from app.entities.report_document import ReportType
from app.entities.report_table import StatementScope, StatementType
from app.models.base import Base


class FinancialMetric(Base):
    __tablename__ = "financial_metrics"
    __table_args__ = (
        # 与 DDL uk_metric 一致：L2 物化/重跑靠它做幂等覆盖
        UniqueConstraint(
            "document_id",
            "metric_code",
            "statement_scope",
            "metric_period",
            "metric_period_type",
            name="uk_metric",
        ),
        {"comment": "标准化财务指标表"},
    )

    id: Mapped[int] = mapped_column(
        # SQLite 自增只认 INTEGER，MySQL 保持 BIGINT
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="自增主键",
    )
    document_id: Mapped[str] = mapped_column(
        String(64), index=True, comment="来源文档ID"
    )
    stock_code: Mapped[str] = mapped_column(String(16), comment="股票代码")
    report_period: Mapped[str] = mapped_column(String(16), comment="来源报告期")
    report_type: Mapped[str] = mapped_column(String(16), comment="来源报告类型")
    metric_period: Mapped[str] = mapped_column(
        String(16), comment="数值所属期间，如 2024 / 2023 / 2024末"
    )
    metric_period_type: Mapped[str] = mapped_column(
        String(16), comment="current/comparative/restated"
    )
    statement_type: Mapped[str] = mapped_column(
        String(16), comment="headline/income/balance/cashflow/ratio"
    )
    statement_scope: Mapped[str] = mapped_column(
        String(16), default="consolidated", comment="consolidated/parent"
    )
    metric_code: Mapped[str] = mapped_column(String(64), comment="标准化指标编码")
    metric_name: Mapped[str | None] = mapped_column(
        String(128), nullable=True, comment="财报原文指标名"
    )
    metric_value: Mapped[float | None] = mapped_column(
        Numeric(24, 6), nullable=True, comment="标准化数值"
    )
    unit: Mapped[str | None] = mapped_column(
        String(16), nullable=True, comment="元/万元/亿元/%/元每股"
    )
    value_raw: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="原文数值字符串"
    )
    is_restated: Mapped[int] = mapped_column(
        Integer, default=0, comment="是否追溯调整值"
    )
    page_no: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="PDF 页码"
    )
    extract_source: Mapped[str] = mapped_column(
        String(16), default="table", comment="headline/table/derived"
    )


class FinancialMetricMapper:
    @staticmethod
    def to_entity(model: FinancialMetric) -> FinancialMetricDbData:
        return FinancialMetricDbData(
            document_id=model.document_id,
            stock_code=model.stock_code,
            report_period=model.report_period,
            report_type=ReportType(model.report_type),
            metric_period=model.metric_period,
            metric_period_type=MetricPeriodType(model.metric_period_type),
            statement_type=StatementType(model.statement_type),
            statement_scope=StatementScope(model.statement_scope),
            metric_code=model.metric_code,
            metric_name=model.metric_name,
            metric_value=model.metric_value,
            unit=model.unit,
            value_raw=model.value_raw,
            is_restated=model.is_restated,
            page_no=model.page_no,
            extract_source=ExtractSource(model.extract_source),
        )

    @staticmethod
    def to_model(entity: FinancialMetricDbData) -> FinancialMetric:
        data = asdict(entity)
        data["report_type"] = entity.report_type.value
        data["metric_period_type"] = entity.metric_period_type.value
        data["statement_type"] = entity.statement_type.value
        data["statement_scope"] = entity.statement_scope.value
        data["extract_source"] = entity.extract_source.value
        return FinancialMetric(**data)
