from dataclasses import asdict

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.entities.metric_definition import (
    MetricCategory,
    MetricDefinitionDbData,
    MetricLevel,
)
from app.entities.report_table import StatementType
from app.models.base import Base


class MetricDefinition(Base):
    __tablename__ = "metric_definitions"
    __table_args__ = {"comment": "指标口径字典表"}

    metric_code: Mapped[str] = mapped_column(
        String(64), primary_key=True, comment="标准化指标编码"
    )
    metric_name: Mapped[str] = mapped_column(String(128), comment="指标中文标准名")
    aliases: Mapped[str | None] = mapped_column(
        String(255), nullable=True, comment="原文别名，竖线分隔"
    )
    category: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="headline/asset/liability/equity/income/cost/cashflow/profitability/solvency/efficiency/growth",
    )
    statement_type: Mapped[str | None] = mapped_column(
        String(16), nullable=True, comment="headline/income/balance/cashflow/ratio"
    )
    metric_level: Mapped[int] = mapped_column(
        Integer, default=1, comment="1-L1 2-L2 3-L3"
    )
    calc_formula: Mapped[str | None] = mapped_column(
        String(255), nullable=True, comment="L2/L3 公式；L1 空"
    )
    unit_default: Mapped[str | None] = mapped_column(
        String(16), nullable=True, comment="默认单位"
    )
    description: Mapped[str | None] = mapped_column(
        String(512), nullable=True, comment="口径说明"
    )


class MetricDefinitionMapper:
    @staticmethod
    def to_entity(model: MetricDefinition) -> MetricDefinitionDbData:
        return MetricDefinitionDbData(
            metric_code=model.metric_code,
            metric_name=model.metric_name,
            aliases=model.aliases,
            category=MetricCategory(model.category) if model.category else None,
            statement_type=StatementType(model.statement_type) if model.statement_type else None,
            metric_level=MetricLevel(model.metric_level),
            calc_formula=model.calc_formula,
            unit_default=model.unit_default,
            description=model.description,
        )

    @staticmethod
    def to_model(entity: MetricDefinitionDbData) -> MetricDefinition:
        data = asdict(entity)
        data["category"] = entity.category.value if entity.category else None
        data["statement_type"] = entity.statement_type.value if entity.statement_type else None
        data["metric_level"] = int(entity.metric_level)
        return MetricDefinition(**data)
