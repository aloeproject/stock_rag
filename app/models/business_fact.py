from dataclasses import asdict

from sqlalchemy import BigInteger, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.entities.business_fact import BusinessFactDbData, DimType
from app.entities.financial_metric import ExtractSource
from app.entities.report_document import ReportType
from app.models.base import Base


class BusinessFact(Base):
    # DDL 只有 created_at 无 updated_at：切片行写入后不改（docs/02g §3.1）
    __tablename__ = "business_facts"
    __table_args__ = (
        # uk 用 metric_name 不用 metric_code：后者可空，MySQL 对多个 NULL
        # 不去重，未映射列会重复插入（docs/02g §3.2）
        UniqueConstraint(
            "document_id",
            "dim_type",
            "dim_name",
            "metric_name",
            "metric_period",
            name="uk_fact",
        ),
        {"comment": "业务切片事实表（开放维度）"},
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
    dim_type: Mapped[str] = mapped_column(String(32), comment="切片类型枚举")
    dim_name: Mapped[str] = mapped_column(
        String(128), comment="切片值原文：煤炭采掘业/自产煤/直销/合计（不入字典）"
    )
    metric_code: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="标准化指标编码；未映射为 NULL"
    )
    metric_name: Mapped[str] = mapped_column(String(128), comment="原文列头")
    metric_period: Mapped[str] = mapped_column(String(16), comment="数值所属期间")
    metric_value: Mapped[float | None] = mapped_column(
        Numeric(24, 6), nullable=True, comment="标准化数值"
    )
    unit: Mapped[str | None] = mapped_column(
        String(16), nullable=True, comment="元/万元/亿元/%"
    )
    value_raw: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="原文数值字符串"
    )
    page_no: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="PDF 页码"
    )
    extract_source: Mapped[str] = mapped_column(
        String(16), default="table", comment="table/derived"
    )


class BusinessFactMapper:
    @staticmethod
    def to_entity(model: BusinessFact) -> BusinessFactDbData:
        return BusinessFactDbData(
            document_id=model.document_id,
            stock_code=model.stock_code,
            report_period=model.report_period,
            report_type=ReportType(model.report_type),
            dim_type=DimType(model.dim_type),
            dim_name=model.dim_name,
            metric_code=model.metric_code,
            metric_name=model.metric_name,
            metric_period=model.metric_period,
            metric_value=model.metric_value,
            unit=model.unit,
            value_raw=model.value_raw,
            page_no=model.page_no,
            extract_source=ExtractSource(model.extract_source),
        )

    @staticmethod
    def to_model(entity: BusinessFactDbData) -> BusinessFact:
        data = asdict(entity)
        data["report_type"] = entity.report_type.value
        data["dim_type"] = entity.dim_type.value
        data["extract_source"] = entity.extract_source.value
        return BusinessFact(**data)
