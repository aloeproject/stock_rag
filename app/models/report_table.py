from dataclasses import asdict

from sqlalchemy import JSON, BigInteger, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.entities.report_table import (
    ExtractStatus,
    ReportTableDbData,
    StatementScope,
    StatementType,
)
from app.models.base import Base, TimestampMixin


class ReportTable(TimestampMixin, Base):
    __tablename__ = "report_tables"
    __table_args__ = {"comment": "财报原始表格表"}

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
    page_no: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="PDF 页码"
    )
    table_title: Mapped[str | None] = mapped_column(
        String(255), nullable=True, comment="表格标题"
    )
    statement_type: Mapped[str | None] = mapped_column(
        String(16), nullable=True, comment="headline/income/balance/cashflow/equity/other"
    )
    statement_scope: Mapped[str | None] = mapped_column(
        String(16), nullable=True, comment="consolidated/parent/empty"
    )
    raw_markdown: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="MinerU Markdown 原文"
    )
    extracted_metrics: Mapped[list | None] = mapped_column(
        JSON, nullable=True, comment="结构化抽取快照"
    )
    extract_status: Mapped[str] = mapped_column(
        String(16),
        default=ExtractStatus.PENDING.value,
        comment="pending/ok/failed/skipped",
    )


class ReportTableMapper:
    @staticmethod
    def to_entity(model: ReportTable) -> ReportTableDbData:
        return ReportTableDbData(
            id=model.id,
            document_id=model.document_id,
            page_no=model.page_no,
            table_title=model.table_title,
            statement_type=StatementType(model.statement_type) if model.statement_type else None,
            statement_scope=StatementScope(model.statement_scope) if model.statement_scope else None,
            raw_markdown=model.raw_markdown,
            extracted_metrics=model.extracted_metrics,
            extract_status=ExtractStatus(model.extract_status),
        )

    @staticmethod
    def to_model(entity: ReportTableDbData) -> ReportTable:
        data = asdict(entity)
        data.pop("id", None)  # 插入时自增；uk 裁决只读 model.id
        data["statement_type"] = entity.statement_type.value if entity.statement_type else None
        data["statement_scope"] = entity.statement_scope.value if entity.statement_scope else None
        data["extract_status"] = entity.extract_status.value
        return ReportTable(**data)
