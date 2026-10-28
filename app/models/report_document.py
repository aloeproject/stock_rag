from dataclasses import asdict

from sqlalchemy import JSON, BigInteger, Integer, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from app.entities.report_document import ParseStatus, ReportDocumentDbData, ReportType
from app.models.base import Base, TimestampMixin


class ReportDocument(TimestampMixin, Base):
    __tablename__ = "report_documents"
    __table_args__ = {"comment": "财报文档表：元数据与解析状态"}

    document_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, comment="文档唯一ID，取 PDF 文件内容 sha1"
    )
    stock_code: Mapped[str] = mapped_column(
        String(16), index=True, comment="股票代码，6位"
    )
    stock_name: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="股票简称"
    )
    report_type: Mapped[str] = mapped_column(
        String(16), comment="报告类型：annual/q1/q2/q3/interim/prospectus"
    )
    report_period: Mapped[str] = mapped_column(
        String(16), comment="报告期 YYYY 或 YYYYQn"
    )
    fiscal_year: Mapped[int] = mapped_column(SmallInteger, comment="会计年度")
    source_file: Mapped[str | None] = mapped_column(
        String(255), nullable=True, comment="原始上传文件名"
    )
    parse_status: Mapped[str] = mapped_column(
        String(24),
        default=ParseStatus.PENDING.value,
        comment="解析状态：pending/parsing/splitting/embedding/extracting/done/failed(part)/failed",
    )
    stats: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, comment="阶段统计：chunks/tables_ok/metrics_rows/unmapped_names"
    )


class ReportDocumentMapper:
    @staticmethod
    def to_entity(model: ReportDocument) -> ReportDocumentDbData:
        return ReportDocumentDbData(
            document_id=model.document_id,
            stock_code=model.stock_code,
            stock_name=model.stock_name,
            report_type=ReportType(model.report_type),
            report_period=model.report_period,
            fiscal_year=model.fiscal_year,
            source_file=model.source_file,
            parse_status=ParseStatus(model.parse_status),
            stats=model.stats,
        )

    @staticmethod
    def to_model(entity: ReportDocumentDbData) -> ReportDocument:
        data = asdict(entity)
        data["report_type"] = entity.report_type.value
        data["parse_status"] = entity.parse_status.value
        return ReportDocument(**data)
