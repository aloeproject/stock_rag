from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.entities.report_document import ParseStatus, ReportDocumentDbData
from app.models.report_document import ReportDocument, ReportDocumentMapper


class ReportDocumentRepository:
    def __init__(self, session: Session):
        self.session = session

    def upsert(self, entity: ReportDocumentDbData) -> None:
        """按 document_id 覆盖更新——同 PDF 重跑不插双份（docs/02 §4）。"""
        model = self.session.get(ReportDocument, entity.document_id)
        if model is None:
            self.session.add(ReportDocumentMapper.to_model(entity))
            return
        model.stock_code = entity.stock_code
        model.stock_name = entity.stock_name
        model.report_type = entity.report_type.value
        model.report_period = entity.report_period
        model.fiscal_year = entity.fiscal_year
        model.source_file = entity.source_file
        model.parse_status = entity.parse_status.value
        model.stats = entity.stats

    def get_by_id(self, document_id: str) -> ReportDocumentDbData | None:
        model = self.session.get(ReportDocument, document_id)
        return ReportDocumentMapper.to_entity(model) if model else None

    def update_parse_status(
        self,
        document_id: str,
        parse_status: ParseStatus,
        stats: dict | None = None,
    ) -> None:
        model = self.session.get(ReportDocument, document_id)
        if model is None:
            return
        model.parse_status = parse_status.value
        if stats is not None:
            model.stats = stats

    def count_by_stock(self, stock_code: str) -> int:
        stmt = select(func.count()).where(ReportDocument.stock_code == stock_code)
        return self.session.execute(stmt).scalar() or 0
