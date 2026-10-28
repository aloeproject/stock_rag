from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.entities.report_table import ExtractStatus, ReportTableDbData
from app.models.report_table import ReportTable, ReportTableMapper


class ReportTableRepository:
    def __init__(self, session: Session):
        self.session = session

    def insert(self, entity: ReportTableDbData) -> int:
        model = ReportTableMapper.to_model(entity)
        self.session.add(model)
        self.session.flush()
        return model.id

    def delete_by_document(self, document_id: str) -> int:
        """重导先删后写：表数量变化时只 upsert 会留残行（docs/02d §5.1）。"""
        result = self.session.execute(
            delete(ReportTable).where(ReportTable.document_id == document_id)
        )
        return result.rowcount or 0

    def list_by_document(self, document_id: str) -> list[ReportTableDbData]:
        stmt = (
            select(ReportTable)
            .where(ReportTable.document_id == document_id)
            .order_by(ReportTable.id.asc())
        )
        rows = self.session.execute(stmt).scalars().all()
        return [ReportTableMapper.to_entity(m) for m in rows]

    def list_ok_by_document(self, document_id: str) -> list[ReportTableDbData]:
        stmt = (
            select(ReportTable)
            .where(
                ReportTable.document_id == document_id,
                ReportTable.extract_status == ExtractStatus.OK.value,
            )
            .order_by(ReportTable.id.asc())
        )
        rows = self.session.execute(stmt).scalars().all()
        return [ReportTableMapper.to_entity(m) for m in rows]

    def list_pending(self, document_id: str) -> list[ReportTableDbData]:
        stmt = select(ReportTable).where(
            ReportTable.document_id == document_id,
            ReportTable.extract_status == ExtractStatus.PENDING.value,
        )
        rows = self.session.execute(stmt).scalars().all()
        return [ReportTableMapper.to_entity(m) for m in rows]

    def update_extract(
        self,
        table_id: int,
        status: ExtractStatus,
        extracted_metrics: list[dict] | None = None,
        statement_scope: str | None = None,
    ) -> None:
        model = self.session.get(ReportTable, table_id)
        if model is None:
            return
        model.extract_status = status.value
        if extracted_metrics is not None:
            model.extracted_metrics = extracted_metrics
        if statement_scope:
            model.statement_scope = statement_scope
