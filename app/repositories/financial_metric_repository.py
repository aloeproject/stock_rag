from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.entities.financial_metric import FinancialMetricDbData
from app.models.financial_metric import FinancialMetric, FinancialMetricMapper


class FinancialMetricRepository:
    def __init__(self, session: Session):
        self.session = session

    def delete_by_document(self, document_id: str) -> int:
        """重跑先删后写：指标集合可能变小，只 upsert 会留残行（docs/02e §8.3）。"""
        result = self.session.execute(
            delete(FinancialMetric).where(FinancialMetric.document_id == document_id)
        )
        return result.rowcount or 0

    def upsert(self, entity: FinancialMetricDbData) -> None:
        """uk_metric 冲突则覆盖（docs/02 §S6 幂等）。

        不用 INSERT..ON DUPLICATE：SQLite 单测跑不通，且 L2 物化写入量不大。
        """
        stmt = select(FinancialMetric).where(
            FinancialMetric.document_id == entity.document_id,
            FinancialMetric.metric_code == entity.metric_code,
            FinancialMetric.statement_scope == entity.statement_scope.value,
            FinancialMetric.metric_period == entity.metric_period,
            FinancialMetric.metric_period_type == entity.metric_period_type.value,
        )
        model = self.session.execute(stmt).scalars().first()
        if model is None:
            self.session.add(FinancialMetricMapper.to_model(entity))
            return
        model.metric_name = entity.metric_name
        model.metric_value = entity.metric_value
        model.unit = entity.unit
        model.value_raw = entity.value_raw
        model.is_restated = entity.is_restated
        model.page_no = entity.page_no
        model.extract_source = entity.extract_source.value
        model.report_type = entity.report_type.value
        model.stock_code = entity.stock_code
        model.report_period = entity.report_period

    def list_by_document(self, document_id: str) -> list[FinancialMetricDbData]:
        stmt = (
            select(FinancialMetric)
            .where(FinancialMetric.document_id == document_id)
            .order_by(FinancialMetric.id.asc())
        )
        rows = self.session.execute(stmt).scalars().all()
        return [FinancialMetricMapper.to_entity(m) for m in rows]

    def list_by_scope(
        self, document_id: str, statement_scope: str
    ) -> list[FinancialMetricDbData]:
        stmt = select(FinancialMetric).where(
            FinancialMetric.document_id == document_id,
            FinancialMetric.statement_scope == statement_scope,
        )
        rows = self.session.execute(stmt).scalars().all()
        return [FinancialMetricMapper.to_entity(m) for m in rows]

    def latest_report_period(self) -> str | None:
        """最新可查报告期——periods 绝对化基准（04a §5.1，不是自然年）。"""
        from sqlalchemy import func

        stmt = select(func.max(FinancialMetric.report_period))
        value = self.session.execute(stmt).scalar()
        return str(value) if value is not None else None

    def query_by_codes(
        self,
        stock_codes: list[str],
        metric_codes: list[str],
        metric_periods: list[str],
        report_types: list[str],
    ) -> list[dict]:
        """查询侧取数（docs/04f）：IN 精确过滤，全量返给节点做口径裁决。

        不在 WHERE 里写死 statement_scope/metric_period_type——母公司独有指标不能查空，
        裁决优先级在节点按 04f §5.1 处理。
        """
        if not stock_codes or not metric_codes:
            return []
        from sqlalchemy import and_

        conds = [
            FinancialMetric.stock_code.in_(stock_codes),
            FinancialMetric.metric_code.in_(metric_codes),
        ]
        if metric_periods:
            conds.append(FinancialMetric.metric_period.in_(metric_periods))
        if report_types:
            conds.append(FinancialMetric.report_type.in_(report_types))

        stmt = select(FinancialMetric).where(and_(*conds))
        models = self.session.execute(stmt).scalars().all()
        return [
            {
                "id": m.id,
                "stock_code": m.stock_code,
                "metric_code": m.metric_code,
                "metric_name": m.metric_name,
                "metric_period": m.metric_period,
                "metric_period_type": m.metric_period_type,
                "metric_value": m.metric_value,
                "unit": m.unit,
                "statement_scope": m.statement_scope,
                "page_no": m.page_no,
            }
            for m in models
        ]
