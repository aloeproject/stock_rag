"""business_facts 仓储（docs/02g）。

uk 取舍：uk_fact 用 metric_name 不用 metric_code——后者可空，MySQL
唯一索引对多个 NULL 不去重，未映射列会重复插入（docs/02g §3.2）。
"""
from __future__ import annotations

from sqlalchemy import and_, delete, select
from sqlalchemy.orm import Session

from app.entities.business_fact import BusinessFactDbData
from app.models.business_fact import BusinessFact, BusinessFactMapper


class BusinessFactRepository:
    def __init__(self, session: Session):
        self.session = session

    def delete_by_document(self, document_id: str) -> int:
        """重跑先删后写：切片集合可能变小，只 upsert 会留残行（docs/02g §6.3）。

        只删 business_facts 不动 financial_metrics——两表独立生命周期。
        """
        result = self.session.execute(
            delete(BusinessFact).where(BusinessFact.document_id == document_id)
        )
        return result.rowcount or 0

    def upsert(self, entity: BusinessFactDbData) -> None:
        """uk_fact 冲突则覆盖。

        不用 INSERT..ON DUPLICATE：SQLite 单测跑不通，与 FinancialMetricRepository
        同策略。
        """
        stmt = select(BusinessFact).where(
            BusinessFact.document_id == entity.document_id,
            BusinessFact.dim_type == entity.dim_type.value,
            BusinessFact.dim_name == entity.dim_name,
            BusinessFact.metric_name == entity.metric_name,
            BusinessFact.metric_period == entity.metric_period,
        )
        model = self.session.execute(stmt).scalars().first()
        if model is None:
            self.session.add(BusinessFactMapper.to_model(entity))
            return
        model.metric_code = entity.metric_code
        model.metric_value = entity.metric_value
        model.unit = entity.unit
        model.value_raw = entity.value_raw
        model.page_no = entity.page_no
        model.extract_source = entity.extract_source.value
        model.report_type = entity.report_type.value
        model.stock_code = entity.stock_code
        model.report_period = entity.report_period

    def list_by_document(self, document_id: str) -> list[BusinessFactDbData]:
        stmt = (
            select(BusinessFact)
            .where(BusinessFact.document_id == document_id)
            .order_by(BusinessFact.id.asc())
        )
        rows = self.session.execute(stmt).scalars().all()
        return [BusinessFactMapper.to_entity(m) for m in rows]

    def query_by_dim_names(
        self,
        stock_codes: list[str],
        dim_names: list[str],
        metric_periods: list[str],
        report_types: list[str],
    ) -> list[dict]:
        """查询侧业务词取数（docs/04f §3.6）：dim_name 精确/包含由节点裁，这里 IN 过滤。

        dim_names 空 = 该股票全部切片行（节点做包含匹配需要全集）。
        返回 dict 而不是 entity：与 FinancialMetricRepository.query_by_codes 同风格，
        节点拼 metric_rows 时少一层转换。
        """
        if not stock_codes:
            return []
        conds = [BusinessFact.stock_code.in_(stock_codes)]
        if dim_names:
            conds.append(BusinessFact.dim_name.in_(dim_names))
        if metric_periods:
            conds.append(BusinessFact.metric_period.in_(metric_periods))
        if report_types:
            conds.append(BusinessFact.report_type.in_(report_types))

        stmt = select(BusinessFact).where(and_(*conds))
        models = self.session.execute(stmt).scalars().all()
        return [
            {
                "id": m.id,
                "stock_code": m.stock_code,
                "metric_code": m.metric_code,
                "metric_name": m.metric_name,
                "metric_period": m.metric_period,
                "metric_value": m.metric_value,
                "unit": m.unit,
                "dim_type": m.dim_type,
                "dim_name": m.dim_name,
                "page_no": m.page_no,
            }
            for m in models
        ]
