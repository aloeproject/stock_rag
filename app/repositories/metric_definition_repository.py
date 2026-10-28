from sqlalchemy import select
from sqlalchemy.orm import Session

from app.entities.metric_definition import (
    MetricDefinitionDbData,
    MetricLevel,
)
from app.models.metric_definition import MetricDefinition, MetricDefinitionMapper


class MetricDefinitionRepository:
    def __init__(self, session: Session):
        self.session = session

    def upsert(self, entity: MetricDefinitionDbData) -> None:
        model = self.session.get(MetricDefinition, entity.metric_code)
        if model is None:
            self.session.add(MetricDefinitionMapper.to_model(entity))
            return
        model.metric_name = entity.metric_name
        model.aliases = entity.aliases
        model.category = entity.category.value if entity.category else None
        model.statement_type = (
            entity.statement_type.value if entity.statement_type else None
        )
        model.metric_level = int(entity.metric_level)
        model.calc_formula = entity.calc_formula
        model.unit_default = entity.unit_default
        model.description = entity.description

    def get_by_code(self, metric_code: str) -> MetricDefinitionDbData | None:
        model = self.session.get(MetricDefinition, metric_code)
        return MetricDefinitionMapper.to_entity(model) if model else None

    def find_by_alias(self, name: str) -> MetricDefinitionDbData | None:
        """按 metric_name 或 aliases（竖线分隔）精确匹配。

        不用 LIKE 包含匹配：「净利润」会误中「归属于上市公司股东的净利润」。
        """
        stmt = select(MetricDefinition)
        for model in self.session.execute(stmt).scalars().all():
            if model.metric_name == name:
                return MetricDefinitionMapper.to_entity(model)
            aliases = (model.aliases or "").split("|")
            if name in aliases:
                return MetricDefinitionMapper.to_entity(model)
        return None

    def list_all(self) -> list[MetricDefinitionDbData]:
        stmt = select(MetricDefinition)
        rows = self.session.execute(stmt).scalars().all()
        return [MetricDefinitionMapper.to_entity(m) for m in rows]

    def list_by_level(self, level: MetricLevel) -> list[MetricDefinitionDbData]:
        stmt = select(MetricDefinition).where(
            MetricDefinition.metric_level == int(level)
        )
        rows = self.session.execute(stmt).scalars().all()
        return [MetricDefinitionMapper.to_entity(m) for m in rows]
