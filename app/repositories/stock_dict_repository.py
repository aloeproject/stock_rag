"""stock_dict 仓储（docs/04 §4.1 / 04a §4.3）。

upsert 幂等：同 stock_code 覆盖（SQLite 单测走 select 后改写，
与 FinancialMetricRepository.upsert 同策略）。
前缀级匹配在内存做（字典量级 ≤ 全 A 股 6k 行），故提供 list_all。
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.entities.stock_dict import StockDictDbData
from app.models.stock_dict import StockDict, StockDictMapper


class StockDictRepository:
    def __init__(self, session: Session):
        self.session = session

    def upsert(self, entity: StockDictDbData) -> None:
        stmt = select(StockDict).where(StockDict.stock_code == entity.stock_code)
        model = self.session.execute(stmt).scalars().first()
        if model is None:
            self.session.add(StockDictMapper.to_model(entity))
            return
        model.stock_name = entity.stock_name
        model.full_name = entity.full_name or ""
        model.aliases = list(entity.aliases or [])
        model.market = entity.market or ""
        model.status = entity.status

    def list_all(self) -> list[StockDictDbData]:
        """全量字典（含停用）；四级匹配自己过滤 status（04a §4.3）。"""
        rows = self.session.execute(select(StockDict).order_by(StockDict.id.asc())).scalars()
        return [StockDictMapper.to_entity(m) for m in rows]
