"""stock_dict ORM 模型（docs/04 §4.1 DDL）。

uk_code + idx_name；aliases 存 JSON。
为什么本地表：MCP search_stock_symbols 实测不可用（04a §1.1）。
"""
from __future__ import annotations

from sqlalchemy import JSON, BigInteger, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.entities.stock_dict import StockDictDbData
from app.models.base import Base, TimestampMixin


class StockDict(TimestampMixin, Base):
    __tablename__ = "stock_dict"
    __table_args__ = (
        UniqueConstraint("stock_code", name="uk_code"),
        {"comment": "股票字典表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="自增主键",
    )
    stock_code: Mapped[str] = mapped_column(String(16), comment="6位代码，如 600519")
    stock_name: Mapped[str] = mapped_column(String(64), index=True, comment="标准简称")
    full_name: Mapped[str] = mapped_column(String(128), default="", comment="全称")
    aliases: Mapped[list | None] = mapped_column(JSON, default=list, comment="别名列表")
    market: Mapped[str] = mapped_column(String(8), default="", comment="SH/SZ/BJ")
    status: Mapped[int] = mapped_column(Integer, default=1, comment="1=正常 0=停用")


class StockDictMapper:
    @staticmethod
    def to_model(entity: StockDictDbData) -> StockDict:
        return StockDict(
            stock_code=entity.stock_code,
            stock_name=entity.stock_name,
            full_name=entity.full_name or "",
            aliases=list(entity.aliases or []),
            market=entity.market or "",
            status=entity.status,
        )

    @staticmethod
    def to_entity(model: StockDict) -> StockDictDbData:
        return StockDictDbData(
            stock_code=model.stock_code,
            stock_name=model.stock_name,
            full_name=model.full_name or "",
            aliases=list(model.aliases or []),
            market=model.market or "",
            status=model.status,
        )
