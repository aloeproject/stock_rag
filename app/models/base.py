from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now():
    return datetime.now(ZoneInfo("Asia/Shanghai"))


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_now, comment="创建时间"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_now, onupdate=_now, comment="最后更新时间"
    )
