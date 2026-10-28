from __future__ import annotations

import logging
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.conf.mysql_config import MysqlConfig, mysql_config

# 配置 SQLAlchemy 日志 - 打印所有 SQL 语句
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)



class MySQLClientManager:
    """MySQL 连接单例：全局共享 engine / session_factory，禁止各处散建连接。

    为什么单例：import_graph 多节点并发写同一库，连接池必须共享否则
    连接数失控；参考 personal-assistant 的 MySQLClientManager。
    用同步 SQLAlchemy：langgraph 节点是同步函数，async 反而要额外套 asyncio。
    """

    _instance: MySQLClientManager | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.engine = None
            cls._instance.session_factory = None
        return cls._instance

    @staticmethod
    def build_url(cfg: MysqlConfig) -> str:
        return (
            f"mysql+pymysql://{cfg.user}:{cfg.password}"
            f"@{cfg.host}:{cfg.port}/{cfg.database}?charset=utf8mb4"
        )

    def init(self, cfg: MysqlConfig | None = None) -> None:
        cfg = cfg or mysql_config
        self.engine = create_engine(
            self.build_url(cfg),
            pool_size=10,
            pool_pre_ping=True,
        )
        self.session_factory = sessionmaker(
            bind=self.engine, autoflush=True, expire_on_commit=False
        )

    def session(self) -> Session:
        if self.session_factory is None:
            self.init()
        return self.session_factory()

    def close(self) -> None:
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None
            self.session_factory = None
