"""API 依赖注入（参考 personal-assistant app/api/dependencies.py）。

约定：路由只写 Depends(get_xxx)，handler 内禁止自建 Repository/Qdrant 客户端。
为什么请求级 session 事务：成功 commit、异常 rollback，与 personal-assistant 一致——
避免 handler 忘记 commit 导致静默丢写。
为什么仓储用组合注入：FastAPI 同请求缓存同一 get_db_session，多个仓储共享一个
session，不会出现半套数据写进两个事务。
"""
from __future__ import annotations

from typing import Annotated, Generator

from fastapi import Depends
from sqlalchemy.orm import Session

from app.clients.mysql_client import MySQLClientManager
from app.clients.qdrant_client import QdrantClientManager
from app.import_graph.api.services.import_service import ImportService
from app.import_graph.context import DataAgentContext
from app.import_graph.progress_hub import ProgressHub
from app.repositories.business_fact_repository import BusinessFactRepository
from app.repositories.financial_metric_repository import FinancialMetricRepository
from app.repositories.metric_definition_repository import MetricDefinitionRepository
from app.repositories.report_document_repository import ReportDocumentRepository
from app.repositories.report_table_repository import ReportTableRepository
from app.repositories.stock_dict_repository import StockDictRepository
from app.import_graph.main_graph import run_ingest


def get_db_session() -> Generator[Session, None, None]:
    session = MySQLClientManager().session()
    try:
        yield session
        session.commit()  # 请求成功，提交事务
    except Exception:
        session.rollback()  # 发生异常，回滚事务
        raise
    finally:
        session.close()


def get_report_document_repository(
    session: Annotated[Session, Depends(get_db_session)],
) -> ReportDocumentRepository:
    return ReportDocumentRepository(session)


def get_report_table_repository(
    session: Annotated[Session, Depends(get_db_session)],
) -> ReportTableRepository:
    return ReportTableRepository(session)


def get_metric_definition_repository(
    session: Annotated[Session, Depends(get_db_session)],
) -> MetricDefinitionRepository:
    return MetricDefinitionRepository(session)


def get_financial_metric_repository(
    session: Annotated[Session, Depends(get_db_session)],
) -> FinancialMetricRepository:
    return FinancialMetricRepository(session)


def get_stock_dict_repository(
    session: Annotated[Session, Depends(get_db_session)],
) -> StockDictRepository:
    return StockDictRepository(session)


def get_business_fact_repository(
    session: Annotated[Session, Depends(get_db_session)],
) -> BusinessFactRepository:
    return BusinessFactRepository(session)


def get_qdrant() -> QdrantClientManager:
    # 全局单例，与 import_graph 节点 _get_qdrant 同一把连接
    return QdrantClientManager()


def get_data_agent_context(
    report_document_repository: Annotated[
        ReportDocumentRepository, Depends(get_report_document_repository)
    ],
    report_table_repository: Annotated[
        ReportTableRepository, Depends(get_report_table_repository)
    ],
    metric_definition_repository: Annotated[
        MetricDefinitionRepository, Depends(get_metric_definition_repository)
    ],
    financial_metric_repository: Annotated[
        FinancialMetricRepository, Depends(get_financial_metric_repository)
    ],
    stock_dict_repository: Annotated[
        StockDictRepository, Depends(get_stock_dict_repository)
    ],
    business_fact_repository: Annotated[
        BusinessFactRepository, Depends(get_business_fact_repository)
    ],
    qdrant: Annotated[QdrantClientManager, Depends(get_qdrant)],
) -> DataAgentContext:
    """装配 import_graph 节点所需的 Runtime context（P4 导入接口用）。"""
    return DataAgentContext(
        qdrant=qdrant,
        report_document_repository=report_document_repository,
        report_table_repository=report_table_repository,
        metric_definition_repository=metric_definition_repository,
        financial_metric_repository=financial_metric_repository,
        stock_dict_repository=stock_dict_repository,
        business_fact_repository=business_fact_repository,
    )


def build_data_agent_context() -> DataAgentContext:
    """后台工作流专用：自建 session，不复用请求级 session。

    为什么不能用 Depends(get_data_agent_context)：get_db_session 在请求结束就
    commit/close，而导入工作流要跑几分钟，session 活不到那时候。
    调用方（ImportService.start_ingest）负责在工作流结束后 close session。
    """
    session = MySQLClientManager().session()
    return DataAgentContext(
        qdrant=QdrantClientManager(),
        report_document_repository=ReportDocumentRepository(session),
        report_table_repository=ReportTableRepository(session),
        metric_definition_repository=MetricDefinitionRepository(session),
        financial_metric_repository=FinancialMetricRepository(session),
        stock_dict_repository=StockDictRepository(session),
        business_fact_repository=BusinessFactRepository(session),
    )


# 进度事件中枢全局单例：后台工作流 publish、SSE 接口 subscribe 必须是同一实例
_progress_hub = ProgressHub()


def get_progress_hub() -> ProgressHub:
    return _progress_hub


def get_import_service(
    progress_hub: Annotated[ProgressHub, Depends(get_progress_hub)],
) -> ImportService:


    return ImportService(
        run_ingest=run_ingest,
        progress_hub=progress_hub,
        context_factory=build_data_agent_context,
    )
