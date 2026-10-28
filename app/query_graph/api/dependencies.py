"""查询 API 依赖注入（docs/05 §5）。

约定：路由只写 Depends(get_xxx)，handler 内禁止自建 Repository/Qdrant/LLM。
为什么 LLM 从 Depends 注入而不是节点内 get_llm_client()：04 §10 规约 2
「Repository/Redis/LLM 一律从 QueryAgentContext 注入」；单测注入 FakeLLM 走同一入口。
为什么没有 build_query_agent_context 后台版：查询在请求生命周期内跑完（P95 < 5s），
session 活得到那时候；导入侧的后台版解决的是「跑几分钟、请求 session 已关」（05 §5）。
"""
from __future__ import annotations

from typing import Annotated, Generator

from fastapi import Depends
from sqlalchemy.orm import Session

from app.clients.mysql_client import MySQLClientManager
from app.clients.qdrant_client import QdrantClientManager
from app.lm.llm_utils import get_llm_client
from app.query_graph.context import QueryAgentContext
from app.query_graph.main_graph import run_query_stream
from app.repositories.business_fact_repository import BusinessFactRepository
from app.repositories.financial_metric_repository import FinancialMetricRepository
from app.repositories.metric_definition_repository import MetricDefinitionRepository
from app.repositories.stock_dict_repository import StockDictRepository


def get_db_session() -> Generator[Session, None, None]:
    # 请求级事务：成功 commit、异常 rollback，与 import_graph 一致
    session = MySQLClientManager().session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_stock_dict_repository(
    session: Annotated[Session, Depends(get_db_session)],
) -> StockDictRepository:
    return StockDictRepository(session)


def get_financial_metric_repository(
    session: Annotated[Session, Depends(get_db_session)],
) -> FinancialMetricRepository:
    return FinancialMetricRepository(session)


def get_metric_definition_repository(
    session: Annotated[Session, Depends(get_db_session)],
) -> MetricDefinitionRepository:
    return MetricDefinitionRepository(session)


def get_business_fact_repository(
    session: Annotated[Session, Depends(get_db_session)],
) -> BusinessFactRepository:
    return BusinessFactRepository(session)


def get_qdrant() -> QdrantClientManager:
    # 全局单例，与 import_graph get_qdrant 同一把连接
    return QdrantClientManager()


def get_llm():
    # 自由文本：generate 流式答案 / HyDE 假设文档。不能开 json_mode——
    # response_format=json_object 要求 prompt 含 "json"，generate_answer 是自由文本，400
    return get_llm_client(json_mode=False)


def get_json_llm():
    # json_mode=True 强制 JSON 输出：plan/intent 的 json.loads 依赖它，
    # 不开则复杂题返回空文本或前置文字（"Expecting value: line 1 column 1"）
    return get_llm_client(json_mode=True)


def get_query_agent_context(
    qdrant: Annotated[QdrantClientManager, Depends(get_qdrant)],
    stock_dict_repository: Annotated[StockDictRepository, Depends(get_stock_dict_repository)],
    financial_metric_repository: Annotated[
        FinancialMetricRepository, Depends(get_financial_metric_repository)
    ],
    metric_definition_repository: Annotated[
        MetricDefinitionRepository, Depends(get_metric_definition_repository)
    ],
    business_fact_repository: Annotated[
        BusinessFactRepository, Depends(get_business_fact_repository)
    ],
    llm: Annotated[object, Depends(get_llm)],
    json_llm: Annotated[object, Depends(get_json_llm)],
) -> QueryAgentContext:
    """装配查询图节点所需的 Runtime context（05 §5）。"""
    return QueryAgentContext(
        qdrant=qdrant,
        stock_dict_repository=stock_dict_repository,
        financial_metric_repository=financial_metric_repository,
        metric_definition_repository=metric_definition_repository,
        business_fact_repository=business_fact_repository,
        llm=llm,
        json_llm=json_llm,
    )


def get_query_service(
    context: Annotated[QueryAgentContext, Depends(get_query_agent_context)],
):
    from app.query_graph.api.services.query_service import QueryService

    return QueryService(run_query_stream=run_query_stream, context=context)
