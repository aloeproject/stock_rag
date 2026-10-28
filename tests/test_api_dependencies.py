"""api dependencies 测试：请求级 session 事务、仓储注入、DataAgentContext 装配。

路由只写 Depends(get_xxx)，handler 内不自建 Repository——与 personal-assistant 同规。
"""
from typing import Annotated

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.import_graph.api.dependencies import (
    get_data_agent_context,
    get_db_session,
    get_qdrant,
    get_report_document_repository,
)
from app.entities.report_document import ParseStatus, ReportDocumentDbData, ReportType
from app.import_graph.context import DataAgentContext
from app.models.base import Base
from app.repositories.report_document_repository import ReportDocumentRepository


@pytest.fixture()
def sqlite_session_factory():
    # StaticPool：TestClient 跨线程取连接时共用同一内存库，否则看不到表
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture()
def client(sqlite_session_factory):
    """用 sqlite 覆盖 get_db_session，验证 DI 装配，不连真实 MySQL。"""
    app = FastAPI()

    def _override_session():
        s = sqlite_session_factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    app.dependency_overrides[get_db_session] = _override_session

    @app.post("/_probe/document")
    def probe(
        repo: Annotated[ReportDocumentRepository, Depends(get_report_document_repository)],
    ):
        repo.upsert(
            ReportDocumentDbData(
                document_id="doc1",
                stock_code="600628",
                stock_name="新世界",
                report_type=ReportType.ANNUAL,
                report_period="2024",
                fiscal_year=2024,
                source_file="600628.pdf",
                parse_status=ParseStatus.DONE,
                stats={"metrics_rows": 1},
            )
        )
        doc = repo.get_by_id("doc1")
        return {"found": doc is not None, "status": doc.parse_status.value}

    @app.post("/_probe/context")
    def probe_ctx(
        ctx: Annotated[DataAgentContext, Depends(get_data_agent_context)],
    ):
        return {
            "has_doc_repo": ctx.report_document_repository is not None,
            "has_table_repo": ctx.report_table_repository is not None,
            "has_def_repo": ctx.metric_definition_repository is not None,
            "has_metric_repo": ctx.financial_metric_repository is not None,
        }

    with TestClient(app) as c:
        yield c


def test_route_injects_repository_and_commits(client):
    """路由 Depends 注入仓储；请求成功 session 落库（个人助理同规）。"""
    r = client.post("/_probe/document")
    assert r.status_code == 200
    assert r.json() == {"found": True, "status": "done"}


def test_route_injects_data_agent_context(client):
    r = client.post("/_probe/context")
    assert r.status_code == 200
    assert r.json() == {
        "has_doc_repo": True,
        "has_table_repo": True,
        "has_def_repo": True,
        "has_metric_repo": True,
    }


def test_providers_construct_repos(sqlite_session_factory):
    session = sqlite_session_factory()
    try:
        doc_repo = get_report_document_repository(session)
        assert isinstance(doc_repo, ReportDocumentRepository)
        assert doc_repo.session is session
    finally:
        session.close()


def test_get_qdrant_is_singleton_manager():
    a = get_qdrant()
    b = get_qdrant()
    assert a is b
