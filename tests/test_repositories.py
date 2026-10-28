"""repositories 层测试：SQLite 内存库跑 CRUD，覆盖导入图读写路径。"""
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.entities.financial_metric import (
    ExtractSource,
    FinancialMetricDbData,
    MetricPeriodType,
)
from app.entities.metric_definition import (
    MetricCategory,
    MetricDefinitionDbData,
    MetricLevel,
)
from app.entities.report_document import ParseStatus, ReportDocumentDbData, ReportType
from app.entities.report_table import (
    ExtractStatus,
    ReportTableDbData,
    StatementScope,
    StatementType,
)
from app.models.base import Base
from app.repositories.financial_metric_repository import FinancialMetricRepository
from app.repositories.metric_definition_repository import MetricDefinitionRepository
from app.repositories.report_document_repository import ReportDocumentRepository
from app.repositories.report_table_repository import ReportTableRepository


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    yield s
    s.close()


def _doc(**kw) -> ReportDocumentDbData:
    base = dict(
        document_id="doc1",
        stock_code="000006",
        stock_name="深振业A",
        report_type=ReportType.ANNUAL,
        report_period="2024",
        fiscal_year=2024,
        source_file="a.pdf",
        parse_status=ParseStatus.PENDING,
        stats=None,
    )
    base.update(kw)
    return ReportDocumentDbData(**base)


def test_report_document_upsert_and_get(session):
    repo = ReportDocumentRepository(session)
    repo.upsert(_doc())
    session.commit()

    got = repo.get_by_id("doc1")
    assert got is not None
    assert got.stock_code == "000006"
    assert got.parse_status == ParseStatus.PENDING

    # 同 document_id 再 upsert 是覆盖不是插双份（docs/02 §4 幂等）
    repo.upsert(_doc(parse_status=ParseStatus.DONE, stats={"chunks": 9}))
    session.commit()
    got = repo.get_by_id("doc1")
    assert got.parse_status == ParseStatus.DONE
    assert got.stats == {"chunks": 9}
    assert repo.count_by_stock("000006") == 1


def test_report_table_insert_and_update_status(session):
    repo = ReportTableRepository(session)
    t = ReportTableDbData(
        document_id="doc1",
        page_no=68,
        table_title="合并资产负债表",
        statement_type=StatementType.BALANCE,
        statement_scope=StatementScope.CONSOLIDATED,
        raw_markdown="| a |",
        extracted_metrics=None,
        extract_status=ExtractStatus.PENDING,
    )
    tid = repo.insert(t)
    session.commit()
    assert tid > 0

    rows = repo.list_by_document("doc1")
    assert len(rows) == 1
    assert rows[0].extract_status == ExtractStatus.PENDING

    repo.update_extract(tid, ExtractStatus.OK, [{"metric_name": "货币资金"}])
    session.commit()
    rows = repo.list_by_document("doc1")
    assert rows[0].extract_status == ExtractStatus.OK
    assert rows[0].extracted_metrics == [{"metric_name": "货币资金"}]


def test_financial_metric_upsert_unique_key(session):
    repo = FinancialMetricRepository(session)
    m = FinancialMetricDbData(
        document_id="doc1",
        stock_code="000006",
        report_period="2024",
        report_type=ReportType.ANNUAL,
        metric_period="2024",
        metric_period_type=MetricPeriodType.CURRENT,
        statement_type=StatementType.HEADLINE,
        statement_scope=StatementScope.CONSOLIDATED,
        metric_code="revenue",
        metric_name="营业收入",
        metric_value=Decimal("100"),
        unit="元",
        value_raw="100",
        is_restated=0,
        page_no=45,
        extract_source=ExtractSource.HEADLINE,
    )
    repo.upsert(m)
    session.commit()

    # 同 uk_metric 再写是覆盖（docs/02 §S6）
    m2 = FinancialMetricDbData(**{**m.__dict__, "metric_value": Decimal("200")})
    repo.upsert(m2)
    session.commit()

    rows = repo.list_by_document("doc1")
    assert len(rows) == 1
    assert rows[0].metric_value == Decimal("200")

    # 不同 metric_code 是另一行
    m3 = FinancialMetricDbData(**{**m.__dict__, "metric_code": "net_profit"})
    repo.upsert(m3)
    session.commit()
    assert len(repo.list_by_document("doc1")) == 2


def test_metric_definition_read_paths(session):
    repo = MetricDefinitionRepository(session)
    repo.upsert(
        MetricDefinitionDbData(
            metric_code="revenue",
            metric_name="营业收入",
            aliases="营业总收入",
            category=MetricCategory.HEADLINE,
            statement_type=StatementType.HEADLINE,
            metric_level=MetricLevel.L1,
            calc_formula=None,
            unit_default="元",
            description=None,
        )
    )
    session.commit()

    assert repo.get_by_code("revenue") is not None
    assert repo.find_by_alias("营业总收入") is not None
    assert repo.find_by_alias("营业收入") is not None
    assert repo.find_by_alias("毛利") is None

    l2 = repo.list_by_level(MetricLevel.L1)
    assert len(l2) == 1


def _fact(**kw) -> "BusinessFactDbData":
    from app.entities.business_fact import BusinessFactDbData, DimType
    from app.entities.financial_metric import ExtractSource

    base = dict(
        document_id="doc1",
        stock_code="601225",
        report_period="2024",
        report_type=ReportType.ANNUAL,
        dim_type=DimType.INDUSTRY,
        dim_name="煤炭采掘业",
        metric_code="revenue",
        metric_name="营业收入",
        metric_period="2024",
        metric_value=Decimal("100"),
        unit="元",
        value_raw="100",
        page_no=12,
        extract_source=ExtractSource.TABLE,
    )
    base.update(kw)
    return BusinessFactDbData(**base)


def test_business_fact_upsert_uk_uses_metric_name(session):
    """uk_fact 用 metric_name 不用 metric_code：后者可空，MySQL NULL 不去重（docs/02g §3.2）。"""
    from app.repositories.business_fact_repository import BusinessFactRepository

    repo = BusinessFactRepository(session)
    repo.upsert(_fact())
    session.commit()
    assert len(repo.list_by_document("doc1")) == 1

    # 同 uk 覆盖
    repo.upsert(_fact(metric_value=Decimal("200")))
    session.commit()
    rows = repo.list_by_document("doc1")
    assert len(rows) == 1 and rows[0].metric_value == Decimal("200")

    # metric_code=NULL 的不同 metric_name 是两行（未映射列头可并存）
    repo.upsert(_fact(metric_code=None, metric_name="毛利率(%)", metric_value=Decimal("30")))
    repo.upsert(_fact(metric_code=None, metric_name="营业收入比上年增减(%)", metric_value=Decimal("-1")))
    session.commit()
    assert len(repo.list_by_document("doc1")) == 3

    # 同 dim_name 不同 dim_type 是两个事实（分行业「电力」≠分产品「电力」）
    from app.entities.business_fact import DimType

    repo.upsert(_fact(dim_type=DimType.PRODUCT, dim_name="电力"))
    repo.upsert(_fact(dim_type=DimType.INDUSTRY, dim_name="电力"))
    session.commit()
    assert len(repo.list_by_document("doc1")) == 5


def test_business_fact_query_by_dim_names(session):
    from app.entities.business_fact import DimType
    from app.repositories.business_fact_repository import BusinessFactRepository

    repo = BusinessFactRepository(session)
    repo.upsert(_fact(dim_name="煤炭采掘业"))
    repo.upsert(_fact(dim_name="合计", metric_value=Decimal("1000")))
    repo.upsert(_fact(dim_type=DimType.REGION, dim_name="西北"))
    session.commit()

    rows = repo.query_by_dim_names(["601225"], ["煤炭采掘业", "合计"], ["2024"], ["annual"])
    assert {r["dim_name"] for r in rows} == {"煤炭采掘业", "合计"}
    assert all(r["dim_type"] == "industry" for r in rows)

    # 空 stock_codes 不打库；空 dim_names = 该股票全部切片行（节点做包含匹配需要全集）
    assert repo.query_by_dim_names([], ["合计"], [], []) == []
    all_rows = repo.query_by_dim_names(["601225"], [], [], [])
    assert {r["dim_name"] for r in all_rows} == {"煤炭采掘业", "合计", "西北"}
