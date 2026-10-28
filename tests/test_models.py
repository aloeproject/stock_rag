"""models 层测试：ORM 字段/表名对齐 DDL，Mapper 往返不丢字段。"""
from datetime import datetime
from decimal import Decimal

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
from app.models.financial_metric import FinancialMetric, FinancialMetricMapper
from app.models.metric_definition import MetricDefinition, MetricDefinitionMapper
from app.models.report_document import ReportDocument, ReportDocumentMapper
from app.models.report_table import ReportTable, ReportTableMapper


def test_tablenames_match_ddl():
    assert ReportDocument.__tablename__ == "report_documents"
    assert ReportTable.__tablename__ == "report_tables"
    assert FinancialMetric.__tablename__ == "financial_metrics"
    assert MetricDefinition.__tablename__ == "metric_definitions"


def test_all_models_share_base():
    for m in (ReportDocument, ReportTable, FinancialMetric, MetricDefinition):
        assert issubclass(m, Base)


def test_report_document_mapper_roundtrip():
    doc = ReportDocumentDbData(
        document_id="abc123",
        stock_code="000006",
        stock_name="深振业A",
        report_type=ReportType.ANNUAL,
        report_period="2024",
        fiscal_year=2024,
        source_file="x.pdf",
        parse_status=ParseStatus.DONE,
        stats={"chunks": 10},
    )
    model = ReportDocumentMapper.to_model(doc)
    assert model.document_id == "abc123"
    assert model.report_type == "annual"
    assert model.parse_status == "done"
    assert model.stats == {"chunks": 10}

    entity = ReportDocumentMapper.to_entity(model)
    assert entity == doc


def test_report_table_mapper_roundtrip():
    t = ReportTableDbData(
        document_id="abc123",
        page_no=68,
        table_title="合并资产负债表",
        statement_type=StatementType.BALANCE,
        statement_scope=StatementScope.CONSOLIDATED,
        raw_markdown="| a |",
        extracted_metrics=[{"metric_name": "货币资金", "value_raw": "1"}],
        extract_status=ExtractStatus.OK,
    )
    model = ReportTableMapper.to_model(t)
    assert model.extract_status == "ok"
    entity = ReportTableMapper.to_entity(model)
    assert entity == t


def test_financial_metric_mapper_roundtrip():
    m = FinancialMetricDbData(
        document_id="abc123",
        stock_code="000006",
        report_period="2024",
        report_type=ReportType.ANNUAL,
        metric_period="2024",
        metric_period_type=MetricPeriodType.CURRENT,
        statement_type=StatementType.HEADLINE,
        statement_scope=StatementScope.CONSOLIDATED,
        metric_code="revenue",
        metric_name="营业收入",
        metric_value=Decimal("6064948157.84"),
        unit="元",
        value_raw="6,064,948,157.84",
        is_restated=0,
        page_no=45,
        extract_source=ExtractSource.HEADLINE,
    )
    model = FinancialMetricMapper.to_model(m)
    assert model.metric_code == "revenue"
    assert model.metric_value == Decimal("6064948157.84")
    entity = FinancialMetricMapper.to_entity(model)
    assert entity == m


def test_metric_definition_mapper_roundtrip():
    d = MetricDefinitionDbData(
        metric_code="gross_margin",
        metric_name="毛利率",
        aliases="毛利",
        category=MetricCategory.PROFITABILITY,
        statement_type=StatementType.RATIO,
        metric_level=MetricLevel.L2,
        calc_formula="(revenue-cost_of_revenue)/revenue*100",
        unit_default="%",
        description="综合毛利率",
    )
    model = MetricDefinitionMapper.to_model(d)
    assert model.metric_level == 2
    entity = MetricDefinitionMapper.to_entity(model)
    assert entity == d


def test_report_document_nullable_fields():
    doc = ReportDocumentDbData(
        document_id="x",
        stock_code="000001",
        stock_name=None,
        report_type=ReportType.ANNUAL,
        report_period="2024",
        fiscal_year=2024,
        source_file=None,
        parse_status=ParseStatus.PENDING,
        stats=None,
    )
    entity = ReportDocumentMapper.to_entity(ReportDocumentMapper.to_model(doc))
    assert entity == doc
