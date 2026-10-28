"""entities 层测试：枚举取值与 DbData 字段对齐 docs/01 §5.1 DDL。"""
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


def test_report_type_values():
    assert ReportType.ANNUAL == "annual"
    assert ReportType.INTERIM == "interim"
    assert {m.value for m in ReportType} == {
        "annual", "q1", "q2", "q3", "interim", "prospectus",
    }


def test_parse_status_values():
    assert ParseStatus.PENDING == "pending"
    assert ParseStatus.DONE == "done"
    assert ParseStatus.FAILED_PART == "failed(part)"
    assert ParseStatus.FAILED == "failed"
    # P4 任务状态机映射用（docs/03 §7）
    assert ParseStatus.PARSING == "parsing"
    assert ParseStatus.SPLITTING == "splitting"
    assert ParseStatus.EMBEDDING == "embedding"
    assert ParseStatus.EXTRACTING == "extracting"


def test_statement_type_and_scope():
    assert StatementType.HEADLINE == "headline"
    assert StatementType.OTHER == "other"
    assert StatementScope.CONSOLIDATED == "consolidated"
    assert StatementScope.PARENT == "parent"
    assert StatementScope.EMPTY == "empty"


def test_metric_period_type_and_extract_source():
    assert MetricPeriodType.CURRENT == "current"
    assert MetricPeriodType.COMPARATIVE == "comparative"
    assert MetricPeriodType.RESTATED == "restated"
    assert ExtractSource.HEADLINE == "headline"
    assert ExtractSource.TABLE == "table"
    assert ExtractSource.DERIVED == "derived"
    assert ExtractStatus.PENDING == "pending"
    assert ExtractStatus.OK == "ok"
    assert ExtractStatus.FAILED == "failed"


def test_metric_level_is_int():
    assert MetricLevel.L1 == 1
    assert MetricLevel.L2 == 2
    assert MetricLevel.L3 == 3
    assert isinstance(MetricLevel.L2, int)


def test_report_document_db_data_fields():
    doc = ReportDocumentDbData(
        document_id="abc123",
        stock_code="000006",
        stock_name="深振业A",
        report_type=ReportType.ANNUAL,
        report_period="2024",
        fiscal_year=2024,
        source_file="000006_2024_....pdf",
        parse_status=ParseStatus.PENDING,
        stats={"chunks": 1},
    )
    assert doc.document_id == "abc123"
    assert doc.report_type == ReportType.ANNUAL
    assert doc.parse_status == ParseStatus.PENDING
    assert doc.stats == {"chunks": 1}


def test_report_table_db_data_fields():
    t = ReportTableDbData(
        document_id="abc123",
        page_no=68,
        table_title="合并资产负债表",
        statement_type=StatementType.BALANCE,
        statement_scope=StatementScope.CONSOLIDATED,
        raw_markdown="| a | b |",
        extracted_metrics=[{"metric_name": "货币资金"}],
        extract_status=ExtractStatus.PENDING,
    )
    assert t.extract_status == ExtractStatus.PENDING
    assert t.raw_markdown == "| a | b |"


def test_financial_metric_db_data_fields():
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
    assert m.metric_code == "revenue"
    assert m.is_restated == 0
    assert m.metric_value == Decimal("6064948157.84")


def test_metric_definition_db_data_fields():
    d = MetricDefinitionDbData(
        metric_code="gross_margin",
        metric_name="毛利率",
        aliases=None,
        category=MetricCategory.PROFITABILITY,
        statement_type=StatementType.RATIO,
        metric_level=MetricLevel.L2,
        calc_formula="(revenue-cost_of_revenue)/revenue*100",
        unit_default="%",
        description="综合毛利率",
    )
    assert d.metric_level == MetricLevel.L2
    assert d.calc_formula is not None
