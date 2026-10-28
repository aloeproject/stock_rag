"""node_metric_normalize 测试：映射、period/单位、uk 裁决、L2 物化（docs/02e）。"""
import asyncio
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.entities.metric_definition import (
    MetricCategory,
    MetricDefinitionDbData,
    MetricLevel,
)
from app.entities.report_table import (
    ExtractStatus,
    ReportTableDbData,
    StatementScope,
    StatementType,
)
from app.import_graph.nodes.metric_normalize import (
    _normalize_unit_value,
    match_code,
    normalize_metric_period,
    node_metric_normalize,
)
from app.models.base import Base
from app.repositories.financial_metric_repository import FinancialMetricRepository
from app.repositories.metric_definition_repository import MetricDefinitionRepository
from app.repositories.report_table_repository import ReportTableRepository


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    yield s
    s.close()


@pytest.fixture()
def ctx_runtime(runtime, session):
    from app.repositories.business_fact_repository import BusinessFactRepository

    runtime.context.report_table_repository = ReportTableRepository(session)
    runtime.context.metric_definition_repository = MetricDefinitionRepository(session)
    runtime.context.financial_metric_repository = FinancialMetricRepository(session)
    runtime.context.business_fact_repository = BusinessFactRepository(session)
    return runtime


def _def(code, name, aliases=None, level=MetricLevel.L1, formula=None, unit="元", stmt=StatementType.INCOME):
    return MetricDefinitionDbData(
        metric_code=code,
        metric_name=name,
        aliases=aliases,
        category=MetricCategory.INCOME,
        statement_type=stmt,
        metric_level=level,
        calc_formula=formula,
        unit_default=unit,
        description=None,
    )


def _seed_defs(repo: MetricDefinitionRepository):
    for d in [
        _def("revenue", "营业收入", "营业总收入"),
        _def("cost_of_revenue", "营业成本", None, stmt=StatementType.INCOME),
        _def("net_profit_attributable", "归属于上市公司股东的净利润", "归属于母公司股东的净利润|归母净利润"),
        _def("net_operating_cashflow", "经营活动产生的现金流量净额", None, stmt=StatementType.CASHFLOW),
        _def("eps_basic", "基本每股收益", None, unit="元每股", stmt=StatementType.HEADLINE),
        _def("total_assets", "总资产", "资产总计|资产合计", stmt=StatementType.BALANCE),
        _def("net_assets_attributable", "归属于上市公司股东的净资产", None, stmt=StatementType.BALANCE),
        _def("current_assets_total", "流动资产合计", None, stmt=StatementType.BALANCE),
        _def("current_liabilities_total", "流动负债合计", None, stmt=StatementType.BALANCE),
        _def("total_liabilities", "负债合计", None, stmt=StatementType.BALANCE),
        _def("share_capital", "股本", "实收资本(或股本)", stmt=StatementType.BALANCE),
        _def(
            "gross_margin",
            "毛利率",
            level=MetricLevel.L2,
            formula="(revenue-cost_of_revenue)/revenue*100",
            unit="%",
            stmt=StatementType.RATIO,
        ),
        _def(
            "debt_to_asset_ratio",
            "资产负债率",
            level=MetricLevel.L2,
            formula="total_liabilities/total_assets*100",
            unit="%",
            stmt=StatementType.RATIO,
        ),
        _def(
            "current_ratio",
            "流动比率",
            level=MetricLevel.L2,
            formula="current_assets_total/current_liabilities_total",
            unit="倍",
            stmt=StatementType.RATIO,
        ),
    ]:
        repo.upsert(d)


def _table(repo, statement_type, scope, items, title="t", page=1):
    return repo.insert(
        ReportTableDbData(
            document_id="doc1",
            page_no=page,
            table_title=title,
            statement_type=statement_type,
            statement_scope=scope,
            raw_markdown="raw",
            extracted_metrics=items,
            extract_status=ExtractStatus.OK,
        )
    )


def _item(name, period, value, unit="元", ptype="current", restated=False):
    return {
        "metric_name": name,
        "metric_period": period,
        "metric_period_type": ptype,
        "value_raw": value,
        "unit": unit,
        "is_restated": restated,
    }


def _state():
    return {
        "document_id": "doc1",
        "stock_code": "000006",
        "report_period": "2024",
        "report_type": "annual",
        "metrics_rows": 0,
        "business_facts_rows": 0,
        "unmapped_names": [],
    }


def _seg_item(dim_type, dim_name, name, period, value, unit="元"):
    return {
        "dim_type": dim_type,
        "dim_name": dim_name,
        "metric_name": name,
        "metric_period": period,
        "metric_period_type": "current",
        "value_raw": value,
        "unit": unit,
        "is_restated": False,
    }


# ---------- 纯函数 ----------

def test_match_code_exact_first():
    defs = [_def("current_assets_total", "流动资产合计"), _def("revenue", "营业收入", "营业总收入")]
    assert match_code("营业收入", defs) == ("revenue", "exact")
    assert match_code("营业总收入", defs) == ("revenue", "exact")
    assert match_code("流动资产合计", defs) == ("current_assets_total", "exact")


def test_no_dangerous_contains_match():
    """「非流动资产合计」不得串进「流动资产合计」，「购建固定资产…」不得串进「固定资产」（02e §2.2）。"""
    defs = [
        _def("current_assets_total", "流动资产合计"),
        _def("fixed_assets", "固定资产"),
        _def("eps_basic", "基本每股收益"),
    ]
    assert match_code("非流动资产合计", defs) == (None, "unmapped")
    assert match_code("购建固定资产、无形资产和其他长期资产支付的现金", defs) == (None, "unmapped")
    assert match_code("扣除非经常性损益后的基本每股收益", defs) == (None, "unmapped")


def test_period_current_prior_resolved():
    assert normalize_metric_period("current", "2024") == "2024"
    assert normalize_metric_period("prior", "2024") == "2023"
    assert normalize_metric_period("2024", "2024") == "2024"
    assert normalize_metric_period("2024Q3", "2024") == "2024Q3"
    assert normalize_metric_period("n/a", "2024") is None


# ---------- 节点 ----------

def test_unmapped_name_reported_not_stored(ctx_runtime):
    _seed_defs(ctx_runtime.context.metric_definition_repository)
    _table(
        ctx_runtime.context.report_table_repository,
        StatementType.BALANCE,
        StatementScope.CONSOLIDATED,
        [_item("少数股东权益", "2024", "1.0"), _item("营业收入", "2024", "100")],
    )
    patch = asyncio.run(node_metric_normalize(_state(), ctx_runtime))
    assert patch["unmapped_names"] == ["少数股东权益"]
    assert patch["metrics_rows"] == 1
    rows = ctx_runtime.context.financial_metric_repository.list_by_document("doc1")
    assert [r.metric_code for r in rows] == ["revenue"]


def test_equity_table_excluded_from_l1(ctx_runtime):
    _seed_defs(ctx_runtime.context.metric_definition_repository)
    _table(
        ctx_runtime.context.report_table_repository,
        StatementType.EQUITY,
        StatementScope.CONSOLIDATED,
        [_item("股本", "2024", "646875384.00"), _item("未分配利润", "2024", "1")],
        title="合并所有者权益变动表",
    )
    patch = asyncio.run(node_metric_normalize(_state(), ctx_runtime))
    assert patch["metrics_rows"] == 0
    assert ctx_runtime.context.financial_metric_repository.list_by_document("doc1") == []


def test_period_current_prior_mapped(ctx_runtime):
    _seed_defs(ctx_runtime.context.metric_definition_repository)
    _table(
        ctx_runtime.context.report_table_repository,
        StatementType.BALANCE,
        StatementScope.CONSOLIDATED,
        [_item("总资产", "current", "100"), _item("总资产", "prior", "90")],
    )
    patch = asyncio.run(node_metric_normalize(_state(), ctx_runtime))
    assert patch["metrics_rows"] == 2
    rows = ctx_runtime.context.financial_metric_repository.list_by_document("doc1")
    periods = {r.metric_period for r in rows}
    assert periods == {"2024", "2023"}


def test_unit_wan_converted_to_yuan(ctx_runtime):
    _seed_defs(ctx_runtime.context.metric_definition_repository)
    _table(
        ctx_runtime.context.report_table_repository,
        StatementType.INCOME,
        StatementScope.CONSOLIDATED,
        [_item("营业收入", "2024", "112.01", unit="万元")],
    )
    asyncio.run(node_metric_normalize(_state(), ctx_runtime))
    rows = ctx_runtime.context.financial_metric_repository.list_by_document("doc1")
    assert rows[0].unit == "元"
    assert rows[0].metric_value == Decimal("1120100.00") or float(rows[0].metric_value) == 1120100.0


def test_uk_conflict_prefers_income_over_headline(ctx_runtime):
    _seed_defs(ctx_runtime.context.metric_definition_repository)
    _table(
        ctx_runtime.context.report_table_repository,
        StatementType.HEADLINE,
        StatementScope.CONSOLIDATED,
        [_item("营业收入", "2024", "1120106782.08")],
        title="主要会计数据",
    )
    _table(
        ctx_runtime.context.report_table_repository,
        StatementType.INCOME,
        StatementScope.CONSOLIDATED,
        [_item("营业总收入", "2024", "1120106782.08")],
        title="合并利润表",
    )
    patch = asyncio.run(node_metric_normalize(_state(), ctx_runtime))
    assert patch["metrics_rows"] == 1
    rows = ctx_runtime.context.financial_metric_repository.list_by_document("doc1")
    assert rows[0].extract_source.value == "table"
    assert float(rows[0].metric_value) == 1120106782.08


def test_uk_prefers_core_statement_over_notes_snippet(ctx_runtime):
    """附注里的联营企业表也有「资产合计/负债合计/净利润」，不得压过主表（真实数据 id=179）。"""
    _seed_defs(ctx_runtime.context.metric_definition_repository)
    _table(
        ctx_runtime.context.report_table_repository,
        StatementType.BALANCE,
        StatementScope.CONSOLIDATED,
        [
            _item("资产总计", "2024", "5738064887.48"),
            _item("负债合计", "2024", "1526214457.61"),
        ],
        title="合并资产负债表",
        page=58,
    )
    _table(
        ctx_runtime.context.report_table_repository,
        StatementType.BALANCE,
        StatementScope.CONSOLIDATED,
        [
            _item("资产合计", "current", "59010202.29"),
            _item("负债合计", "current", "1040.00"),
        ],
        title="单位：元 币种：人民币",
        page=136,
    )
    asyncio.run(node_metric_normalize(_state(), ctx_runtime))
    rows = {
        r.metric_code: r
        for r in ctx_runtime.context.financial_metric_repository.list_by_document("doc1")
    }
    assert abs(float(rows["total_assets"].metric_value) - 5738064887.48) < 0.01
    assert abs(float(rows["total_liabilities"].metric_value) - 1526214457.61) < 0.01


def test_l2_gross_margin_derived(ctx_runtime):
    _seed_defs(ctx_runtime.context.metric_definition_repository)
    _table(
        ctx_runtime.context.report_table_repository,
        StatementType.INCOME,
        StatementScope.CONSOLIDATED,
        [_item("营业收入", "2024", "1120106782.08"), _item("营业成本", "2024", "652665861.02")],
    )
    asyncio.run(node_metric_normalize(_state(), ctx_runtime))
    rows = {
        r.metric_code: r
        for r in ctx_runtime.context.financial_metric_repository.list_by_document("doc1")
    }
    assert "gross_margin" in rows
    gm = rows["gross_margin"]
    assert gm.extract_source.value == "derived"
    assert abs(float(gm.metric_value) - 41.73) < 0.1
    assert gm.unit == "%"


def test_l2_missing_dependency_skipped(ctx_runtime):
    _seed_defs(ctx_runtime.context.metric_definition_repository)
    _table(
        ctx_runtime.context.report_table_repository,
        StatementType.INCOME,
        StatementScope.CONSOLIDATED,
        [_item("营业收入", "2024", "100")],  # 缺 cost_of_revenue
    )
    asyncio.run(node_metric_normalize(_state(), ctx_runtime))
    codes = [r.metric_code for r in ctx_runtime.context.financial_metric_repository.list_by_document("doc1")]
    assert "revenue" in codes
    assert "gross_margin" not in codes


def test_rerun_idempotent(ctx_runtime):
    _seed_defs(ctx_runtime.context.metric_definition_repository)
    _table(
        ctx_runtime.context.report_table_repository,
        StatementType.INCOME,
        StatementScope.CONSOLIDATED,
        [_item("营业收入", "2024", "100"), _item("营业成本", "2024", "60")],
    )
    asyncio.run(node_metric_normalize(_state(), ctx_runtime))
    first = ctx_runtime.context.financial_metric_repository.list_by_document("doc1")
    asyncio.run(node_metric_normalize(_state(), ctx_runtime))
    second = ctx_runtime.context.financial_metric_repository.list_by_document("doc1")
    assert len(first) == len(second) == 3  # revenue + cost + gross_margin


# ---------- 百分号写进 value_raw（真实日志：加权ROE='1.86%'）----------


def test_percent_sign_inside_value_raw_parsed():
    # LLM 把 % 写进 value_raw，Decimal 直接解析会失败丢整条
    assert _normalize_unit_value("1.86%", "%", "%") == (Decimal("1.86"), "%")
    assert _normalize_unit_value("-140.99%", None, "%") == (Decimal("-140.99"), "%")
    assert _normalize_unit_value("175.55％", "％", None) == (Decimal("175.55"), "%")


def test_percent_sign_forces_percent_unit():
    # 值自带 % 时以值为准，unit 写错/缺失也不能落成「元」
    assert _normalize_unit_value("1.86%", None, "元") == (Decimal("1.86"), "%")
    assert _normalize_unit_value("1.86%", "万元", None) == (Decimal("1.86"), "%")


def test_percent_metric_stored_via_node(ctx_runtime):
    _seed_defs(ctx_runtime.context.metric_definition_repository)
    ctx_runtime.context.metric_definition_repository.upsert(
        _def("roe_weighted", "加权平均净资产收益率", unit="%", stmt=StatementType.RATIO)
    )
    _table(
        ctx_runtime.context.report_table_repository,
        StatementType.HEADLINE,
        StatementScope.CONSOLIDATED,
        [_item("加权平均净资产收益率", "2024", "1.86%", unit="%")],
        title="主要财务指标",
    )
    asyncio.run(node_metric_normalize(_state(), ctx_runtime))
    rows = ctx_runtime.context.financial_metric_repository.list_by_document("doc1")
    assert len(rows) == 1
    assert float(rows[0].metric_value) == 1.86
    assert rows[0].unit == "%"


# ---------- uk 裁决：0 坏值不得压过真值 ----------


def test_uk_zero_bad_value_never_beats_real(ctx_runtime):
    """专用表 0.00 排序天然压过 headline 真值，0 是抽取坏值必须让位。"""
    _seed_defs(ctx_runtime.context.metric_definition_repository)
    _table(
        ctx_runtime.context.report_table_repository,
        StatementType.HEADLINE,
        StatementScope.CONSOLIDATED,
        [_item("营业收入", "2024", "1120106782.08")],
        title="主要会计数据",
    )
    _table(
        ctx_runtime.context.report_table_repository,
        StatementType.INCOME,
        StatementScope.CONSOLIDATED,
        [_item("营业收入", "2024", "0.00")],
        title="合并利润表",
    )
    asyncio.run(node_metric_normalize(_state(), ctx_runtime))
    rows = ctx_runtime.context.financial_metric_repository.list_by_document("doc1")
    assert len(rows) == 1
    assert float(rows[0].metric_value) == 1120106782.08


def test_uk_all_zero_still_stores_one(ctx_runtime):
    _seed_defs(ctx_runtime.context.metric_definition_repository)
    _table(
        ctx_runtime.context.report_table_repository,
        StatementType.INCOME,
        StatementScope.CONSOLIDATED,
        [_item("营业收入", "2024", "0")],
        title="合并利润表",
    )
    _table(
        ctx_runtime.context.report_table_repository,
        StatementType.INCOME,
        StatementScope.PARENT,
        [_item("营业收入", "2024", "0.00")],
        title="母公司利润表",
    )
    asyncio.run(node_metric_normalize(_state(), ctx_runtime))
    rows = ctx_runtime.context.financial_metric_repository.list_by_document("doc1")
    # scope 不同是两行，各自保留 0
    assert len(rows) == 2
    assert all(float(r.metric_value) == 0 for r in rows)


# ---------- 业务切片 → business_facts（docs/02g） ----------


def test_segment_items_routed_to_business_facts(ctx_runtime):
    """segment 不进 financial_metrics：开放维度不该掺进公司级 uk（docs/02g §1.1）。"""
    from app.entities.business_fact import DimType

    _seed_defs(ctx_runtime.context.metric_definition_repository)
    _table(
        ctx_runtime.context.report_table_repository,
        StatementType.SEGMENT,
        StatementScope.CONSOLIDATED,
        [
            _seg_item("industry", "煤炭采掘业", "营业收入", "2024", "100"),
            _seg_item("industry", "合计", "营业收入", "2024", "1000"),
            _seg_item("product", "电力", "营业收入", "2024", "50"),
            # 同名不同段是两个事实（docs/02g §5.2）
            _seg_item("industry", "电力", "营业收入", "2024", "30"),
        ],
        title="(1). 主营业务分行业、分产品、分地区、分销售模式情况",
    )
    patch = asyncio.run(node_metric_normalize(_state(), ctx_runtime))
    assert patch["metrics_rows"] == 0
    assert patch["business_facts_rows"] == 4
    assert ctx_runtime.context.financial_metric_repository.list_by_document("doc1") == []

    facts = ctx_runtime.context.business_fact_repository.list_by_document("doc1")
    assert len(facts) == 4
    by_key = {(f.dim_type, f.dim_name): f for f in facts}
    assert by_key[(DimType.INDUSTRY, "煤炭采掘业")].metric_code == "revenue"
    assert by_key[(DimType.INDUSTRY, "合计")].metric_value == Decimal("1000") or float(
        by_key[(DimType.INDUSTRY, "合计")].metric_value
    ) == 1000.0
    # 分行业「电力」与分产品「电力」两行
    assert (DimType.INDUSTRY, "电力") in by_key
    assert (DimType.PRODUCT, "电力") in by_key


def test_segment_metric_code_optional_mapping(ctx_runtime):
    """「毛利率(%)」去单位后缀映射；「比上年增减(%)」不映射 yoy 码（docs/02g §6.2）。"""
    _seed_defs(ctx_runtime.context.metric_definition_repository)
    _table(
        ctx_runtime.context.report_table_repository,
        StatementType.SEGMENT,
        StatementScope.CONSOLIDATED,
        [
            _seg_item("industry", "煤炭采掘业", "毛利率(%)", "2024", "34.2", unit="%"),
            _seg_item("industry", "煤炭采掘业", "营业收入比上年增减(%)", "2024", "-1.2", unit="%"),
            _seg_item("industry", "煤炭采掘业", "研发投入", "2024", "88"),
        ],
        title="主营业务分行业情况",
    )
    asyncio.run(node_metric_normalize(_state(), ctx_runtime))
    facts = {
        f.metric_name: f
        for f in ctx_runtime.context.business_fact_repository.list_by_document("doc1")
    }
    assert facts["毛利率(%)"].metric_code == "gross_margin"
    # 披露的 yoy ≠ 查询侧算的 L3，撞 uk 会分不清（docs/02g §6.2）
    assert facts["营业收入比上年增减(%)"].metric_code is None
    # 新列头 0 字典改动可存（docs/02g §7.2）
    assert facts["研发投入"].metric_code is None
    assert float(facts["研发投入"].metric_value) == 88.0


def test_segment_item_missing_dim_dropped(ctx_runtime):
    """缺 dim_type/dim_name 丢弃 + 不猜维度（docs/02g §6.1/§9）。"""
    _seed_defs(ctx_runtime.context.metric_definition_repository)
    _table(
        ctx_runtime.context.report_table_repository,
        StatementType.SEGMENT,
        StatementScope.CONSOLIDATED,
        [
            {"metric_name": "营业收入", "metric_period": "2024", "value_raw": "100", "unit": "元"},
            _seg_item("industry", "煤炭采掘业", "营业收入", "2024", "100"),
        ],
        title="主营业务分行业情况",
    )
    patch = asyncio.run(node_metric_normalize(_state(), ctx_runtime))
    assert patch["business_facts_rows"] == 1
    facts = ctx_runtime.context.business_fact_repository.list_by_document("doc1")
    assert len(facts) == 1 and facts[0].dim_name == "煤炭采掘业"


def test_segment_unknown_dim_type_normalized_to_other(ctx_runtime):
    """dim_type 不在枚举归 other 不丢行（docs/02g §7.3/§9）。"""
    from app.entities.business_fact import DimType

    _seed_defs(ctx_runtime.context.metric_definition_repository)
    _table(
        ctx_runtime.context.report_table_repository,
        StatementType.SEGMENT,
        StatementScope.CONSOLIDATED,
        [_seg_item("地区", "西北", "营业收入", "2024", "100")],
        title="主营业务分地区情况",
    )
    patch = asyncio.run(node_metric_normalize(_state(), ctx_runtime))
    assert patch["business_facts_rows"] == 1
    facts = ctx_runtime.context.business_fact_repository.list_by_document("doc1")
    assert facts[0].dim_type == DimType.OTHER
    assert facts[0].dim_name == "西北"


def test_segment_rerun_idempotent(ctx_runtime):
    _seed_defs(ctx_runtime.context.metric_definition_repository)
    _table(
        ctx_runtime.context.report_table_repository,
        StatementType.SEGMENT,
        StatementScope.CONSOLIDATED,
        [
            _seg_item("industry", "煤炭采掘业", "营业收入", "2024", "100"),
            _seg_item("industry", "合计", "营业收入", "2024", "1000"),
        ],
        title="主营业务分行业情况",
    )
    asyncio.run(node_metric_normalize(_state(), ctx_runtime))
    first = ctx_runtime.context.business_fact_repository.list_by_document("doc1")
    asyncio.run(node_metric_normalize(_state(), ctx_runtime))
    second = ctx_runtime.context.business_fact_repository.list_by_document("doc1")
    assert len(first) == len(second) == 2
