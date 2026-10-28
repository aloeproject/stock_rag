"""node_metric_query 纯规则测试（docs/04f §3.2/§5.1）。

词→码三层映射（别名优先于包含）、口径裁决优先级；不碰 MySQL。
"""
from __future__ import annotations

from app.entities.metric_definition import (
    MetricCategory,
    MetricDefinitionDbData,
    MetricLevel,
)
from app.entities.report_table import StatementType
from app.query_graph.matching.metric_map import (
    map_metric_words,
    pick_metric_row,
)


def _def(
    code: str,
    name: str,
    aliases: str | None = None,
    level: MetricLevel = MetricLevel.L1,
) -> MetricDefinitionDbData:
    return MetricDefinitionDbData(
        metric_code=code,
        metric_name=name,
        aliases=aliases,
        category=MetricCategory.HEADLINE,
        statement_type=StatementType.HEADLINE,
        metric_level=level,
        calc_formula=None,
        unit_default=None,
        description=None,
    )


DEFS = [
    _def("revenue", "营业收入", "营业总收入|营收"),
    _def("net_profit_attributable", "归属于上市公司股东的净利润", "归母净利润|净利润"),
    _def("gross_margin", "销售毛利率", "毛利率", MetricLevel.L2),
    _def("revenue_yoy", "营业收入同比", None, MetricLevel.L3),
]


# ---------- map_metric_words 三层映射 ----------


def test_exact_name():
    codes, unmapped = map_metric_words(["营业收入"], DEFS)
    assert codes == ["revenue"]
    assert unmapped == []


def test_alias_match():
    codes, _ = map_metric_words(["归母净利润"], DEFS)
    assert codes == ["net_profit_attributable"]


def test_alias_beats_contains():
    """「归母净利润」是别名，不能被「净利润」的包含关系截胡（04f §3.2）。"""
    codes, _ = map_metric_words(["归母净利润"], DEFS)
    assert codes == ["net_profit_attributable"]
    assert "net_profit_attributable" in codes


def test_contains_match():
    codes, _ = map_metric_words(["毛利率"], DEFS)
    assert codes == ["gross_margin"]


def test_unmapped_word_dropped():
    codes, unmapped = map_metric_words(["重要指标"], DEFS)
    assert codes == []
    assert unmapped == ["重要指标"]


def test_l3_code_excluded():
    """L3 不在 financial_metrics，映射到了也查空（04f §3.2）。"""
    codes, unmapped = map_metric_words(["营业收入同比"], DEFS)
    assert codes == []
    assert unmapped == ["营业收入同比"]


def test_multi_words_dedup():
    codes, _ = map_metric_words(["营收", "营业收入", "毛利率"], DEFS)
    assert codes == ["revenue", "gross_margin"]


# 真实字典形状：net_profit 与归母/扣非并存，包含匹配不能互相截胡
DEFS_REAL = [
    _def("revenue", "营业收入", "营业总收入"),
    _def("net_profit_attributable", "归属于上市公司股东的净利润", "归属于母公司股东的净利润|归母净利润"),
    _def("net_profit_attributable_deducted", "归属于上市公司股东的扣除非经常性损益的净利润", "扣非归母净利润|扣非净利润"),
    _def("net_profit", "净利润", None),
    _def("net_margin", "净利率", None, MetricLevel.L2),
]


def test_deducted_alias_not_stolen_by_net_profit():
    """「扣非净利润」必须映射扣非码，不能被 name-in-word 的「净利润」截胡去 net_profit。"""
    codes, unmapped = map_metric_words(["扣非净利润"], DEFS_REAL)
    assert codes == ["net_profit_attributable_deducted"]
    assert unmapped == []


def test_contains_word_in_name_prefers_shortest():
    """词是 name 的碎片时取最短 name：「净利」→ 净利率/净利润，不进扣非全称（尾部含「净利」）。"""
    codes, _ = map_metric_words(["净利"], DEFS_REAL)
    assert codes in (["net_margin"], ["net_profit"])


def test_name_in_word_prefers_longest():
    """name 是词的碎片且无别名命中时取最长 name，避免先遍历到的短 name 抢走。"""
    # 同时含扣非全称与「净利润」碎片，取最长的扣非全称
    word = "归属于上市公司股东的扣除非经常性损益的净利润是多少"
    codes, _ = map_metric_words([word], DEFS_REAL)
    assert codes == ["net_profit_attributable_deducted"]


# ---------- pick_metric_row 口径裁决 ----------


def _row(scope="consolidated", period_type="current", row_id=1):
    return {
        "id": row_id,
        "stock_code": "000008",
        "metric_code": "revenue",
        "metric_period": "2024",
        "statement_scope": scope,
        "metric_period_type": period_type,
        "metric_value": 20.82,
    }


def test_scope_prefers_consolidated():
    picked = pick_metric_row([_row("parent", "current"), _row("consolidated", "current")])
    assert picked["statement_scope"] == "consolidated"


def test_scope_falls_back_to_parent():
    picked = pick_metric_row([_row("parent", "current")])
    assert picked["statement_scope"] == "parent"


def test_period_type_prefers_restated():
    picked = pick_metric_row([_row("consolidated", "current"), _row("consolidated", "restated")])
    assert picked["metric_period_type"] == "restated"


def test_period_type_current_over_comparative():
    picked = pick_metric_row([_row("consolidated", "comparative"), _row("consolidated", "current")])
    assert picked["metric_period_type"] == "current"


def test_empty_group_returns_none():
    assert pick_metric_row([]) is None
