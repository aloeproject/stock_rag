"""node_retrieve 纯规则测试（docs/04e §4/§5）。

filter 键映射、fan-out 展开、HyDE 三条件；不碰 Qdrant/LLM。
"""
from __future__ import annotations

from app.query_graph.matching.retrieve_rules import (
    build_payload_filter,
    fanout_units,
    has_metric_marker,
    should_use_hyde,
)

# ---------- fanout_units ----------


def test_fanout_single_code_single_period():
    assert fanout_units(["000008"], ["2024"]) == [("000008", "2024")]


def test_fanout_multi_period():
    assert fanout_units(["000008"], ["2022", "2023", "2024"]) == [
        ("000008", "2022"),
        ("000008", "2023"),
        ("000008", "2024"),
    ]


def test_fanout_empty_periods_yields_none_period():
    """simple_qa 可无期：不加 report_period 过滤，只按 stock_code。"""
    assert fanout_units(["000008"], []) == [("000008", None)]


def test_fanout_multi_code_multi_period_cartesian():
    """2 公司 × 2 期 = 4 单元，各有独立召回名额（04e §3.2）。"""
    units = fanout_units(["000008", "000007"], ["2023", "2024"])
    assert len(units) == 4
    assert ("000008", "2023") in units
    assert ("000007", "2024") in units


def test_fanout_empty_codes():
    assert fanout_units([], ["2024"]) == []


# ---------- build_payload_filter ----------


def test_build_filter_maps_plural_keys_to_payload():
    f = build_payload_filter(
        {
            "stock_codes": ["000008"],
            "report_periods": ["2024"],
            "report_types": ["annual"],
            "sections": ["risk"],
        }
    )
    assert f == {
        "stock_code": ["000008"],
        "report_period": ["2024"],
        "report_type": ["annual"],
        "section": ["risk"],
    }


def test_build_filter_omits_empty_keys():
    f = build_payload_filter({"stock_codes": ["000008"], "report_periods": ["2024"]})
    assert f == {"stock_code": ["000008"], "report_period": ["2024"]}
    assert "section" not in f
    assert "report_type" not in f


def test_build_filter_fanout_unit_single_values():
    """fan-out 后单元内是单值，映射仍走列表（MatchAny 口径统一）。"""
    f = build_payload_filter({"stock_codes": ["000008"], "report_periods": ["2024"]})
    assert f["stock_code"] == ["000008"]


# ---------- has_metric_marker ----------


def test_metric_marker_hits_colloquial():
    assert has_metric_marker("毛利率是多少") is True
    assert has_metric_marker("营收变化") is True


def test_metric_marker_misses_narrative():
    assert has_metric_marker("公司面临哪些风险") is False
    assert has_metric_marker("主要产品是什么") is False


# ---------- should_use_hyde 三条件 ----------
# 隔离测法：测某一条时用「长句/含指标词/高分」压掉另外两条


def test_hyde_cond1_vector_route_md_section():
    assert should_use_hyde("vector", ["management_discussion"], "毛利率下降的经营原因是什么情况呢", 0.9) is True


def test_hyde_cond1_vector_route_risk_section():
    assert should_use_hyde("vector", ["risk"], "毛利率下降面临的主要风险是什么情况呢", 0.9) is True


def test_hyde_cond1_requires_vector_route():
    """mixed 不吃条件 1——它还有 metric_sql 兜底，假设文档容易拉偏数字。"""
    assert should_use_hyde("mixed", ["management_discussion"], "毛利率下降的经营原因是什么情况呢", 0.9) is False


def test_hyde_cond2_short_question_no_metric():
    assert should_use_hyde("vector", [], "主要产品是什么", 0.9) is True


def test_hyde_cond2_short_question_with_metric_not_triggered():
    assert should_use_hyde("vector", [], "毛利率下降的原因", 0.9) is False


def test_hyde_cond2_long_question_not_triggered():
    q = "公司近两年在轨交检修装备市场的竞争格局变化以及对应的战略调整方向是什么"
    assert should_use_hyde("vector", [], q, 0.9) is False


def test_hyde_cond3_low_top1_score():
    assert should_use_hyde("vector", [], "毛利率下降的经营原因是什么情况呢", 0.20) is True


def test_hyde_cond3_high_top1_score_not_triggered():
    assert should_use_hyde("vector", [], "毛利率下降的经营原因是什么情况呢", 0.80) is False


def test_hyde_cond3_none_score_treated_as_weak():
    """首轮全空 → top1 视为 0，走 HyDE 补路。"""
    assert should_use_hyde("vector", [], "毛利率下降的经营原因是什么情况呢", None) is True


def test_hyde_not_triggered_when_no_condition():
    assert should_use_hyde("vector", ["notes"], "毛利率下降的经营原因是什么情况呢", 0.9) is False
