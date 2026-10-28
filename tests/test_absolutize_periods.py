"""periods 绝对化测试（docs/04a §5）。

基准 = 最新可查报告期，不是自然年——1 月窗口里「近三年」指刚出的年报倒推。
"""
from __future__ import annotations

from app.query_graph.matching.periods import absolutize_periods, normalize_report_types


def test_absolute_year_kept():
    assert absolutize_periods(["2024"], "2024") == ["2024"]


def test_multiple_years_sorted_dedup():
    assert absolutize_periods(["2024", "2023", "2024"], "2024") == ["2023", "2024"]


def test_recent_three_years():
    assert absolutize_periods(["近三年"], "2024") == ["2022", "2023", "2024"]


def test_recent_three_years_digit():
    assert absolutize_periods(["近3年"], "2024") == ["2022", "2023", "2024"]


def test_recent_two_years():
    assert absolutize_periods(["近两年"], "2024") == ["2023", "2024"]


def test_last_year():
    assert absolutize_periods(["去年"], "2024") == ["2023"]


def test_year_before_last():
    assert absolutize_periods(["前年"], "2024") == ["2022"]


def test_this_year_uses_base_not_natural_year():
    """「今年」= 基准年（最新报告期），不是 2026 自然年。"""
    assert absolutize_periods(["今年"], "2024") == ["2024"]


def test_unrecognized_kept_as_is():
    assert absolutize_periods(["远古"], "2024") == ["远古"]


def test_empty_input():
    assert absolutize_periods([], "2024") == []


def test_mixed_relative_and_absolute():
    assert absolutize_periods(["去年", "2024"], "2024") == ["2023", "2024"]


def test_normalize_report_types():
    assert normalize_report_types(["年报", "半年报", "季度报", "神秘类型"]) == [
        "annual",
        "semi_annual",
        "quarterly",
        "神秘类型",
    ]
