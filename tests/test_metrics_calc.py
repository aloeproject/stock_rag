"""app/metrics/calc.py 纯函数测试（docs/04g §7.3）。

五函数 yoy/qoq/cagr/multi_period/share：数值、status 三态、除零/缺期边界。
不碰 DB / LLM / 状态——纯函数层。
"""
from __future__ import annotations

import math

import pytest

from app.metrics.calc import cagr, multi_period, qoq, share, yoy


# ---------- yoy / qoq ----------


def test_yoy_positive_growth():
    r = yoy(200.0, 100.0)
    assert r["status"] == "ok"
    assert r["kind"] == "yoy"
    assert r["unit"] == "%"
    assert r["value"] == pytest.approx(100.0)


def test_yoy_negative_base_uses_abs():
    """亏损收窄也是「增长」：分母取 abs(prior)（04g §7.3）。"""
    r = yoy(-50.0, -100.0)
    assert r["status"] == "ok"
    assert r["value"] == pytest.approx(50.0)


def test_yoy_worsening_loss():
    r = yoy(-150.0, -100.0)
    assert r["status"] == "ok"
    assert r["value"] == pytest.approx(-50.0)


def test_yoy_prior_zero():
    r = yoy(100.0, 0.0)
    assert r["status"] == "insufficient_periods"
    assert r["value"] is None
    assert r["reason"]


def test_qoq_same_formula_as_yoy():
    r = qoq(120.0, 100.0)
    assert r["kind"] == "qoq"
    assert r["status"] == "ok"
    assert r["value"] == pytest.approx(20.0)


# ---------- cagr ----------


def test_cagr_three_years():
    # 100 → 200 三年：(2^(1/3)-1)*100
    r = cagr(100.0, 200.0, 3)
    assert r["status"] == "ok"
    assert r["kind"] == "cagr"
    assert r["value"] == pytest.approx((2 ** (1 / 3) - 1) * 100, abs=1e-9)


def test_cagr_one_year_equals_yoy():
    r = cagr(100.0, 150.0, 1)
    assert r["status"] == "ok"
    assert r["value"] == pytest.approx(50.0)


def test_cagr_start_zero():
    r = cagr(0.0, 200.0, 3)
    assert r["status"] == "insufficient_periods"
    assert r["value"] is None


def test_cagr_start_negative():
    r = cagr(-100.0, 200.0, 3)
    assert r["status"] == "insufficient_periods"


def test_cagr_years_zero():
    r = cagr(100.0, 200.0, 0)
    assert r["status"] == "insufficient_periods"


# ---------- multi_period ----------


def test_multi_period_series():
    r = multi_period([("2022", 100.0), ("2023", 150.0), ("2024", 200.0)])
    assert r["status"] == "ok"
    assert r["kind"] == "multi_period"
    assert r["value"] is None
    assert r["values"] == [100.0, 150.0, 200.0]
    assert r["periods"] == ["2022", "2023", "2024"]


def test_multi_period_too_short():
    r = multi_period([("2024", 100.0)])
    assert r["status"] == "insufficient_periods"
    assert r["value"] is None
    assert r["values"] is None


def test_multi_period_empty():
    r = multi_period([])
    assert r["status"] == "insufficient_periods"
    assert r["value"] is None


# ---------- share ----------


def test_share_basic():
    r = share(30.0, 100.0)
    assert r["status"] == "ok"
    assert r["kind"] == "share"
    assert r["unit"] == "%"
    assert r["value"] == pytest.approx(30.0)
    assert r["formula"] == "part/total*100"


def test_share_missing_total_insufficient():
    """缺合计行不猜 0/100，标 insufficient（docs/02g §8.3）。"""
    r = share(30.0, None)
    assert r["status"] == "insufficient_periods"
    assert r["value"] is None
    assert r["reason"] == "缺合计行"


def test_share_missing_part_insufficient():
    r = share(None, 100.0)
    assert r["status"] == "insufficient_periods"
    assert r["reason"] == "缺合计行"


def test_share_total_zero_insufficient():
    r = share(30.0, 0.0)
    assert r["status"] == "insufficient_periods"
    assert r["value"] is None
    assert r["reason"] == "分母为 0"


# ---------- 共性 ----------


def test_all_results_have_status_and_formula():
    for r in (
        yoy(1.0, 2.0),
        qoq(1.0, 2.0),
        cagr(1.0, 2.0, 2),
        multi_period([("2023", 1.0), ("2024", 2.0)]),
        share(1.0, 2.0),
    ):
        assert r["status"] in ("ok", "insufficient_periods", "unsupported")
        assert isinstance(r["formula"], str) and r["formula"]
        assert "kind" in r


def test_yoy_inputs_untouched_by_none():
    """None 输入按缺期处理，不抛异常。"""
    r = yoy(None, 100.0)
    assert r["status"] == "insufficient_periods"
    assert r["value"] is None
