"""L3 计算（docs/04g §7.3）。

纯 Python 同比 / 环比 / CAGR / 多期对比 / 业务占比；L2 一律物化值（导入侧已算），此处不现算 L2。
不调 LLM、不打库；缺期/除零返回 status 而不是抛异常或编数（04g §4.3 不插值不外推）。
为什么 share 不物化：占比跨表跨期，导入是单文档流水，用户未必问（docs/02g §8.3）。
"""
from __future__ import annotations

# 除零/缺期归数据问题，不是计算结果（04g §7.3）
_INSUFFICIENT = "insufficient_periods"

_YOY_FORMULA = "(current-prior)/abs(prior)*100"
_CAGR_FORMULA = "((end/start)**(1/years)-1)*100"
_MULTI_FORMULA = "series[末]/series[首]-1"
_SHARE_FORMULA = "part/total*100"


def _result(kind: str, formula: str, **fields) -> dict:
    out = {
        "kind": kind,
        "value": None,
        "values": None,
        "unit": "%",
        "status": "ok",
        "reason": None,
        "formula": formula,
    }
    out.update(fields)
    return out


def _insufficient(kind: str, formula: str, reason: str) -> dict:
    return _result(kind, formula, status=_INSUFFICIENT, reason=reason)


def _growth_pct(kind: str, current: float | None, prior: float | None) -> dict:
    """同比/环比共用：(c-p)/abs(p)*100。

    分母取 abs(prior)：亏损收窄也是「增长」，直接除负数会反号（04g §7.3）。
    """
    if current is None or prior is None:
        return _insufficient(kind, _YOY_FORMULA, "基期值缺失")
    if prior == 0:
        return _insufficient(kind, _YOY_FORMULA, "分母为 0")
    value = (current - prior) / abs(prior) * 100
    return _result(kind, _YOY_FORMULA, value=value)


def yoy(current: float | None, prior: float | None) -> dict:
    """同比。"""
    return _growth_pct("yoy", current, prior)


def qoq(current: float | None, prior: float | None) -> dict:
    """环比。公式同 yoy，kind 不同供 generate 区分话术。"""
    return _growth_pct("qoq", current, prior)


def cagr(start: float | None, end: float | None, years: float | None) -> dict:
    """复合增长率。start≤0 或 years≤0 归 insufficient——无实数年化根。"""
    if start is None or end is None or years is None:
        return _insufficient("cagr", _CAGR_FORMULA, "基期值缺失")
    if start <= 0:
        return _insufficient("cagr", _CAGR_FORMULA, "起始值非正，无法年化")
    if years <= 0:
        return _insufficient("cagr", _CAGR_FORMULA, "年数非正")
    value = ((end / start) ** (1 / years) - 1) * 100
    return _result("cagr", _CAGR_FORMULA, value=value)


def multi_period(series: list[tuple[str, float | None]]) -> dict:
    """多期对比。value=None，values 给序列；<2 期归 insufficient。"""
    periods = [p for p, _ in series]
    vals = [v for _, v in series]
    if len(series) < 2 or any(v is None for v in vals):
        return _insufficient("multi_period", _MULTI_FORMULA, "有效序列不足 2 期")
    return _result(
        "multi_period",
        _MULTI_FORMULA,
        value=None,
        values=vals,
        periods=periods,
        unit="",
    )


def share(part: float | None, total: float | None) -> dict:
    """业务切片占比 = 分片值 / 合计值 × 100（docs/02g §8.3）。

    分母只认 dim_name='合计' 行；缺合计/分母 0 归 insufficient，
    不猜 0 也不猜 100——那是编数（docs/02g §8.3）。
    """
    if part is None or total is None:
        return _insufficient("share", _SHARE_FORMULA, "缺合计行")
    if total == 0:
        return _insufficient("share", _SHARE_FORMULA, "分母为 0")
    return _result("share", _SHARE_FORMULA, value=part / total * 100)
