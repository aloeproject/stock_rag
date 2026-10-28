"""指代消解测试（docs/04a §6）。

触发词 + history 回填；子公司复数指代不当股票实体（04a §6.3 样例）。
"""
from __future__ import annotations

from app.query_graph.matching.coref import (
    carry_forward_from_history,
    resolve_coreference,
)


def _user(
    content: str,
    codes: list[str] | None = None,
    names: list[str] | None = None,
    periods: list[str] | None = None,
) -> dict:
    return {
        "role": "user",
        "content": content,
        "stock_codes": codes or [],
        "stock_names": names or [],
        "periods": periods or [],
    }


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def test_singular_pronoun_takes_one_code():
    history = [
        _user("贵州茅台毛利率", codes=["600519"], names=["贵州茅台"]),
        _assistant("毛利率91%"),
    ]
    r = resolve_coreference("它的营收呢", history)
    assert r.trigger == "它"
    assert r.codes == ["600519"]
    assert "贵州茅台" in (r.rewritten_query or "")


def test_plural_pronoun_takes_all_codes():
    history = [
        _user("对比茅台和五粮液", codes=["600519", "000858"], names=["贵州茅台", "五粮液"]),
        _assistant("对比如下"),
    ]
    r = resolve_coreference("他们的毛利率呢", history)
    assert r.trigger == "他们"
    assert r.codes == ["600519", "000858"]


def test_subsidiary_pronoun_not_treated_as_stock():
    """历史在讲子公司时，「他们」指子公司，不能当股票代码沿用（04a §6.3）。"""
    history = [
        _user(
            "神州高铁旗下有新联铁、交大微联、南京拓控、武汉利德等子公司。",
            codes=["000008"],
            names=["神州高铁"],
        ),
        _assistant("是的，这些子公司在轨道交通检修装备领域各有专长。"),
    ]
    r = resolve_coreference("他们各自的核心产品是什么？", history)
    assert r.codes == []
    assert r.trigger == "他们"


def test_no_trigger_returns_empty():
    r = resolve_coreference("贵州茅台2024年毛利率是多少", [])
    assert r.trigger is None
    assert r.codes == []
    assert r.rewritten_query is None


def test_other_is_not_pronoun():
    """「其他/其它」是「别的」不是公司指代，不能误触发。"""
    r = resolve_coreference("还有其他重要的经营指标吗？", [])
    assert r.trigger is None


def test_carry_forward_when_no_new_names():
    history = [
        _user("神州高铁2024年经营情况", codes=["000008"], names=["神州高铁"], periods=["2024"]),
        _assistant("营收20.82亿"),
    ]
    codes, names, periods = carry_forward_from_history(history)
    assert codes == ["000008"]
    assert names == ["神州高铁"]
    assert periods == ["2024"]


def test_carry_forward_skips_empty():
    codes, names, periods = carry_forward_from_history([_user("你好", codes=[]), _assistant("hi")])
    assert codes == []
    assert names == []
    assert periods == []


def test_carry_forward_includes_periods():
    """periods 盲沿用：无代词省略句也要能拿回上一轮报告期（04b §4.4）。"""
    history = [
        _user("贵州茅台2024年盈利多少", codes=["600519"], names=["贵州茅台"], periods=["2024"]),
        _assistant("净利润862亿"),
    ]
    codes, names, periods = carry_forward_from_history(history)
    assert codes == ["600519"]
    assert periods == ["2024"]


def test_ordinal_pronoun_takes_previous_subject():
    history = [
        _user("茅台怎么样", codes=["600519"], names=["贵州茅台"]),
        _assistant("不错"),
        _user("五粮液呢", codes=["000858"], names=["五粮液"]),
        _assistant("也不错"),
    ]
    r = resolve_coreference("上一家的营收", history)
    assert r.trigger == "上一家"
    assert r.codes == ["600519"]
