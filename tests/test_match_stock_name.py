"""stock_dict 四级名称匹配测试（docs/04a §4）。

纯函数：传入 rows 不碰 DB，逐级短路可断言。
"""
from __future__ import annotations

import pytest

from app.entities.stock_dict import StockDictDbData
from app.query_graph.matching.stock_name import match_stock_names


def _row(
    code: str,
    name: str,
    aliases: list[str] | None = None,
    full_name: str = "",
    status: int = 1,
) -> StockDictDbData:
    return StockDictDbData(
        stock_code=code,
        stock_name=name,
        full_name=full_name,
        aliases=aliases or [],
        market="SH",
        status=status,
    )


@pytest.fixture()
def rows() -> list[StockDictDbData]:
    return [
        _row("600519", "贵州茅台", ["茅台", "贵州茅台酒"], "贵州茅台酒股份有限公司"),
        _row("000008", "神州高铁", [], "神州高铁技术股份有限公司"),
        _row("600628", "新世界", ["上海新世界"], "上海新世界股份有限公司"),
        _row("000006", "深振业A", [], "深圳市振业(集团)股份有限公司"),
    ]


def test_level1_exact_stock_name(rows):
    resolved, unresolved = match_stock_names(["贵州茅台"], rows)
    assert resolved == {"贵州茅台": "600519"}
    assert unresolved == []


def test_level2_alias_exact(rows):
    resolved, unresolved = match_stock_names(["茅台"], rows)
    assert resolved == {"茅台": "600519"}
    assert unresolved == []


def test_level3_stock_code(rows):
    resolved, unresolved = match_stock_names(["600519"], rows)
    assert resolved == {"600519": "600519"}
    assert unresolved == []


def test_level4_unique_prefix(rows):
    resolved, unresolved = match_stock_names(["神州高"], rows)
    assert resolved == {"神州高": "000008"}
    assert unresolved == []


def test_unknown_name_goes_unresolved(rows):
    resolved, unresolved = match_stock_names(["不存在的公司"], rows)
    assert resolved == {}
    assert unresolved == ["不存在的公司"]


def test_ambiguous_prefix_goes_unresolved():
    """前缀命中多行必须全部进 unresolved，禁止静默取第一条（04a §4.2）。

    这里刻意不用带「茅台」alias 的字典，否则会先在第 2 级精确命中。
    """
    rows = [
        _row("600001", "茅台股份", [], "茅台股份有限公司"),
        _row("600002", "茅台集团", [], "茅台集团有限公司"),
    ]
    resolved, unresolved = match_stock_names(["茅台"], rows)
    assert resolved == {}
    assert unresolved == ["茅台"]


def test_disabled_stock_goes_unresolved(rows):
    extra = rows + [_row("300001", "停用股", [], "停用公司", status=0)]
    resolved, unresolved = match_stock_names(["停用股"], extra)
    assert resolved == {}
    assert unresolved == ["停用股"]


def test_single_char_prefix_not_matched(rows):
    """前缀最短 2 字：「茅」不应命中「贵州茅台」或别名「茅台」。"""
    resolved, unresolved = match_stock_names(["茅"], rows)
    assert resolved == {}
    assert unresolved == ["茅"]


def test_code_not_in_dict_unresolved(rows):
    resolved, unresolved = match_stock_names(["999999"], rows)
    assert resolved == {}
    assert unresolved == ["999999"]


def test_mixed_batch(rows):
    resolved, unresolved = match_stock_names(["茅台", "600628", "火星公司"], rows)
    assert resolved == {"茅台": "600519", "600628": "600628"}
    assert unresolved == ["火星公司"]


def test_full_name_prefix_matches(rows):
    resolved, unresolved = match_stock_names(["振业"], rows)
    assert resolved == {"振业": "000006"}
    assert unresolved == []
