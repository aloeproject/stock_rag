"""表格抽取测试——docs/02a §4：正文跳过 <table>，表格进独立轨道。"""
from app.import_graph.split.table_extract import (
    extract_tables,
    strip_tables,
    html_table_to_markdown,
)


SAMPLE = (
    "# 标题\n"
    "前言文字\n"
    "<table><tr><td>科目</td><td>本期数</td></tr>"
    "<tr><td>营业收入</td><td>1,120</td></tr></table>\n"
    "后文文字\n"
    "<table><tr><td>资产</td><td>期末</td></tr>"
    "<tr><td>货币资金</td><td>500</td></tr></table>\n"
)


def test_extract_tables_finds_all_tables():
    tables = extract_tables(SAMPLE)
    assert len(tables) == 2
    assert tables[0]["char_start"] < tables[1]["char_start"]


def test_extract_tables_records_offsets():
    tables = extract_tables(SAMPLE)
    t = tables[0]
    assert SAMPLE[t["char_start"]:t["char_end"]] == t["html"]
    assert t["html"].startswith("<table>")
    assert t["html"].endswith("</table>")


def test_strip_tables_leaves_narrative_with_placeholder():
    stripped = strip_tables(SAMPLE)
    assert "<table>" not in stripped
    assert "前言文字" in stripped
    assert "后文文字" in stripped
    # 占位保持长度一致，char_start/end 溯源不断裂
    assert len(stripped) == len(SAMPLE)


def test_html_table_to_markdown_basic():
    html = "<table><tr><td>科目</td><td>本期数</td></tr><tr><td>营业收入</td><td>1,120</td></tr></table>"
    md = html_table_to_markdown(html)
    lines = md.splitlines()
    assert lines[0] == "| 科目 | 本期数 |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| 营业收入 | 1,120 |"


def test_html_table_to_markdown_expands_colspan():
    html = (
        "<table><tr><td>名称</td><td colspan='2'>期末</td></tr>"
        "<tr><td>A</td><td>1</td><td>2</td></tr></table>"
    )
    md = html_table_to_markdown(html)
    # colspan 展开为多列，内容留首格、其余填空（不丢格子数）
    assert "| 名称 | 期末 |  |" in md
    assert "| A | 1 | 2 |" in md


def test_html_table_to_markdown_expands_rowspan():
    html = (
        "<table><tr><td rowspan='2'>账龄</td><td>期末</td></tr>"
        "<tr><td>期初</td></tr></table>"
    )
    md = html_table_to_markdown(html)
    lines = [l for l in md.splitlines() if l.startswith("|") and "---" not in l]
    assert lines[0] == "| 账龄 | 期末 |"
    assert lines[1] == "|  | 期初 |"
