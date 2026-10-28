"""表格轨道测试——docs/02a §4：小表整表、大表按行组、每片带表头。"""
from app.import_graph.split.config import SplitConfig
from app.import_graph.split.table_split import (
    extract_caption,
    split_table_rows,
    build_table_chunks,
)


HTML_SMALL = (
    "<table><tr><td>科目</td><td>本期数</td></tr>"
    "<tr><td>营业收入</td><td>1,120</td></tr>"
    "<tr><td>营业成本</td><td>653</td></tr></table>"
)

HTML_BIG = (
    "<table><tr><td>科目</td><td>金额</td></tr>"
    + "".join(f"<tr><td>行{i}</td><td>{i}</td></tr>" for i in range(40))
    + "<tr><td>小计</td><td>820</td></tr>"
    + "".join(f"<tr><td>续{i}</td><td>{i}</td></tr>" for i in range(20))
    + "</table>"
)


def test_extract_caption_takes_heading_and_unit():
    md = (
        "## 1、利润表及现金流量表相关科目变动分析表\n"
        "单位：元 币种：人民币\n"
        "<table></table>\n"
    )
    tables = [{"char_start": md.index("<table>"), "char_end": len(md), "html": "<table></table>"}]
    caption = extract_caption(md, tables[0])
    assert "利润表及现金流量表相关科目变动分析表" in caption
    assert "单位：元" in caption


def test_small_table_single_chunk():
    cfg = SplitConfig()
    chunks = build_table_chunks(
        html=HTML_SMALL,
        table_id="table_10_1",
        caption="利润表分析",
        section_path="第三节/五、经营情况",
        char_start=0,
        char_end=len(HTML_SMALL),
        cfg=cfg,
    )
    assert len(chunks) == 1
    assert chunks[0]["chunk_type"] == "table"
    assert chunks[0]["table_id"] == "table_10_1"
    assert "利润表分析" in chunks[0]["content"]
    assert "| 科目 | 本期数 |" in chunks[0]["content"]


def test_big_table_splits_by_row_group():
    cfg = SplitConfig(max_token=50, min_token=10, overlap_token=5)
    chunks = build_table_chunks(
        html=HTML_BIG,
        table_id="table_20_1",
        caption="大表",
        section_path="第十节",
        char_start=0,
        char_end=len(HTML_BIG),
        cfg=cfg,
    )
    assert len(chunks) > 1
    # 每片必须带完整表头
    for c in chunks:
        assert "| 科目 | 金额 |" in c["content"]
        assert c["part"] >= 1
    # 小计行是断点，不应被切碎到中间
    parts = [c["content"] for c in chunks]
    assert any("小计" in p for p in parts)


def test_split_table_rows_breaks_on_subtotal():
    rows = [["A", "1"], ["B", "2"], ["小计", "3"], ["C", "4"]]
    groups = split_table_rows(rows, header=["科目", "金额"], max_token=20, est=lambda r: 5)
    assert len(groups) == 2
    assert ["A", "1"] in groups[0]
    assert ["小计", "3"] in groups[0]
    assert ["C", "4"] in groups[1]
