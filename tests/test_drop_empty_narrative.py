"""纯表题章节丢弃——标题+表格抠空后只剩空白，属表题不属叙述（真实文档 26 处）。"""
from app.import_graph.split.config import SplitConfig
from app.import_graph.split.dual_track import split_document


# 仿 013d 神州高铁「释义」章节：标题下直接跟表格，无叙述文字
MD_HEADING_TABLE_ONLY = (
    "# 第一节 重要提示、目录和释义\n"
    "前言一句。\n"
    "## 释义\n"
    "\n"
    "<table><tr><td>释义项</td><td>指</td><td>释义内容</td></tr>"
    "<tr><td>公司</td><td>指</td><td>神州高铁技术股份有限公司</td></tr></table>\n"
    "\n"
    "## 二、公司简介\n"
    "公司简介的叙述正文。\n"
)

PAGES = [{"char_start": 0, "char_end": 10000, "page_no": 1}]


def test_heading_only_table_section_not_in_narrative():
    """「释义」这类纯表题章节：叙述侧丢弃，不产出只有标题+空白的 chunk。"""
    chunks = split_document(MD_HEADING_TABLE_ONLY, PAGES, "d1", "x", SplitConfig())
    narrative = [c for c in chunks if c["chunk_type"] == "narrative"]
    # 不能有「标题后只有空白」的叙述块（含纯表题 ## 释义）
    for c in narrative:
        body = c["content"].split("\n", 1)[-1] if "\n" in c["content"] else ""
        assert body.strip(), f"空叙述块: {c['content']!r}"
    titles = [c["title"].strip() for c in narrative]
    assert "## 释义" not in titles


def test_real_narrative_text_survives():
    """正常叙述不能被误删。"""
    chunks = split_document(MD_HEADING_TABLE_ONLY, PAGES, "d1", "x", SplitConfig())
    narrative = [c for c in chunks if c["chunk_type"] == "narrative"]
    assert any("公司简介的叙述正文" in c["content"] for c in narrative)
    assert any("前言一句" in c["content"] for c in narrative)


def test_table_still_produced_with_caption():
    """表题丢弃后，表格侧要能继承 caption，否则标题信息彻底丢失。"""
    chunks = split_document(MD_HEADING_TABLE_ONLY, PAGES, "d1", "x", SplitConfig())
    tables = [c for c in chunks if c["chunk_type"] == "table"]
    assert len(tables) == 1
    assert "释义" in tables[0]["table_caption"] or "释义" in tables[0]["content"]
