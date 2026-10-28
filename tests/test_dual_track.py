"""双轨管线测试——docs/02a §1：叙述/表格分开切，chunk_id 稳定可重跑。"""
from app.import_graph.split.config import SplitConfig
from app.import_graph.split.dual_track import split_document


SAMPLE_MD = (
    "# 第十节 财务报告\n"
    "这是财务报告的前言说明。\n"
    "## 五、合并财务报表\n"
    "合并报表范围说明。\n"
    "## 1、利润表及现金流量表相关科目变动分析表\n"
    "单位：元 币种：人民币\n"
    "<table><tr><td>科目</td><td>本期数</td></tr>"
    "<tr><td>营业收入</td><td>1,120,106,782.08</td></tr></table>\n"
    "2、成本：本年度公司营业成本下降。\n"
)

PAGES = [
    {"char_start": 0, "char_end": 80, "page_no": 55},
    {"char_start": 81, "char_end": 400, "page_no": 56},
]


def test_dual_track_produces_both_types():
    chunks = split_document(
        md_content=SAMPLE_MD,
        pages=PAGES,
        document_id="doc123",
        file_title="新世界2024",
        cfg=SplitConfig(),
    )
    types = {c["chunk_type"] for c in chunks}
    assert types == {"narrative", "table"}


def test_narrative_chunks_skip_table_html():
    chunks = split_document(SAMPLE_MD, PAGES, "doc123", "x", SplitConfig())
    for c in chunks:
        if c["chunk_type"] == "narrative":
            assert "<table>" not in c["content"]


def test_table_chunk_has_id_and_caption():
    chunks = split_document(SAMPLE_MD, PAGES, "doc123", "x", SplitConfig())
    tables = [c for c in chunks if c["chunk_type"] == "table"]
    assert len(tables) == 1
    assert tables[0]["table_id"].startswith("table_")
    assert "利润表" in tables[0]["table_caption"]
    assert "| 科目 | 本期数 |" in tables[0]["content"]


def test_chunk_id_stable_across_runs():
    a = split_document(SAMPLE_MD, PAGES, "doc123", "x", SplitConfig())
    b = split_document(SAMPLE_MD, PAGES, "doc123", "x", SplitConfig())
    assert [c["chunk_id"] for c in a] == [c["chunk_id"] for c in b]
    assert all(c["chunk_id"].startswith("doc123:") for c in a)


def test_chunks_carry_page_and_tokens():
    chunks = split_document(SAMPLE_MD, PAGES, "doc123", "x", SplitConfig())
    for c in chunks:
        assert c["page_no"] >= 55
        assert c["token_count"] > 0
        assert c["section_path"]


def test_chunk_seq_ordering():
    chunks = split_document(SAMPLE_MD, PAGES, "doc123", "x", SplitConfig())
    seqs = [int(c["chunk_id"].split(":")[1]) for c in chunks]
    assert seqs == sorted(seqs)
