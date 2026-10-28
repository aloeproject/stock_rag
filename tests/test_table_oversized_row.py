"""不可约超大行——单行本身超 hard_max 时保留完整内容，不硬切破坏表格结构。"""
from app.import_graph.split.config import SplitConfig
from app.import_graph.split.table_split import build_table_chunks, split_table_rows
from app.import_graph.split.token_estimator import estimate_token_count


def test_single_row_larger_than_max_kept_whole():
    """单行超预算：自成一组，内容不丢（表格行被切断就无法还原语义）。"""
    header = ["承诺背景", "承诺类型", "承诺方", "承诺内容"]
    huge = ["收购报告书", "解决同业竞争", "某集团", "很长的承诺内容。" * 80]
    assert estimate_token_count("| " + " | ".join(huge) + " |") > 200
    groups = split_table_rows([huge, ["a", "b", "c", "d"]], header, max_token=100)
    assert groups[0] == [huge]
    assert groups[1] == [["a", "b", "c", "d"]]


def test_oversized_row_chunk_reports_flag():
    """build_table_chunks 对不可约超限块打 oversized 标记，供上层记 warning。"""
    header = ["a", "b"]
    huge_row = "<tr><td>甲</td><td>" + "很长的说明。" * 100 + "</td></tr>"
    html = f"<table><tr><td>{header[0]}</td><td>{header[1]}</td></tr>{huge_row}</table>"
    cfg = SplitConfig(max_token=50, min_token=10, overlap_token=5, hard_max_token=80)
    chunks = build_table_chunks(
        html=html,
        table_id="table_1_1",
        caption="承诺事项",
        section_path="第六节",
        char_start=0,
        char_end=10,
        cfg=cfg,
    )
    assert len(chunks) == 1
    assert chunks[0].get("oversized") is True
    # 内容完整保留
    assert "很长的说明" in chunks[0]["content"]
