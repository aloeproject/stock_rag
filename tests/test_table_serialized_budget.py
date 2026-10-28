"""序列化预算回归——每组都要计表头/caption 开销，且按序列化实测而非 join 估算。"""
from app.import_graph.split.config import SplitConfig
from app.import_graph.split.table_split import build_table_chunks, split_table_rows
from app.import_graph.split.token_estimator import estimate_token_count


def test_second_group_still_budgets_header():
    """分组后表头重复出现，下一组预算必须重新计表头，否则第二组起必然超限。"""
    header = ["h"] * 10          # est=10
    rows = [["r"] * 10 for _ in range(20)]  # est=10/行
    groups = split_table_rows(rows, header, max_token=30, est=lambda r: 10)
    assert len(groups) > 1
    # 正确：header(10)+2 行(20)=30 → 每组 2 行；bug 会让第二组放 3 行
    for g in groups:
        assert len(g) <= 2, f"group size {len(g)} 未计入重复表头预算"


def test_wide_table_serialized_tokens_stay_near_max():
    """宽表 7 列长说明：按序列化后实测 token 分组，不许挤成超大块。"""
    header_html = (
        "<tr><td>项目名称</td><td>本期期末数</td><td>本期期末数占总资产的比例(%)</td>"
        "<td>上期期末数</td><td>上期期末数占总资产的比例(%)</td>"
        "<td>本期期末金额较上期期末变动比例(%)</td><td>情况说明</td></tr>"
    )
    row_html = (
        "<tr><td>科目{}</td><td>1,378,780,862.67</td><td>24.03</td>"
        "<td>1,143,116,859.51</td><td>20.10</td><td>20.62</td>"
        "<td>主要系本期期末持有的结构性存款减少所致，详见附注说明。</td></tr>"
    )
    rows = "".join(row_html.format(i) for i in range(20))
    html = f"<table>{header_html}{rows}</table>"
    cfg = SplitConfig(max_token=200, min_token=20, overlap_token=10, hard_max_token=550)
    chunks = build_table_chunks(
        html=html,
        table_id="table_1_1",
        caption="1、资产及负债状况\n单位：元",
        section_path="第三节 管理层讨论与分析/五、报告期内主要经营情况",
        char_start=0,
        char_end=100,
        cfg=cfg,
    )
    assert len(chunks) > 1
    for c in chunks:
        actual = estimate_token_count(c["content"])
        assert actual <= cfg.hard_max_token
        # 关键：不得因低估而膨胀到接近 2 倍 max
        assert actual <= cfg.max_token * 1.5, f"part={c['part']} token={actual} 远超 max={cfg.max_token}"
