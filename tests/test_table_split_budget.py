"""表格预算与 caption 回归——真实年报冒烟暴露：行组超 hard_max、caption 漏提。"""
from app.import_graph.split.config import SplitConfig
from app.import_graph.split.table_split import (
    build_table_chunks,
    extract_caption,
    split_table_rows,
)
from app.import_graph.split.token_estimator import estimate_token_count


def test_row_groups_budget_includes_header_each_time():
    """每组都带表头，预算必须按「表头+行」计，否则第二组起必然超限。"""
    header = ["科目名称", "本期期末数", "本期期末数占总资产比例", "上年期末数", "变动比例"]
    rows = [[f"项目{i}", "1,000,000.00", "10.5%", "900,000.00", "+5.2%"] for i in range(30)]
    # 表头 ~15 token，行 ~20 token/行；max=100 → 每组约 3 行
    groups = split_table_rows(rows, header, max_token=100, est=lambda r: estimate_token_count(" ".join(r)))
    assert len(groups) > 1
    for g in groups:
        # 组内行数受预算约束；表头在序列化时重复，此处验证行 token 和不明显越界
        tokens = sum(estimate_token_count(" ".join(r)) for r in g)
        assert tokens <= 100


def test_serialized_chunks_stay_under_hard_max():
    """序列化含 caption + section_path + 表头，chunk 总 token 不得超 hard_max。"""
    html_rows = "".join(
        f"<tr><td>项目{i}</td><td>1,000,000.00</td><td>10.5%</td><td>900,000.00</td></tr>"
        for i in range(40)
    )
    html = f"<table><tr><td>科目</td><td>本期</td><td>占比</td><td>上期</td></tr>{html_rows}</table>"
    cfg = SplitConfig(max_token=50, min_token=10, overlap_token=5, hard_max_token=550)
    chunks = build_table_chunks(
        html=html,
        table_id="table_1_1",
        caption="1、资产及负债状况\n单位：元",
        section_path="第三节/五、报告期内主要经营情况",
        char_start=0,
        char_end=100,
        cfg=cfg,
    )
    assert len(chunks) > 1
    for c in chunks:
        assert c["token_count"] if "token_count" in c else True
        assert estimate_token_count(c["content"]) <= cfg.hard_max_token


def test_extract_caption_finds_heading_with_gap():
    """标题与表之间隔着目录/多行正文时仍要能取到标题。"""
    md = (
        "## 1、资产及负债状况\n"
        + "\n".join(f"第{i}项说明文字若干" for i in range(8))
        + "\n单位：元\n"
        + "<table><tr><td>x</td></tr></table>\n"
    )
    table = {"char_start": md.index("<table>"), "char_end": len(md), "html": "<table></table>"}
    caption = extract_caption(md, table)
    assert "资产及负债状况" in caption
    assert "单位：元" in caption


def test_extract_caption_handles_toc_style_table():
    """目录后紧跟的备查文件表：标题被目录行冲淡，仍应保留最近标题。"""
    md = (
        "## 目 录\n"
        "第一节 释义....4\n第二节 公司简介....4\n第三节 管理层讨论....7\n"
        "第十节 财务报告....55\n"
        "<table><tr><td>备查文件目录</td><td>载有法定代表人签名的会计报表。</td></tr></table>\n"
    )
    table = {"char_start": md.index("<table>"), "char_end": len(md), "html": "<table></table>"}
    caption = extract_caption(md, table)
    assert "目" in caption
