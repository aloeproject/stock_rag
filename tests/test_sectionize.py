"""S4 章节归类测试——docs/02b §4：关键词规则、优先级、边界样例。"""
from langgraph.runtime import Runtime

from app.import_graph.context import DataAgentContext
from app.import_graph.nodes.sectionize import classify_section, node_sectionize


def _chunk(section_path: str, chunk_type: str = "narrative", **kw) -> dict:
    return {
        "chunk_type": chunk_type,
        "section_path": section_path,
        "title": section_path.rsplit("/", 1)[-1],
        "content": "正文",
        **kw,
    }


def test_table_chunk_is_table_desc():
    """chunk_type=table 不走关键词，直接 table_desc（docs/02b §3.2）。"""
    c = _chunk("第十节 财务报告/七、合并财务报表项目注释", chunk_type="table")
    assert classify_section(c) == "table_desc"


def test_management_discussion_matched():
    c = _chunk("第三节 管理层讨论与分析/四、主营业务分析/2、收入与成本")
    assert classify_section(c) == "management_discussion"


def test_risk_under_management_discussion_wins():
    """risk 优先于 management_discussion——业务风险比所属章节更具体。"""
    c = _chunk(
        "第三节 管理层讨论与分析/十一、公司未来发展的展望/"
        "（二）公司可能面临的风险及应对措施"
    )
    assert classify_section(c) == "risk"


def test_standalone_risk_section():
    c = _chunk("十、重大风险提示")
    assert classify_section(c) == "risk"
    assert classify_section(_chunk("六、前瞻性陈述的风险声明")) == "risk"


def test_risk_children_without_parent_phrase():
    """「（二）可能面临的风险」下子节 path 无父级短语，靠自身「风险」二字识别。"""
    c = _chunk("第三节 管理层讨论与分析/十一、公司未来发展的展望/1、宏观环境风险")
    assert classify_section(c) == "risk"


def test_financial_instrument_risk_is_notes_not_risk():
    """与金融工具相关的风险挂在财务报告下，是附注不是业务风险。"""
    c = _chunk("第十节 财务报告/十二、与金融工具相关的风险")
    assert classify_section(c) == "notes"


def test_notes_l1_financial_report():
    c = _chunk("第十节 财务报告/七、合并财务报表项目注释")
    assert classify_section(c) == "notes"
    # 财务报告在第九节的老年报
    assert classify_section(_chunk("第九节 财务报告/一、审计报告")) == "notes"


def test_financial_report_phrase_in_other_section_is_not_notes():
    """「与上年度财务报告相比」含关键词但属重要事项，不能误归 notes。"""
    c = _chunk("第六节 重要事项/六、与上年度财务报告相比，合并报表范围发生变化的情况说明")
    assert classify_section(c) == "md_narrative"


def test_fallback_md_narrative():
    assert classify_section(_chunk("第四节 公司治理/一、公司治理的基本状况")) == "md_narrative"
    assert classify_section(_chunk("")) == "md_narrative"


def test_node_sectionize_fills_all_chunks(runtime: Runtime[DataAgentContext]):
    """节点就地填充 section，不增删 chunk。"""
    chunks = [
        _chunk("第三节 管理层讨论与分析/四、主营业务分析"),
        _chunk("第十节 财务报告/五、重要会计政策"),
        _chunk("第四节 公司治理/一、公司治理的基本状况"),
        _chunk("任意", chunk_type="table"),
    ]
    patch = node_sectionize({"chunks": chunks}, runtime)
    out = patch["chunks"]
    assert len(out) == 4
    assert all(c["section"] for c in out)
    assert out[0]["section"] == "management_discussion"
    assert out[1]["section"] == "notes"
    assert out[2]["section"] == "md_narrative"
    assert out[3]["section"] == "table_desc"


def test_node_sectionize_idempotent(runtime: Runtime[DataAgentContext]):
    """重跑结果不变——纯函数，同输入同输出。"""
    chunks = [_chunk("第三节 管理层讨论与分析/四、主营业务分析")]
    once = node_sectionize({"chunks": [dict(chunks[0])]}, runtime)["chunks"]
    twice = node_sectionize({"chunks": [dict(once[0])]}, runtime)["chunks"]
    assert once[0]["section"] == twice[0]["section"] == "management_discussion"
