"""S5a 编码串测试：格式、主体降级、指代替换边界（docs/02c §3/§4）。"""
from app.import_graph.encode import build_encode_text, resolve_pronoun


def test_encode_text_with_stock_name():
    chunk = {"section": "management_discussion", "content": "本集团2024年度实现营业收入100元"}
    assert (
        build_encode_text(chunk, "神州高铁", "000008")
        == "神州高铁·管理层讨论：本集团2024年度实现营业收入100元"
    )


def test_encode_text_fallback_to_stock_code():
    chunk = {"section": "risk", "content": "公司可能面临的风险如下"}
    assert (
        build_encode_text(chunk, "", "000008")
        == "000008·风险因素：公司可能面临的风险如下"
    )


def test_encode_text_no_subject():
    chunk = {"section": "notes", "content": "七、合并财务报表项目注释"}
    assert (
        build_encode_text(chunk, "", "")
        == "财务附注：七、合并财务报表项目注释"
    )


def test_encode_text_unknown_section_uses_fallback_cn():
    chunk = {"content": "正文"}
    assert build_encode_text(chunk, "新世界", "600628") == "新世界·报告正文：正文"


def test_encode_text_does_not_mutate_chunk():
    chunk = {"section": "management_discussion", "content": "本集团实现营收"}
    build_encode_text(chunk, "神州高铁", "000008", resolve=True)
    assert chunk["content"] == "本集团实现营收"


def test_resolve_pronoun_group_and_company():
    text = "本集团及本公司合计实现营收"
    assert resolve_pronoun(text, "神州高铁") == "神州高铁集团及神州高铁合计实现营收"


def test_resolve_pronoun_keeps_compound_company():
    text = "子公司和上市公司和公司章程不受影响"
    assert resolve_pronoun(text, "新世界") == text


def test_resolve_pronoun_skips_generic_and_auditor_voice():
    # 该公司=会计政策泛指、我们=审计师口吻（docs/02c §3.2）
    text = "该公司是以向投资者提供投资管理服务为目的。我们认为财务报表公允反映"
    assert resolve_pronoun(text, "新世界") == text


def test_resolve_pronoun_empty_stock_name():
    text = "本公司实现营收"
    assert resolve_pronoun(text, "") == text


def test_build_encode_text_resolve_only_on_encode_string():
    chunk = {"section": "management_discussion", "content": "本集团实现营收，子公司不受影响"}
    out = build_encode_text(chunk, "神州高铁", "000008", resolve=True)
    assert out == "神州高铁·管理层讨论：神州高铁集团实现营收，子公司不受影响"
