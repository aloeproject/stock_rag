"""S5a 编码串生成（docs/02c）。

为什么注入前缀而不是 LLM 指代消解：指代词模板化（本公司/本集团），
改写有幻觉风险且每报数百次调用（docs/02c §2）。
为什么改编码串不改 content：检索命中后要原样引用/溯源（docs/02c §1.2）。
为什么用 stock_name 不用 stock_code：数字串无语义，简称能进向量空间（docs/02c §3.1）。
"""
from __future__ import annotations

# 02b §3.1 的 section 中文名；缺失/未知落「报告正文」与 classify_section 兜底一致
SECTION_CN = {
    "management_discussion": "管理层讨论",
    "risk": "风险因素",
    "notes": "财务附注",
    "table_desc": "表格说明",
    "md_narrative": "报告正文",
}

# 只替换词义确定的自指词。禁止加入「该公司/我们」——实测是泛指/审计师口吻，
# 替换即改错主语（docs/02c §3.2，四报统计见 §1.1）
PRONOUN_MAP = (
    ("本集团", "{name}集团"),
    ("本公司", "{name}"),
)


def resolve_pronoun(text: str, stock_name: str) -> str:
    """只替换确定自指的指代词，不动「子公司/上市公司」和「该公司」。"""
    if not stock_name:
        return text
    for src, tpl in PRONOUN_MAP:
        text = text.replace(src, tpl.format(name=stock_name))
    return text


def build_encode_text(
    chunk: dict,
    stock_name: str,
    stock_code: str,
    resolve: bool = False,
) -> str:
    """S5a 编码串：主体 + 段落类型 + 原文。只用于 embedding，不回写 chunk。

    为什么拼前缀而不是改写 content：原文必须保持可引用（docs/02c §1.2）。
    resolve=True 时只替换编码串上的指代词，chunk["content"] 保持不变。
    """
    subject = stock_name or stock_code or ""
    section_cn = SECTION_CN.get(chunk.get("section"), "报告正文")
    content = chunk.get("content") or ""
    if resolve and stock_name:
        content = resolve_pronoun(content, stock_name)
    if not subject:
        return f"{section_cn}：{content}"
    return f"{subject}·{section_cn}：{content}"
