from __future__ import annotations

import re

from langgraph.runtime import Runtime
from langgraph.types import StreamWriter

from app.core.logger import logger
from app.import_graph.context import DataAgentContext
from app.import_graph.nodes import node_split
from app.import_graph.nodes.progress import progress_step
from app.import_graph.state import ImportState
from app.utils.file import sha1_file
from app.utils.path_util import PROJECT_ROOT

# 为什么关键词而不是 LLM：A 股年报 L1 节名由证监会格式准则强制（docs/02b §2），
# 5 个粗类 + 标题关键词即可覆盖；LLM 每 chunk 一次调用且结果会漂，成本/确定性都不划算。
# 规则顺序 = 匹配优先级，见 docs/02b §3.2
MANAGEMENT_KEYWORDS = (
    "管理层讨论与分析", "经营情况讨论", "报告期内主要经营",
)
# notes 用 L1 节名锚定：「与上年度财务报告相比」含关键词但属重要事项，不能误伤
NOTES_L1_RE = re.compile(r"第[0-9零一二三四五六七八九十百千]+[节章部分].*财务报告")
NOTES_KEYWORDS = (
    "财务报表附注", "合并财务报表", "报表项目注释",
)


SECTION_TABLE = "table_desc" #表格说明
SECTION_RISK = "risk" #风险因素
SECTION_MANAGEMENT = "management_discussion"
SECTION_NOTES = "notes" #财务附注
SECTION_FALLBACK = "md_narrative" #报告正文


def classify_section(chunk: dict) -> str:
    """section_path → 标准 section。未命中落 md_narrative，不抛错。

    匹配 section_path 而非 title：路径带 L1 祖先，能区分
    「与金融工具相关的风险」（财务报告下=notes）和「可能面临的风险」（=risk）。
    """
    if chunk.get("chunk_type") == "table":
        return SECTION_TABLE

    path = chunk.get("section_path") or chunk.get("title") or ""

    # notes 先于 risk：财务报告下的「与金融工具相关的风险」是附注不是业务风险
    l1 = path.split("/", 1)[0]
    if NOTES_L1_RE.match(l1) or any(k in path for k in NOTES_KEYWORDS):
        return SECTION_NOTES

    # 用「风险」二字兜底而非精确短语：「（二）可能面临的风险」常是空标题壳，
    # 子节 path 只剩「1、宏观环境风险」，父级短语丢了也得归 risk
    if "风险" in path:
        return SECTION_RISK

    if any(k in path for k in MANAGEMENT_KEYWORDS):
        return SECTION_MANAGEMENT

    return SECTION_FALLBACK


def node_sectionize(state: ImportState,runtime: Runtime[DataAgentContext], writer: StreamWriter | None = None) -> dict:
    """S4 章节归类：就地填充 chunks 的 section，不增删、不落库。

    归类是检索过滤维度不是数据质量门禁——单块异常落兜底继续，
    不存在节点级失败（docs/02b §6）。
    """
    logger.info(f"[node_sectionize] 开始处理文档: {state.get('document_id')}")
    with progress_step(writer, "章节归类"):
        chunks = state.get("chunks") or []
        for c in chunks:
            try:
                c["section"] = classify_section(c)
            except Exception:
                logger.warning(f"[node_sectionize] 归类异常，落兜底: chunk_id={c.get('chunk_id')}")
                c["section"] = SECTION_FALLBACK
        return {"chunks": chunks}


if __name__ == "__main__":
    pdf_path = PROJECT_ROOT / "report" / "000006_2024_深振业A_2024年年度报告_2025-04-26.pdf"
    document_id = sha1_file(pdf_path)
    path_root = PROJECT_ROOT / "data" / "parse" / "45d1b670ffb64d02650010648076fdb14aa01d42"
    markdown_path = (
            path_root / "content.md"
    )
    # parse.py 落盘的是 pages.json，不是 pages_map.json（state 键名叫 pages_map_path）
    pages_map_path = path_root / "pages.json"
    state: ImportState = {
        "pdf_path": str(pdf_path),
        "markdown_path": str(markdown_path),
        "pages_map_path": str(pages_map_path),
        "document_id": document_id,
    }  # type: ignore[typeddict-item]
    tmp = node_split(state, None)
    state['chunks'] = tmp['chunks']
    node_sectionize(state,None)