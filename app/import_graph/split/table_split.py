from __future__ import annotations

import re

from app.import_graph.split.config import SplitConfig
from app.import_graph.split.table_extract import html_table_to_markdown
from app.import_graph.split.token_estimator import estimate_token_count

# 小计/合计/总计是自然语义边界：切开不会把明细和它的汇总拆到两块
_SUBTOTAL_RE = re.compile(r"小计|合计|总计|总额")
_UNIT_RE = re.compile(r"单位\s*[:：]|币种\s*[:：]")


def extract_caption(md_content: str, table: dict) -> str:
    """取表格正上方的标题行 + 计量说明。

    为什么归 caption 不归叙述 chunk：避免「表题/单位」在叙述与表格两侧重复。
    标题与表之间可能隔着目录行/多段说明（年报实测），扫描窗口放宽到 500 字符，
    且从下往上找最近标题——目录类表格也要能挂上标题，不能留空。
    """
    before = md_content[max(0, table["char_start"] - 500): table["char_start"]]
    lines = [ln.strip() for ln in before.splitlines() if ln.strip()]

    caption_lines: list[str] = []
    unit_line = ""
    heading = ""
    # 从下往上扫：先收单位行，再找最近标题
    for ln in reversed(lines):
        if _UNIT_RE.search(ln):
            if not unit_line:
                unit_line = ln
            continue
        if ln.startswith("#"):
            heading = ln.lstrip("#").strip()
            break
    if heading:
        caption_lines.append(heading)
    if unit_line:
        caption_lines.append(unit_line)
    return "\n".join(caption_lines)


def _row_tokens(row: list[str]) -> int:
    """按 Markdown 序列化后的实测 token 计费。

    为什么不用 `" ".join(row)`：实测宽表低估约 30%（| 分隔符、单元格空格、
    数字里的逗号分段），导致行组膨胀到 2 倍 max（真实年报冒烟 max=1004）。
    """
    return estimate_token_count("| " + " | ".join(row) + " |")


def split_table_rows(
    rows: list[list[str]],
    header: list[str],
    max_token: int,
    est=None,
) -> list[list[list[str]]]:
    """按行组切大表，小计/合计/总计行后断开。每组独立可读。

    每片都带完整表头，所以每组预算都要计表头开销——分组后归零不重计，
    第二组起必然超限（回归测试 test_second_group_still_budgets_header）。
    """
    if est is None:
        est = _row_tokens

    header_tokens = est(header)
    groups: list[list[list[str]]] = []
    current: list[list[str]] = []
    current_tokens = header_tokens

    for row in rows:
        row_tokens = est(row)
        if current and current_tokens + row_tokens > max_token:
            groups.append(current)
            current = []
            current_tokens = header_tokens  # 新组重新计表头
        current.append(row)
        current_tokens += row_tokens
        # 小计/合计行后强制断开：明细与其汇总同组
        if any(_SUBTOTAL_RE.search(c) for c in row):
            groups.append(current)
            current = []
            current_tokens = header_tokens

    if current:
        groups.append(current)
    return groups


def _serialize(caption: str, section_path: str, header: list[str], rows: list[list[str]]) -> str:
    """Markdown 序列化（docs/02a §4.3）：caption + section_path + 表。"""
    width = max([len(header)] + [len(r) for r in rows]) if rows else len(header)
    header = header + [""] * (width - len(header))
    lines = []
    if caption:
        lines.append(caption)
    if section_path:
        lines.append(section_path)
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * width) + " |")
    for r in rows:
        r = r + [""] * (width - len(r))
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def build_table_chunks(
    html: str,
    table_id: str,
    caption: str,
    section_path: str,
    char_start: int,
    char_end: int,
    cfg: SplitConfig,
) -> list[dict]:
    """整表或按行组产 table chunk。每片带完整表头——单独命中即可读懂。"""
    md = html_table_to_markdown(html)
    lines = [ln for ln in md.splitlines() if ln.strip()]
    if not lines:
        return []

    def _parse_row(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    header = _parse_row(lines[0])
    body_lines = lines[2:] if len(lines) >= 2 and set(lines[1].replace("|", "").replace(" ", "")) <= {"-"} else lines[1:]
    rows = [_parse_row(ln) for ln in body_lines]

    total_tokens = estimate_token_count(md)
    if total_tokens <= cfg.max_token:
        groups = [rows] if rows else []
    else:
        # 行组预算让出 caption+section_path 开销，否则序列化后仍会超 max
        overhead = estimate_token_count(
            "\n".join(filter(None, [caption, section_path]))
        )
        row_budget = max(1, cfg.max_token - overhead)
        groups = split_table_rows(rows, header, row_budget)

    out = []
    for i, group in enumerate(groups, start=1):
        content = _serialize(caption, section_path, header, group)
        tokens = estimate_token_count(content)
        item = {
            "chunk_type": "table",
            "content": content,
            "title": caption.splitlines()[0] if caption else table_id,
            "parent_title": caption.splitlines()[0] if caption else "",
            "section_path": section_path,
            "part": i,
            "table_id": table_id,
            "table_caption": caption,
            "char_start": char_start,
            "char_end": char_end,
            "token_count": tokens,
        }
        # 单行本身超 hard_max 时无法再切（切断行=丢表头-格语义），保留完整并标记
        if tokens > cfg.hard_max_token:
            item["oversized"] = True
        out.append(item)
    return out
