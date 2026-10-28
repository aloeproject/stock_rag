from __future__ import annotations

import re

# 非贪婪：一文多表；DOTALL 让 <table> 内换行被吃掉
_TABLE_RE = re.compile(r"<table>.*?</table>", re.DOTALL)
_CELL_RE = re.compile(r"<(td|th)([^>]*)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
_ATTR_RE = re.compile(r"(rowspan|colspan)\s*=\s*['\"]?(\d+)", re.IGNORECASE)
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)


def extract_tables(md_content: str) -> list[dict]:
    """抽出全部 <table> 区域，记录在原文中的 char 偏移（溯源用）。"""
    tables = []
    for m in _TABLE_RE.finditer(md_content):
        tables.append({
            "char_start": m.start(),
            "char_end": m.end(),
            "html": m.group(),
        })
    return tables


def strip_tables(md_content: str) -> str:
    """把 <table> 区域替换为等长空白占位。

    为什么等长：叙述 chunk 的 char_start/end 相对原始 content.md，
    抠表后长度不变，溯源偏移才不漂移。
    """
    def _blank(m: re.Match) -> str:
        return " " * (m.end() - m.start())

    return _TABLE_RE.sub(_blank, md_content)


def _cell_attrs(attr_text: str) -> tuple[int, int]:
    rowspan, colspan = 1, 1
    for name, val in _ATTR_RE.findall(attr_text):
        n = max(1, int(val))
        if name.lower() == "rowspan":
            rowspan = n
        else:
            colspan = n
    return rowspan, colspan


def _cell_text(inner: str) -> str:
    # 去掉内部标签（如 <b>），保留可见文本
    text = re.sub(r"<[^>]+>", "", inner)
    return text.replace("\n", " ").strip()


def html_table_to_markdown(html: str) -> str:
    """HTML 表 → Markdown 表。

    rowspan/colspan 展开为多格：内容留首个单元格、其余填空——
    不丢格子数，保证行列对齐（docs/02a §4.1）。
    """
    rows_html = _ROW_RE.findall(html)
    grid: list[list[str]] = []
    # 跨行待填充：col -> (剩余行数, 文本)
    pending: dict[int, tuple[int, str]] = {}

    for row_html in rows_html:
        row: list[str] = []
        col = 0
        cells = _CELL_RE.findall(row_html)
        for _tag, attrs, inner in cells:
            # 先填入上方 rowspan 延伸下来的格子
            while col in pending:
                rem, text = pending[col]
                row.append(text)
                rem -= 1
                if rem <= 0:
                    del pending[col]
                else:
                    pending[col] = (rem, text)
                col += 1

            rowspan, colspan = _cell_attrs(attrs)
            text = _cell_text(inner)
            for cs in range(colspan):
                row.append(text if cs == 0 else "")
                if rowspan > 1:
                    # 后续行对应格填空（内容已留首行首格）
                    pending[col] = (rowspan - 1, "")
                col += 1

        while col in pending:
            rem, text = pending[col]
            row.append(text)
            rem -= 1
            if rem <= 0:
                del pending[col]
            else:
                pending[col] = (rem, text)
            col += 1

        grid.append(row)

    if not grid:
        return ""

    width = max(len(r) for r in grid)
    for r in grid:
        while len(r) < width:
            r.append("")

    lines = ["| " + " | ".join(grid[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for r in grid[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)
