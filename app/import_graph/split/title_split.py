from __future__ import annotations

import re

# 为什么不用 Markdown 标题层级：MinerU 产出的年报 content.md 705 个标题几乎全是 ##，
# 层级信息失效；改用中文公文序号推断（docs/02a §3.1）
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_L1_RE = re.compile(r"^第[0-9零一二三四五六七八九十百千]+[节章部分]")
_L2_RE = re.compile(r"^[一二三四五六七八九十百]+、")
_L3_RE = re.compile(r"^(（[一二三四五六七八九十]+）|\([一二三四五六七八九十]+\)|\d+[、.．])")

FALLBACK_TITLE = "文档正文"


def clean_heading(text: str) -> str:
    """去掉 # 前缀，保留标题正文（section_path 用）。"""
    m = _HEADING_RE.match(text.strip())
    return m.group(2).strip() if m else text.strip()


def infer_level(heading_text: str) -> int:
    """中文序号 → 层级。0=无序号（挂最近祖先）。"""
    text = heading_text.strip()
    if _L1_RE.match(text):
        return 1
    if _L2_RE.match(text):
        return 2
    if _L3_RE.match(text):
        return 3
    return 0


def _push_stack(stack: list[tuple[int, str]], level: int, text: str) -> list[tuple[int, str]]:
    """按层级维护祖先栈：真层级弹出同级及更深节点；0 级不入栈（临时子节点）。"""
    if level == 0:
        return stack
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, text))
    return stack


def _section_path(stack: list[tuple[int, str]], title_text: str, level: int) -> str:
    if level == 0:
        parts = [t for _, t in stack] + [title_text]
    else:
        parts = [t for _, t in stack]
    return "/".join(parts)


def _find_parent(stack: list[tuple[int, str]], level: int, file_title: str) -> str:
    """父 = 最近的更高级祖先；同级前兄弟不是父。

    为什么不能取栈顶：L3 兄弟入栈前栈顶就是前兄弟，各挂各的父后
    same_parent 永远不成立，短块合并不了（真实年报「4、研发投入」被
    挂到「3、费用」就是这么来的）。
    """
    if level == 0:
        return stack[-1][1] if stack else file_title
    for lv, text in reversed(stack):
        if lv < level:
            return text
    return file_title


def split_by_title(md_content: str, file_title: str = "default_file") -> list[dict]:
    """按标题粗切为章节块，带 section_path / char_start / char_end。

    代码块内的伪标题跳过（年报无代码块，保留防御，沿用参考实现）。
    """
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

    sections: list[dict] = []
    stack: list[tuple[int, str]] = []
    current_title = ""
    current_text = ""
    current_path = ""
    current_parent = file_title
    current_start = 0
    in_code = False
    offset = 0

    def _flush(end: int) -> None:
        nonlocal current_title, current_text
        if not current_title:
            return
        sections.append({
            "title": current_title,
            "content": current_text,
            "file_title": file_title,
            # 粗切即产出 parent_title：合并阶段靠它判断同级（docs/02a §3.1）
            "parent_title": current_parent,
            "section_path": current_path,
            "char_start": current_start,
            "char_end": end,
        })
        current_title = ""
        current_text = ""

    for line in md_content.split("\n"):
        line_end = offset + len(line)
        strip_line = line.strip()

        if strip_line.startswith("```") or strip_line.startswith("~~~"):
            in_code = not in_code
            if current_title:
                current_text += line + "\n"
            offset = line_end + 1
            continue

        is_heading = bool(_HEADING_RE.match(strip_line)) and not in_code
        if is_heading:
            _flush(offset)
            raw_title = strip_line
            title_text = clean_heading(raw_title)
            level = infer_level(title_text)
            # 先取父再入栈：父是最近更高级祖先，不是同级前兄弟
            current_parent = _find_parent(stack, level, file_title)
            _push_stack(stack, level, title_text)
            current_title = raw_title
            current_path = _section_path(stack, title_text, level)
            current_text = raw_title + "\n"
            current_start = offset
        elif current_title:
            current_text += line + "\n"
        offset = line_end + 1

    _flush(len(md_content.rstrip()))

    if not sections:
        sections.append({
            "title": FALLBACK_TITLE,
            "content": md_content,
            "file_title": file_title,
            "parent_title": file_title,
            "section_path": FALLBACK_TITLE,
            "char_start": 0,
            "char_end": len(md_content.rstrip()),
        })

    return sections
