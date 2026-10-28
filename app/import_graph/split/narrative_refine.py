from __future__ import annotations

import re

from app.import_graph.split.config import SplitConfig
from app.import_graph.split.text_split import (
    estimate_char_budget,
    split_text_recursive,
)
from app.import_graph.split.token_estimator import estimate_token_count

_SPACE_RUN_RE = re.compile(r"[ \t]+")


def _normalize_whitespace(text: str) -> str:
    """清掉 strip_tables 等长空白垫：行内空串折叠、行尾空白、连续空行。

    为什么必须清：strip_tables 用等长空白占位保 char offset，这些空白会
    灌进叙述 content（实测 1939 字只算 16 token 的短块就是这么来的），
    不清会污染合并判断和检索可读性。
    """
    collapsed = [_SPACE_RUN_RE.sub(" ", ln).rstrip() for ln in text.split("\n")]
    out: list[str] = []
    blank = False
    for ln in collapsed:
        if not ln.strip():
            if blank:
                continue
            out.append("")
            blank = True
        else:
            out.append(ln)
            blank = False
    return "\n".join(out).strip()


def _parent_from_section_path(section: dict) -> str:
    """section_path 倒数第二段即父标题；顶层章节退回 file_title。

    为什么从 path 反推：title_split 早期版本没带 parent_title，
    而 path 一直可靠——合并前补上真实父，同级短块才并得起来。
    """
    parts = [p for p in (section.get("section_path") or "").split("/") if p]
    if len(parts) >= 2:
        return parts[-2]
    return section.get("file_title") or ""


def split_long_section(section: dict, cfg: SplitConfig) -> list[dict]:
    """超长章节 → 段落递归细切。子块 title=原标题_i，parent_title 保留。

    为什么保留 parent_title：合并阶段靠它判断「同级可并」，也供检索溯源。
    """
    content = section.get("content") or ""
    if estimate_token_count(content) <= cfg.max_token:
        return [section]

    max_chars = estimate_char_budget(cfg.max_token)
    overlap_chars = estimate_char_budget(cfg.overlap_token)
    pieces = split_text_recursive(content, max_chars, overlap_chars)

    parent = section.get("parent_title") or section.get("title") or ""
    out = []
    for index, piece in enumerate(pieces, start=1):
        text = piece.strip()
        if not text:
            continue
        out.append({
            "title": f"{section.get('title')}_{index}",
            "content": text,
            "parent_title": parent,
            "part": index,
            "section_path": section.get("section_path", ""),
            "char_start": section.get("char_start"),
            "char_end": section.get("char_end"),
            "file_title": section.get("file_title"),
        })
    return out or [section]


def merge_short_sections(sections: list[dict], cfg: SplitConfig) -> list[dict]:
    """过短块同级合并。仅 parent_title 相同才并，合并后不超过 hard_max_token。

    为什么收紧硬上限：参考实现允许微超，但长句/大表场景会失控（docs/02a §3.2）。
    """
    if not sections:
        return []

    merged: list[dict] = []
    prev = sections[0]

    for cur in sections[1:]:
        prev_tokens = estimate_token_count(prev.get("content") or "")
        cur_tokens = estimate_token_count(cur.get("content") or "")
        same_parent = bool(prev.get("parent_title")) and (
            prev.get("parent_title") == cur.get("parent_title")
        )
        # 短 prev 或短 cur 都要并：只看 prev 会漏掉「长块+短尾巴」
        if same_parent and (prev_tokens < cfg.min_token or cur_tokens < cfg.min_token):
            if prev_tokens + cur_tokens <= cfg.hard_max_token:
                prev["content"] = prev["content"] + "\n\n" + cur["content"]
                prev["part"] = cur.get("part", prev.get("part", 1))
                continue
        merged.append(prev)
        prev = cur

    merged.append(prev)
    return merged


def refine_sections(sections: list[dict], cfg: SplitConfig) -> list[dict]:
    """细化总入口：清空白垫 → 补父标题 → 切长 → 合短。

    parent_title 必须在 merge 前补齐：merge 靠它判同级，
    事后 setdefault 会写成自身标题，同级短块永远并不上。
    """
    prepared: list[dict] = []
    for sec in sections:
        item = dict(sec)
        item["content"] = _normalize_whitespace(item.get("content") or "")
        if not item.get("parent_title"):
            item["parent_title"] = _parent_from_section_path(item)
        prepared.append(item)

    final: list[dict] = []
    for sec in prepared:
        final.extend(split_long_section(sec, cfg))
    final = merge_short_sections(final, cfg)
    for sec in final:
        sec.setdefault("part", 1)
    return final
