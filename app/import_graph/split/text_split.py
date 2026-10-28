from __future__ import annotations

from app.import_graph.split.token_estimator import estimate_token_count

# 段落优先 → 换行 → 句末 → 空格，保证切口落在语义边界（docs/02a §3.2）
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "：", " "]


def estimate_char_budget(token_budget: int) -> int:
    """token 预算 → 字符预算。

    中文年报 CJK 为主，1 token ≈ 1 字符；英文混排按估算器校正。
    splitter 收字符长度，这里做保守换算（宁可略小不超切）。
    """
    return max(1, int(token_budget * 0.9))


def split_text_recursive(
    text: str,
    max_chars: int,
    overlap_chars: int,
    separators: list[str] | None = None,
) -> list[str]:
    """递归按分隔符切分（自实现，零依赖；语义对齐 RecursiveCharacterTextSplitter）。

    为什么自实现：项目未引入 langchain_text_splitters，且本场景只需
    段落/句子级切口，百行内可测可控。
    """
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    seps = separators if separators is not None else DEFAULT_SEPARATORS
    for i, sep in enumerate(seps):
        if sep == "":
            return _hard_split(text, max_chars, overlap_chars)
        if sep not in text:
            continue
        pieces = text.split(sep)
        merged = _merge_pieces(pieces, sep, max_chars, overlap_chars)
        if all(len(p) <= max_chars for p in merged):
            # 段落一次切开也要 overlap——漏掉这步续块边界就是硬切口
            return _apply_overlap(merged, overlap_chars)
        # 仍有超长块 → 用更细的分隔符递归
        rest_seps = seps[i + 1:]
        out: list[str] = []
        for p in merged:
            if len(p) <= max_chars:
                out.append(p)
            else:
                out.extend(split_text_recursive(p, max_chars, overlap_chars, rest_seps))
        return _apply_overlap(out, overlap_chars)
    return _hard_split(text, max_chars, overlap_chars)


def _merge_pieces(pieces: list[str], sep: str, max_chars: int, overlap_chars: int) -> list[str]:
    """把碎段拼到接近 max_chars，拼不上就落盘成块。"""
    out: list[str] = []
    buf = ""
    for p in pieces:
        candidate = f"{buf}{sep}{p}" if buf else p
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def _hard_split(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """分隔符用尽仍超长：按字符硬切，内容不丢。"""
    step = max(1, max_chars - overlap_chars)
    return [text[i:i + max_chars] for i in range(0, len(text), step)]


def _apply_overlap(chunks: list[str], overlap_chars: int) -> list[str]:
    if overlap_chars <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev[-overlap_chars:] if len(prev) >= overlap_chars else prev
        if not cur.startswith(tail):
            cur = tail + cur
        out.append(cur)
    return out
