from __future__ import annotations

import math
import re

# ASCII 词按 BPE 常被切成多 token，系数 1.3；CJK 一字≈一 token
# 为什么不用 tiktoken/transformers：依赖重、全量年报 tokenize 慢，
# 且 ±10% 误差对 500 粒度足够（docs/02a §6）
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def estimate_token_count(text: str) -> int:
    """确定性 token 估算：ceil(CJK 字符数 × 1.0 + ASCII 词数 × 1.3)。"""
    if not text:
        return 0

    ascii_words = len(_ASCII_WORD_RE.findall(text))
    # 剔除 ASCII 词后，其余非空字符（CJK、全角/半角标点）按 1 计
    non_ascii = _ASCII_WORD_RE.sub("", text)
    other_chars = sum(1 for ch in non_ascii if not ch.isspace())

    return math.ceil(other_chars + ascii_words * 1.3)
