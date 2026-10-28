"""stock_dict 四级名称→代码匹配（docs/04a §4）。

为什么是本地表不是 MCP：personal-assistant 的 search_stock_symbols 实测不可用
（中文/英文/代码查询均返回 [ERROR]）；本地表离线可靠、可单测、可 SQL 审计。
为什么不用模糊相似度：歧义不可解释、阈值难定；精确+唯一前缀每一级可断言（04a §4.2）。
"""
from __future__ import annotations

import re

from app.entities.stock_dict import StockDictDbData

# 前缀/包含最短长度：1 字（如「茅」）歧义爆炸（04a §2）
MIN_PREFIX_LEN = 2

_CODE_RE = re.compile(r"^\d{6}$")


def find_candidates(word: str, rows: list[StockDictDbData]) -> list[str]:
    """返回可用于追问的 stock_name 列表（去重保序）。

    未识别原词 → []；前缀/包含命中多家 → 对应 stock_name。
    与 _match_one 同一套命中条件，避免「匹配说歧义、候选却另一套」。
    """
    word = (word or "").strip()
    if not word:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for r in rows:
        if r.status != 1 or not _row_hit(word, r):
            continue
        if r.stock_name not in seen:
            seen.add(r.stock_name)
            out.append(r.stock_name)
    return out


def match_stock_names(
    names: list[str], rows: list[StockDictDbData]
) -> tuple[dict[str, str], list[str]]:
    """名称 → (resolved: name→code, unresolved: 原词列表)。

    四级短路：stock_name 精确 → aliases 精确 → code 精确 → 唯一前缀/包含。
    前缀命中多行进 unresolved——宁可追问，禁止静默取第一条。
    只认 status=1；停用股视为未命中。
    """
    active = [r for r in rows if r.status == 1]
    resolved: dict[str, str] = {}
    unresolved: list[str] = []

    for name in names:
        word = (name or "").strip()
        if not word:
            continue
        code = _match_one(word, active)
        if code:
            resolved[word] = code
        else:
            unresolved.append(word)
    return resolved, unresolved


def _match_one(word: str, rows: list[StockDictDbData]) -> str | None:
    # 1. stock_name 精确
    for r in rows:
        if r.stock_name == word:
            return r.stock_code

    # 2. aliases 元素精确
    for r in rows:
        if word in (r.aliases or []):
            return r.stock_code

    # 3. 6 位代码精确（仍要求字典有记录，防输错号）
    if _CODE_RE.match(word):
        for r in rows:
            if r.stock_code == word:
                return r.stock_code
        return None

    # 4. 前缀/包含，≥2 字且全库唯一
    if len(word) < MIN_PREFIX_LEN:
        return None
    hits = [r for r in rows if _contains_hit(word, r)]
    codes = {r.stock_code for r in hits}
    return hits[0].stock_code if len(codes) == 1 else None


def _row_hit(word: str, r: StockDictDbData) -> bool:
    """find_candidates 的命中面：与 _match_one 四级同一套条件，只答「像不像」不答「是不是唯一」。"""
    if r.stock_name == word or word in (r.aliases or []):
        return True
    if _CODE_RE.match(word) and r.stock_code == word:
        return True
    return len(word) >= MIN_PREFIX_LEN and _contains_hit(word, r)


def _contains_hit(word: str, r: StockDictDbData) -> bool:
    return (
        word in r.stock_name
        or word in (r.full_name or "")
        or any(word in a for a in (r.aliases or []))
    )
