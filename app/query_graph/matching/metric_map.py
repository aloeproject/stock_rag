"""node_metric_query 纯规则（docs/04f §3.2/§5.1）。

词→码三层映射、口径裁决优先级。
为什么不走 text-to-SQL：LLM 写码必错，词→码是查表（04f §6）。
"""
from __future__ import annotations

# 只认 L1/L2：L3 码不在 financial_metrics，映射到了也查空（04f §3.2）
_METRIC_LEVELS = (1, 2)

# 口径裁决优先级（04f §5.1）：合并>母公司；追溯>本期>上期
SCOPE_PRIORITY = ("consolidated", "parent")
PERIOD_TYPE_PRIORITY = ("restated", "current", "comparative")

# 包含匹配的最短词长：「毛利」这类碎片词误中面太大
_MIN_CONTAINS_LEN = 2


def map_metric_words(words: list[str], definitions: list) -> tuple[list[str], list[str]]:
    """通俗词 → metric_code；返回 (codes, unmapped)。

    三层：精确 name → 别名 → 包含。别名在前——「归母净利润」是别名，
    不该被「净利润」的包含匹配截胡（04f §3.2）。
    精确/别名命中 L3 直接拒：「营业收入同比」是要 YoY 不是营收，不能被包含匹配捞去 L1。
    """
    defs = list(definitions or [])
    usable = [d for d in defs if int(getattr(d, "metric_level", 0)) in _METRIC_LEVELS]
    codes: list[str] = []
    unmapped: list[str] = []
    for word in dict.fromkeys(w for w in words if w):
        if _hits_l3_name_or_alias(word, defs):
            unmapped.append(word)
            continue
        code = _match_one(word, usable)
        if code is None:
            unmapped.append(word)
        elif code not in codes:
            codes.append(code)
    return codes, unmapped


def _hits_l3_name_or_alias(word: str, definitions: list) -> bool:
    for d in definitions:
        if int(getattr(d, "metric_level", 0)) in _METRIC_LEVELS:
            continue
        if d.metric_name == word:
            return True
        if word in (d.aliases or "").split("|"):
            return True
    return False


def _match_one(word: str, definitions: list) -> str | None:
    for d in definitions:
        if d.metric_name == word:
            return d.metric_code
    for d in definitions:
        aliases = (d.aliases or "").split("|")
        if word in aliases:
            return d.metric_code
    if len(word) >= _MIN_CONTAINS_LEN:
        # 两层包含分开：先「词是 name 的碎片」取最短 name（最贴近），没有再
        # 「name 是词的碎片」取最长 name（最具体）。混作一层取最长会让
        # 「净利」命中 17 字扣非全称（尾部含「净利」），见 test_contains_word_in_name_shortest。
        best_code, best_len = None, 0
        for d in definitions:
            name = d.metric_name or ""
            if word in name:
                if best_code is None or len(name) < best_len:
                    best_code, best_len = d.metric_code, len(name)
        if best_code is not None:
            return best_code
        for d in definitions:
            name = d.metric_name or ""
            if name and name in word:
                if best_code is None or len(name) > best_len:
                    best_code, best_len = d.metric_code, len(name)
        return best_code
    return None


def pick_metric_row(rows: list[dict]) -> dict | None:
    """同 (stock_code, metric_code, metric_period) 多行取 1：scope 再 period_type，仍并列取 id 最小。

    先取全再降级而不是 WHERE 写死：母公司独有指标不能查空（04f §3.4）。
    """
    if not rows:
        return None

    def _rank(r: dict) -> tuple:
        scope = str(r.get("statement_scope") or "")
        ptype = str(r.get("metric_period_type") or "")
        return (
            SCOPE_PRIORITY.index(scope) if scope in SCOPE_PRIORITY else len(SCOPE_PRIORITY),
            PERIOD_TYPE_PRIORITY.index(ptype) if ptype in PERIOD_TYPE_PRIORITY else len(PERIOD_TYPE_PRIORITY),
            int(r.get("id") or 0),
        )

    return min(rows, key=_rank)
