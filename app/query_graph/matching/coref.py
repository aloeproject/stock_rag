"""查询侧指代消解（docs/04a §6）。

只处理多轮对话里的「它/他们/上一家」→ 历史 stock_codes。
导入侧「本公司/本集团」是另一类问题（02c 编码前缀），不要混。

为什么规则兜底而不是只信 LLM：test_queries.json 的 context_dependent
20 条全靠历史消解；LLM 偶尔漏改写，下游 plan 会拆出无主语子问题。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# 单数 → 取最近 1 个；复数 → 取全部；序数 → 倒数第 2 条有 codes 的 user
_SINGULAR_LONG = ("该公司", "这家")
_PLURAL = ("这两家", "他们", "它们", "这些")
_ORDINAL = ("上一家", "前一个")

# 「它/其」不能是「其它/其他」的一部分——那是指「别的」，不是公司指代
_PRONOUN_RE = re.compile(r"(?<!其)它|(?<!其)其(?![他它])")

# 「他们」若在讲子公司，不能当股票代码（04a §6.3）
_SUBSIDIARY_MARKS = ("子公司", "旗下")


@dataclass
class CorefResult:
    trigger: str | None = None
    codes: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    rewritten_query: str | None = None


def resolve_coreference(query: str, history: list[dict]) -> CorefResult:
    """扫描触发词并从 history 回填主体；无触发词返回空结果。"""
    word = query or ""
    trigger = _find_trigger(word)
    if trigger is None:
        return CorefResult()

    kind, token = trigger
    # 子公司语境下的复数指代不是股票主体
    if kind == "plural" and _mentions_subsidiary(word, history):
        return CorefResult(trigger=token)

    target = _pick_history(history, kind)
    if target is None:
        return CorefResult(trigger=token)

    codes = list(target.get("stock_codes") or [])
    names = list(target.get("stock_names") or [])
    if kind == "singular" and codes:
        codes = codes[:1]
        names = names[:1]

    rewritten = None
    if names:
        rewritten = word.replace(token, "、".join(names), 1)

    return CorefResult(trigger=token, codes=codes, names=names, rewritten_query=rewritten)


def carry_forward_from_history(history: list[dict]) -> tuple[list[str], list[str], list[str]]:
    """本轮无新实体时沿用最近一条有 codes 的 user 主体与期间（04a §4.4）。

    periods 也沿用：多轮默认「还在聊同期」，无代词省略句（「有哪些风险」）靠这条补年份。
    """
    target = _pick_history(history, "plural")
    if target is None:
        return [], [], []
    return (
        list(target.get("stock_codes") or []),
        list(target.get("stock_names") or []),
        list(target.get("periods") or []),
    )


def _find_trigger(word: str) -> tuple[str, str] | None:
    # 长词优先，避免「这两家」被「这家/它」截胡
    for token in _ORDINAL:
        if token in word:
            return "ordinal", token
    for token in _PLURAL:
        if token in word:
            return "plural", token
    for token in _SINGULAR_LONG:
        if token in word:
            return "singular", token
    m = _PRONOUN_RE.search(word)
    if m:
        return "singular", m.group(0)
    return None


def _mentions_subsidiary(query: str, history: list[dict]) -> bool:
    text = query or ""
    for m in reversed(history or []):
        text += "\n" + (m.get("content") or "")
        if any(mark in text for mark in _SUBSIDIARY_MARKS):
            return True
    return False


def _pick_history(history: list[dict], kind: str) -> dict | None:
    """倒序找有 codes 的 user 条；ordinal 取倒数第 2 条。"""
    hits = [
        m
        for m in reversed(history or [])
        if m.get("role") == "user" and (m.get("stock_codes") or [])
    ]
    if kind == "ordinal":
        return hits[1] if len(hits) >= 2 else None
    return hits[0] if hits else None
