"""期间绝对化（docs/04a §5）。

「近三年」以最新可查报告期为基准倒推，不是自然年——1 月发年报的窗口里，
用户说的「近三年」指刚出的 2024 年报倒推，而不是 2026 倒推。
"""
from __future__ import annotations

import re

# 「近N年」的 N：阿拉伯数字或中文数字（用户两种写法都常见）
_RECENT_RE = re.compile(r"^近\s*(\d+|[一两二三四五六七八九十])\s*年$")
_YEAR_RE = re.compile(r"^(\d{4})\s*年?$")

_CN_NUM = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

_RELATIVE_ONE = {"去年", "上一年", "上年"}
_RELATIVE_TWO = {"前年"}
_RELATIVE_THIS = {"今年", "本年", "当年"}

_REPORT_TYPES = {
    "年报": "annual",
    "年度报告": "annual",
    "annual": "annual",
    "半年报": "semi_annual",
    "中期报告": "semi_annual",
    "semi_annual": "semi_annual",
    "季报": "quarterly",
    "季度报": "quarterly",
    "quarterly": "quarterly",
}


def absolutize_periods(raw: list[str], latest_period: str) -> list[str]:
    """相对表述 → 绝对年份；已是绝对年份保留。返回升序去重列表。

    无法识别的词原样保留（调用方 WARN），不中断——宁可少过滤不过度猜测。
    """
    base = int(latest_period) if str(latest_period).isdigit() else None
    out: set[str] = set()

    for item in raw or []:
        word = (item or "").strip()
        if not word:
            continue

        m = _YEAR_RE.match(word)
        if m:
            out.add(m.group(1))
            continue

        m = _RECENT_RE.match(word)
        if m and base is not None:
            token = m.group(1)
            n = int(token) if token.isdigit() else _CN_NUM.get(token, 1)
            n = max(n, 1)
            for y in range(base - n + 1, base + 1):
                out.add(str(y))
            continue

        if word in _RELATIVE_ONE and base is not None:
            out.add(str(base - 1))
            continue
        if word in _RELATIVE_TWO and base is not None:
            out.add(str(base - 2))
            continue
        if word in _RELATIVE_THIS and base is not None:
            out.add(str(base))
            continue

        out.add(word)

    return sorted(out)


def normalize_report_types(raw: list[str]) -> list[str]:
    """中文报告类型 → 枚举；识别不了原样保留（04a §5.2）。"""
    out: list[str] = []
    for item in raw or []:
        word = (item or "").strip()
        if not word:
            continue
        out.append(_REPORT_TYPES.get(word, word))
    return out
