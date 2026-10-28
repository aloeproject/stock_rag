"""node_retrieve 纯规则（docs/04e §4/§5）。

filter 键映射、fan-out 展开、HyDE 三条件判定。
为什么拆纯函数：可单测到每条规则，节点只做编排（同 coref/periods 纪律）。
"""
from __future__ import annotations

# 触发条件 1 的目标章节：归因/风险题问句与答案句体裁不同，才值得 HyDE
HYDE_SECTIONS = {"management_discussion", "risk"}

# 触发条件 2 的「标准 metric_code」词面（含通俗词）：有指标锚点的短句不必 HyDE
METRIC_MARKERS = (
    "毛利率",
    "营收",
    "营业收入",
    "净利润",
    "归母",
    "净利率",
    "毛利",
    "roe",
    "roa",
)

# 触发条件 3 阈值（04 §5.12）：首轮 dense+sparse top1 太弱才补假设文档
FALLBACK_THRESHOLD = 0.35

# 条件 2 的短句阈值（04e §2）
SHORT_QUESTION_CHARS = 30

# filters 契约键 → Qdrant payload 单数键（04e §5.1）
_FILTER_KEY_MAP = {
    "stock_codes": "stock_code",
    "report_periods": "report_period",
    "report_types": "report_type",
    "sections": "section",
}


def fanout_units(stock_codes: list[str], report_periods: list[str]) -> list[tuple[str, str | None]]:
    """按 (stock_code, report_period) 展开笛卡尔积；无期时 period=None。

    为什么 fan-out 而不是一次多值 filter：多值 filter 下 top_k 会被量大的公司/期
    占满，小公司/边缘期颗粒无收（04e §3.2）。
    """
    periods = list(report_periods) if report_periods else [None]
    return [(code, period) for code in stock_codes for period in periods]


def build_payload_filter(filters: dict) -> dict[str, list[str]]:
    """filters 契约 → Qdrant payload 匹配条件；空键不出现（不加该条件）。

    键名映射在这里做，plan 不翻译（04d §7.2）；值统一列表，MatchAny 口径。
    """
    out: dict[str, list[str]] = {}
    for src, dst in _FILTER_KEY_MAP.items():
        values = [str(v) for v in (filters.get(src) or []) if v]
        if values:
            out[dst] = values
    return out


def has_metric_marker(question: str) -> bool:
    """问句是否含标准指标词（触发条件 2 的否定项）。"""
    text = (question or "").lower()
    return any(marker.lower() in text for marker in METRIC_MARKERS)


def should_use_hyde(
    route: str,
    sections: list[str],
    question: str,
    top1_score: float | None,
) -> bool:
    """HyDE 三条件任一命中才跑（04 §5.12 / 04e §3.4）。

    条件 3 依赖首轮得分，所以 HyDE 必须放在首轮检索之后。
    top1_score=None（首轮全空）按 0.0 处理——比阈值更弱，更该补路。
    """
    if route == "vector" and set(sections) & HYDE_SECTIONS:
        return True
    if len(question or "") < SHORT_QUESTION_CHARS and not has_metric_marker(question):
        return True
    score = 0.0 if top1_score is None else float(top1_score)
    return score < FALLBACK_THRESHOLD
