"""S5b 表格规则分类（docs/02d §3/§4）。

为什么双通道而不是只看标题：深振业A 的合并资产负债表 caption 是单位行
「（除特别注明外，金额单位均为人民币元）」，只认标题会漏核心报表。
为什么 sig 只在弱标题启用：有明确标题的附注不该被表内「货币资金/期末余额」误判。
为什么不全表 LLM：四报实测 905 表只有 44 张核心报表（4.9%），其余调 LLM
会产出无口径的假指标（docs/02d §1.1）。
"""
from __future__ import annotations

import re

# 标题先匹配排除词——「现金流量表补充资料」含报表字样但不是报表
TITLE_EXCLUDE = re.compile(
    r"关键审计|审计意见|非经常性损益|变动分析|情况说明|重大变动|资产构成|受限情况|补充资料"
)
# 纯单位行：深振业A 整本报表的 caption 就是这种，走 sig 通道
UNIT_ONLY = re.compile(r"^(（[^）]*）|单位\s*[:：].*|币种\s*[:：].*|[\s·—-]*)+$")

# equity 置顶：后续加规则时避免被「资产负债表」吞掉；（续）后缀天然命中
# segment 置底：「分季度主要财务」含「分」字先命中 headline；切片列头与 income
# 重叠，只走标题通道不进 sig（docs/02g §4.1）
TITLE_RULES = (
    ("equity", r"(合并|母公司)(所有者|股东)权益变动表"),
    ("balance", r"(合并|母公司)资产负债表"),
    ("income", r"(合并|母公司)利润表"),
    ("cashflow", r"(合并|母公司)现金流量表"),
    ("headline", r"主要会计数据|主要财务指标|分季度主要财务"),
    ("segment", r"主营业务分|主营业务构成|分行业情况|分产品情况|分地区情况|分销售模式"),
)

# 三条件必须同时命中：两条件会被「货币资金明细」等附注误伤（docs/02d §3.3）
# h 含「本年年末数」——深振业A 用词不是准则名「期末余额」
SIG = {
    "balance": {
        "h": r"本年年末数|上年年末数|期末余额|期初余额|期末数|年初数",
        "r": r"流动资产|货币资金",
        "extra": r"资产总计|负债合计|负债和所有者权益|所有者权益合计",
    },
    "income": {
        "h": r"本年金额|上年金额|2024\s*年度|2023\s*年度",
        "r": r"营业总收入|营业收入",
        "extra": r"利润总额|净利润|营业利润",
    },
    "cashflow": {
        "h": r"本年金额|上年金额|2024\s*年度|本期金额",
        "r": r"经营活动产生的现金流量",
        "extra": r"投资活动|筹资活动|现金及现金等价物",
    },
    "equity": {
        "h": r"本期金额|上期金额|本年金额|上年金额",
        "r": r"股本|资本公积",
        "extra": r"未分配利润|归属于母公司",
    },
    "headline": {
        "h": r"2024\s*年|2023\s*年|本报告期末",
        "r": r"营业收入|归属于上市公司股东的净利润|基本每股收益",
        "extra": r"总资产|净资产|经营活动",
    },
}

_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_CELL_RE = re.compile(r"<(td|th)[^>]*>(.*?)</\1>", re.DOTALL | re.IGNORECASE)


def parse_cells(html: str) -> list[list[str]]:
    grid: list[list[str]] = []
    for row_html in _ROW_RE.findall(html or ""):
        cells = [re.sub(r"<[^>]+>", "", inner).strip() for _t, inner in _CELL_RE.findall(row_html)]
        grid.append(cells)
    return grid


def classify_table(caption: str, html: str) -> tuple[str | None, str, str]:
    """返回 (statement_type|None, how=title|sig|skip, scope)。

    scope：title 通道从「合并/母公司」取；sig 通道留空由 LLM 回填。
    """
    cap0 = (caption or "").splitlines()[0].strip() if caption else ""

    # A 标题通道：caption 像表名才走
    if cap0 and not UNIT_ONLY.match(cap0) and not TITLE_EXCLUDE.search(cap0):
        for kind, pat in TITLE_RULES:
            if re.search(pat, cap0):
                scope = "parent" if "母公司" in cap0 else "consolidated"
                return kind, "title", scope

    # B 结构通道：仅 caption 弱（空/纯单位行）时启用
    if (not cap0) or UNIT_ONLY.match(cap0):
        head = "\n".join("|".join(row) for row in parse_cells(html)[:30])
        for kind, sig in SIG.items():
            if (
                re.search(sig["h"], head)
                and re.search(sig["r"], head)
                and re.search(sig["extra"], head)
            ):
                return kind, "sig", ""

    return None, "skip", ""
