"""S6 口径映射 + L2 物化（docs/02e）。

为什么不能把 items 直接入 financial_metrics：同义多名、同指标多表、
未入字典、period 不规范、权益变动表同名多行（docs/02e §1）。
本节点是标准化层，不做二次 LLM。
"""
from __future__ import annotations

import ast
import operator
import re
from decimal import Decimal, InvalidOperation

from langgraph.runtime import Runtime
from langgraph.types import StreamWriter

from app.core.logger import logger
from app.entities.business_fact import BusinessFactDbData, DimType
from app.entities.financial_metric import (
    ExtractSource,
    FinancialMetricDbData,
    MetricPeriodType,
)
from app.entities.metric_definition import MetricDefinitionDbData, MetricLevel
from app.entities.report_document import ReportType
from app.entities.report_table import StatementScope, StatementType
from app.import_graph.context import DataAgentContext
from app.import_graph.nodes.progress import emit_progress, progress_step
from app.import_graph.state import ImportState

# P0 进 L1 的报表类型；equity 不进（同名多行：期初/增减/期末，uk 装不下，docs/02e §3）
_L1_STATEMENT_TYPES = {
    StatementType.HEADLINE,
    StatementType.INCOME,
    StatementType.BALANCE,
    StatementType.CASHFLOW,
}

_PERIOD_RE = re.compile(r"^\d{4}(?:Q[1-4])?$")

# 主表标题（合并/母公司 × 四表 + 主要会计数据/财务指标）；弱 caption 附注表不在此列
_CORE_TITLE_RE = re.compile(
    r"(合并|母公司)?.*(资产负债表|利润表|现金流量表|所有者权益变动表)|主要会计数据|主要财务指标"
)

# 万元/亿元 → 元；% / 元每股 / 倍原样（docs/02e §5.3）
_UNIT_FACTOR = {
    "元": Decimal("1"),
    "万元": Decimal("10000"),
    "亿元": Decimal("100000000"),
}
_UNIT_KEEP = {
    "%": "%",
    "％": "%",
    "元/股": "元每股",
    "元每股": "元每股",
    "倍": "倍",
}

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def match_code(name: str, defs: list[MetricDefinitionDbData]) -> tuple[str | None, str]:
    """名称 → (metric_code, how∈{exact,contains,unmapped})。

    为什么精确优先：宽松包含会把「非流动资产合计」串进「流动资产合计」、
    「购建固定资产…支付的现金」串进「固定资产」、「扣非基本每股收益」串进
    「基本每股收益」（docs/02e §2.2 实测）。P0 只做精确匹配，contains 通道
    等字典收紧（最长别名≥4字 + 唯一命中）后再开。
    """
    for d in defs:
        if d.metric_name == name:
            return d.metric_code, "exact"
    for d in defs:
        aliases = [a for a in (d.aliases or "").split("|") if a]
        if name in aliases:
            return d.metric_code, "exact"
    return None, "unmapped"


def normalize_metric_period(period: str, report_period: str) -> str | None:
    """metric_period 规范。

    current/prior 来源：弱 caption 表只吐相对期（新世界 id=179，docs/02e §5.1）。
    """
    if period == "current":
        return report_period
    if period == "prior":
        if _PERIOD_RE.match(report_period):
            year = int(report_period[:4])
            return f"{year - 1}{report_period[4:]}"
        return None
    if _PERIOD_RE.match(period):
        return period
    return None


def _parse_decimal(value_raw: str | None) -> Decimal | None:
    s = (value_raw or "").strip().replace(",", "")
    if not s or s in {"—", "-", "–"}:
        return None
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return Decimal(s)
    except (InvalidOperation, ValueError):
        return None


def _normalize_unit_value(
    value_raw: str | None, unit: str | None, unit_default: str | None
) -> tuple[Decimal, str] | None:
    s = (value_raw or "").strip()
    # LLM 常把 % 写进 value_raw（加权ROE='1.86%'）；Decimal 解析不了，整条被丢。
    # 值自带百分号时以值为准，unit 写错/缺失也不能落成「元」
    if s.endswith("%") or s.endswith("％"):
        raw = _parse_decimal(s[:-1])
        return None if raw is None else (raw, "%")
    raw = _parse_decimal(s)
    if raw is None:
        return None
    u = (unit or "").strip() or (unit_default or "").strip()
    if u in _UNIT_KEEP:
        return raw, _UNIT_KEEP[u]
    factor = _UNIT_FACTOR.get(u)
    if factor is not None:
        return raw * factor, "元"
    # 空/未知：按 unit_default 原样，不做换算（docs/02e §5.3）
    return raw, (unit_default or u or "元")


def _eval_formula(formula: str, values: dict[str, Decimal]) -> Decimal | None:
    """受限表达式求值；缺依赖/除零/未知节点 → None（跳过该 L2）。"""
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return None

    def _walk(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_walk(node.left), _walk(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_walk(node.operand)
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise KeyError(node.id)
            return values[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Decimal(str(node.value))
        raise ValueError(type(node).__name__)

    try:
        return _walk(tree)
    except (KeyError, ValueError, ZeroDivisionError, InvalidOperation):
        return None


def _resolve_value(item: dict, report_period: str, unit_default: str | None):
    period = normalize_metric_period(str(item.get("metric_period") or ""), report_period)
    if period is None:
        logger.warning(
            f"[metric_normalize] period 非法跳过: {item.get('metric_period')!r} name={item.get('metric_name')!r}"
        )
        return None
    norm = _normalize_unit_value(item.get("value_raw"), item.get("unit"), unit_default)
    if norm is None:
        logger.warning(
            f"[metric_normalize] value_raw 不可转数值跳过: {item.get('value_raw')!r} name={item.get('metric_name')!r}"
        )
        return None
    value, unit = norm
    try:
        ptype = MetricPeriodType(item.get("metric_period_type") or "current")
    except ValueError:
        ptype = MetricPeriodType.CURRENT
    return period, ptype, value, unit


# 单位后缀：「毛利率(%)」去掉后才能对上字典「毛利率」；去的是单位不是业务词（docs/02g §6.2）
_UNIT_SUFFIX_RE = re.compile(r"[（(]\s*(%|％|元|万元|亿元|元/股|元每股|倍)\s*[)）]\s*$")


def _strip_unit_suffix(name: str) -> str:
    return _UNIT_SUFFIX_RE.sub("", name or "").strip()


def _normalize_dim_type(raw: str | None) -> DimType:
    """dim_type 枚举归一；认不出归 other 不丢行（docs/02g §7.3/§9）。"""
    s = (raw or "").strip().lower()
    try:
        return DimType(s)
    except ValueError:
        return DimType.OTHER


def _map_segment_metric_code(
    name: str, defs: list[MetricDefinitionDbData]
) -> str | None:
    """切片列头 → metric_code，可失败（docs/02g §6.2）。

    为什么「比上年增减(%)」不映射 yoy 码：那是披露值不是查询侧算的 L3，
    撞 uk 会分不清该信哪个（docs/02g §6.2）。宁可 NULL，靠 metric_name 查。
    """
    stripped = _strip_unit_suffix(name)
    if "比上年增减" in stripped or "同比增减" in stripped:
        return None
    code, _how = match_code(stripped, defs)
    if code is None and stripped != name:
        code, _how = match_code(name, defs)
    return code


def _resolve_segment_item(
    item: dict, defs: list[MetricDefinitionDbData], report_period: str
) -> dict | None:
    """segment item → business_facts 行草稿；缺 dim 直接丢（docs/02g §6.1）。

    为什么不猜维度：dim 是开放维度的定位键，猜错比丢弃更糟。
    """
    dim_type_raw = item.get("dim_type")
    dim_name = (item.get("dim_name") or "").strip()
    if not dim_type_raw or not dim_name:
        logger.warning(
            f"[metric_normalize] segment item 缺 dim 丢弃: {item.get('metric_name')!r}"
        )
        return None
    dim_type = _normalize_dim_type(dim_type_raw)
    if dim_type == DimType.OTHER and str(dim_type_raw).strip().lower() != DimType.OTHER:
        logger.warning(
            f"[metric_normalize] dim_type 不在枚举归 other: {dim_type_raw!r} dim_name={dim_name!r}"
        )
    name = (item.get("metric_name") or "").strip()
    if not name:
        return None
    period = normalize_metric_period(str(item.get("metric_period") or ""), report_period)
    if period is None:
        logger.warning(
            f"[metric_normalize] segment period 非法跳过: {item.get('metric_period')!r} name={name!r}"
        )
        return None
    norm = _normalize_unit_value(item.get("value_raw"), item.get("unit"), None)
    if norm is None:
        logger.warning(
            f"[metric_normalize] segment value_raw 不可转数值跳过: "
            f"{item.get('value_raw')!r} name={name!r}"
        )
        return None
    value, unit = norm
    return {
        "dim_type": dim_type,
        "dim_name": dim_name,
        "metric_code": _map_segment_metric_code(name, defs),
        "metric_name": name,
        "metric_period": period,
        "metric_value": value,
        "unit": unit,
        "value_raw": str(item.get("value_raw") or ""),
        "extract_source": ExtractSource.TABLE,
    }


async def node_metric_normalize(
    state: ImportState,
    runtime: Runtime[DataAgentContext],
    writer: StreamWriter | None = None,
) -> dict:
    """S6: 口径映射、单位换算、L2 物化写 MySQL。

    事务边界：无 LLM；L1/L2 各一短事务。重跑先 delete_by_document，
    否则指标集合变小会留残行（docs/02e §8.3）。
    仓储由 DataAgentContext 注入，节点内不自建 Repository。
    """
    logger.info(f"[node_metric_normalize] 开始处理文档: {state.get('document_id')}")
    emit_progress(writer, "指标归一", "running")
    document_id = state.get("document_id") or ""
    stock_code = state.get("stock_code") or ""
    report_period = state.get("report_period") or ""
    report_type = ReportType(state.get("report_type") or "annual")

    table_repo = runtime.context.report_table_repository
    def_repo = runtime.context.metric_definition_repository
    metric_repo = runtime.context.financial_metric_repository
    fact_repo = runtime.context.business_fact_repository
    if table_repo is None or def_repo is None or metric_repo is None:
        raise ValueError("[metric_normalize] DataAgentContext 仓储未注入")

    # 字典必须先种：不种则 l1_defs 空，所有指标名「未映射」→ financial_metrics 零行。
    # upsert 幂等，每次调用无副作用（02e §4.2）
    _seed_definitions(def_repo)
    def_repo.session.commit()

    defs = def_repo.list_all()
    l1_defs = [d for d in defs if d.metric_level == MetricLevel.L1]
    l2_defs = [d for d in defs if d.metric_level == MetricLevel.L2 and d.calc_formula]
    def_by_code = {d.metric_code: d for d in defs}

    unmapped_names: list[str] = []
    # key → candidates[]；uk 裁决前内存聚合（docs/02e §6）
    buckets: dict[tuple, list[dict]] = {}
    # segment items → business_facts（docs/02g §6.1），不进 L1 buckets
    fact_rows: list[dict] = []

    tables = table_repo.list_ok_by_document(document_id)
    for table in tables:
        # 业务切片改道 business_facts：开放维度不该掺进公司级 uk 与 L2 公式（docs/02g §1.1）
        if table.statement_type == StatementType.SEGMENT:
            if fact_repo is None:
                logger.error(
                    f"[metric_normalize] segment 表缺 business_fact_repository，丢弃 table_id={table.id}"
                )
                continue
            for item in table.extracted_metrics or []:
                # 全字典映射：切片列头可命中 L2（「毛利率(%)」→ gross_margin，docs/02g §6.2）
                row = _resolve_segment_item(item, defs, report_period)
                if row is not None:
                    row["page_no"] = table.page_no
                    fact_rows.append(row)
            continue
        # 为什么变动表不进 L1：同名多行（期初/增减/期末），期末余额已在 balance（docs/02e §3）
        if table.statement_type not in _L1_STATEMENT_TYPES:
            continue
        scope = table.statement_scope or StatementScope.CONSOLIDATED
        if scope == StatementScope.EMPTY:
            scope = StatementScope.CONSOLIDATED
        # 为什么专用表 > headline 的 extract_source：裁决用（docs/02e §6）
        extract_source = (
            ExtractSource.HEADLINE
            if table.statement_type == StatementType.HEADLINE
            else ExtractSource.TABLE
        )
        for item in table.extracted_metrics or []:
            name = (item.get("metric_name") or "").strip()
            if not name:
                continue
            code, how = match_code(name, l1_defs)
            if code is None:
                if name not in unmapped_names:
                    unmapped_names.append(name)
                continue
            resolved = _resolve_value(item, report_period, def_by_code[code].unit_default)
            if resolved is None:
                continue
            period, ptype, value, unit = resolved
            key = (code, scope, period, ptype)
            buckets.setdefault(key, []).append(
                {
                    "metric_code": code,
                    "metric_name": name,
                    "metric_value": value,
                    "unit": unit,
                    "value_raw": str(item.get("value_raw") or ""),
                    "is_restated": 1 if item.get("is_restated") else 0,
                    "statement_type": table.statement_type,
                    "statement_scope": scope,
                    "metric_period": period,
                    "metric_period_type": ptype,
                    "page_no": table.page_no,
                    "extract_source": extract_source,
                    "table_id": table.id or 0,
                    "table_title": table.table_title or "",
                    # 标准名 > 别名（「总资产」压过「资产合计/资产总计」）
                    "official_name": int(name == def_by_code[code].metric_name),
                }
            )

    for name in unmapped_names:
        logger.warning(f"[metric_normalize] 名称未映射: {name}")

    # uk 裁决（docs/02e §2.3 值一致性，§6 优先级）：
    # 1) 专用报表 > headline  2) 主表标题 > 弱 caption 附注切片
    # 3) 标准名 > 别名  4) table_id 大者
    # 为什么 2 必须在 table_id 之前：附注里的联营企业表（如新世界 id=179
    # 「单位：元 币种：人民币」）也有「资产合计/负债合计/净利润」，table_id
    # 更大但值是投资主体的，会把合并报表真值顶掉（02e §6 已回写）。
    # 为什么 0 值先剔除：LLM 对空单元格/占位吐 0.00，排序上「专用表 > headline」
    # 会让坏值压过真值（真实日志 revenue 2024 win=0.00 lose=11.2亿）。有非 0
    # 候选时 0 不参与裁决；全 0 才保留，报表里的真 0 仍可入账。
    winners: list[dict] = []
    for key, cands in buckets.items():
        nonzero = [c for c in cands if c["metric_value"] != 0]
        if nonzero:
            cands = nonzero
        cands_sorted = sorted(
            cands,
            key=lambda c: (
                0 if c["extract_source"] == ExtractSource.HEADLINE else 1,
                1 if _CORE_TITLE_RE.search(c["table_title"]) else 0,
                c["official_name"],
                c["table_id"],
            ),
            reverse=True,
        )
        win = cands_sorted[0]
        winners.append(win)
        for other in cands_sorted[1:]:
            a, b = win["metric_value"], other["metric_value"]
            denom = max(abs(a), abs(b))
            if denom > 0 and abs(a - b) / denom > Decimal("0.001"):
                logger.warning(
                    f"[metric_normalize] uk 值冲突 key={key} "
                    f"win={win['value_raw']}({win['extract_source']}) "
                    f"lose={other['value_raw']}({other['extract_source']})"
                )

    session = metric_repo.session
    metrics_rows = 0
    try:
        metric_repo.delete_by_document(document_id)
        for w in winners:
            metric_repo.upsert(
                FinancialMetricDbData(
                    document_id=document_id,
                    stock_code=stock_code,
                    report_period=report_period,
                    report_type=report_type,
                    metric_period=w["metric_period"],
                    metric_period_type=w["metric_period_type"],
                    statement_type=w["statement_type"],
                    statement_scope=w["statement_scope"],
                    metric_code=w["metric_code"],
                    metric_name=w["metric_name"],
                    metric_value=w["metric_value"],
                    unit=w["unit"],
                    value_raw=w["value_raw"],
                    is_restated=w["is_restated"],
                    page_no=w["page_no"],
                    extract_source=w["extract_source"],
                )
            )
            metrics_rows += 1
        session.commit()
    except Exception as e:
        logger.error(f"[metric_normalize] L1 写库失败: {e}", exc_info=True)
        session.rollback()
        emit_progress(writer, "指标归一", "failed")
        return {
            "metrics_rows": 0,
            "business_facts_rows": 0,
            "unmapped_names": unmapped_names,
        }

    # business_facts 独立短事务：与 financial_metrics 生命周期分离（docs/02g §6.3）
    fact_rows_written = 0
    if fact_repo is not None:
        try:
            fact_repo.delete_by_document(document_id)
            for w in fact_rows:
                fact_repo.upsert(
                    BusinessFactDbData(
                        document_id=document_id,
                        stock_code=stock_code,
                        report_period=report_period,
                        report_type=report_type,
                        dim_type=w["dim_type"],
                        dim_name=w["dim_name"],
                        metric_code=w["metric_code"],
                        metric_name=w["metric_name"],
                        metric_period=w["metric_period"],
                        metric_value=w["metric_value"],
                        unit=w["unit"],
                        value_raw=w["value_raw"],
                        page_no=w["page_no"],
                        extract_source=w["extract_source"],
                    )
                )
                fact_rows_written += 1
            fact_repo.session.commit()
        except Exception as e:
            logger.error(f"[metric_normalize] business_facts 写库失败: {e}", exc_info=True)
            fact_repo.session.rollback()
            emit_progress(writer, "指标归一", "failed")
            return {
                "metrics_rows": metrics_rows,
                "unmapped_names": unmapped_names,
                "business_facts_rows": 0,
            }

    # L2 只算 L1、只算 current；缺依赖跳过，禁止用原名凑数（docs/02e §7）
    l1_values: dict[tuple, dict[str, Decimal]] = {}
    for w in winners:
        if w["metric_period_type"] != MetricPeriodType.CURRENT:
            continue
        group = (w["statement_scope"], w["metric_period"])
        l1_values.setdefault(group, {})[w["metric_code"]] = w["metric_value"]

    l2_rows = 0
    try:
        for (scope, period), values in l1_values.items():
            for d in l2_defs:
                computed = _eval_formula(d.calc_formula or "", values)
                if computed is None:
                    logger.warning(
                        f"[metric_normalize] L2 缺依赖跳过: {d.metric_code} "
                        f"period={period} scope={scope} formula={d.calc_formula}"
                    )
                    continue
                metric_repo.upsert(
                    FinancialMetricDbData(
                        document_id=document_id,
                        stock_code=stock_code,
                        report_period=report_period,
                        report_type=report_type,
                        metric_period=period,
                        metric_period_type=MetricPeriodType.CURRENT,
                        statement_type=d.statement_type or StatementType.RATIO,
                        statement_scope=scope,
                        metric_code=d.metric_code,
                        metric_name=d.metric_name,
                        metric_value=computed,
                        unit=d.unit_default,
                        value_raw=str(computed),
                        is_restated=0,
                        page_no=None,
                        extract_source=ExtractSource.DERIVED,
                    )
                )
                l2_rows += 1
        session.commit()
    except Exception as e:
        logger.error(f"[metric_normalize] L2 写库失败: {e}", exc_info=True)
        session.rollback()
        emit_progress(writer, "指标归一", "failed")
        return {
            "metrics_rows": metrics_rows,
            "business_facts_rows": fact_rows_written,
            "unmapped_names": unmapped_names,
        }

    metrics_rows += l2_rows
    logger.info(
        f"[metric_normalize] 完成 l1={metrics_rows - l2_rows} l2={l2_rows} "
        f"facts={fact_rows_written} unmapped={len(unmapped_names)} tables={len(tables)}"
    )
    emit_progress(writer, "指标归一", "done")
    return {
        "metrics_rows": metrics_rows,
        "business_facts_rows": fact_rows_written,
        "unmapped_names": unmapped_names,
    }


def _seed_definitions(repo) -> int:
    """首批字典 + docs/02e §4.2 真实数据别名补丁。

    cost_of_revenue 不含「营业总成本」：营业总成本含费用，会污染毛利率。
    """
    from app.entities.metric_definition import MetricCategory

    def _d(code, name, aliases, cat, stmt, level, formula=None, unit="元"):
        return MetricDefinitionDbData(
            metric_code=code,
            metric_name=name,
            aliases=aliases,
            category=MetricCategory(cat),
            statement_type=stmt,
            metric_level=level,
            calc_formula=formula,
            unit_default=unit,
            description=None,
        )

    L1, L2 = MetricLevel.L1, MetricLevel.L2
    rows = [
        _d("revenue", "营业收入", "营业总收入", "headline", StatementType.HEADLINE, L1),
        _d("net_profit_attributable", "归属于上市公司股东的净利润", "归属于母公司股东的净利润|归母净利润", "headline", StatementType.HEADLINE, L1),
        # 「扣非净利润」是口语高频写法，缺它会被包含匹配截胡到 net_profit（好当家×陕西煤业 compare 实测）
        _d("net_profit_attributable_deducted", "归属于上市公司股东的扣除非经常性损益的净利润", "扣非归母净利润|扣非净利润", "headline", StatementType.HEADLINE, L1),
        _d("net_operating_cashflow", "经营活动产生的现金流量净额", "经营活动现金流量净额", "cashflow", StatementType.CASHFLOW, L1),
        _d("eps_basic", "基本每股收益", None, "headline", StatementType.HEADLINE, L1, unit="元每股"),
        _d("eps_diluted", "稀释每股收益", None, "headline", StatementType.HEADLINE, L1, unit="元每股"),
        _d("roe_weighted", "加权平均净资产收益率", "加权平均ROE", "profitability", StatementType.RATIO, L1, unit="%"),
        # 02e §4.2：补「资产合计」表内写法
        _d("total_assets", "总资产", "资产总计|资产合计", "asset", StatementType.BALANCE, L1),
        _d(
            "net_assets_attributable",
            "归属于上市公司股东的净资产",
            "归属于母公司股东的净资产|归母净资产|归属于母公司所有者权益(或股东权益)合计|归属于母公司所有者权益小计",
            "equity",
            StatementType.BALANCE,
            L1,
        ),
        # 02e §4.2：摘掉「营业总成本」别名（含费用，不能当营业成本）
        _d("cost_of_revenue", "营业成本", None, "cost", StatementType.INCOME, L1),
        _d("taxes_surcharges", "税金及附加", None, "cost", StatementType.INCOME, L1),
        _d("selling_expenses", "销售费用", None, "cost", StatementType.INCOME, L1),
        _d("admin_expenses", "管理费用", None, "cost", StatementType.INCOME, L1),
        _d("rd_expenses", "研发费用", None, "cost", StatementType.INCOME, L1),
        _d("finance_expenses", "财务费用", None, "cost", StatementType.INCOME, L1),
        _d("operating_profit", "营业利润", None, "income", StatementType.INCOME, L1),
        _d("total_profit", "利润总额", None, "income", StatementType.INCOME, L1),
        _d("net_profit", "净利润", None, "income", StatementType.INCOME, L1),
        _d("minority_profit", "少数股东损益", None, "income", StatementType.INCOME, L1),
        _d("cash_and_equivalents", "货币资金", None, "asset", StatementType.BALANCE, L1),
        _d("accounts_receivable", "应收账款", None, "asset", StatementType.BALANCE, L1),
        _d("inventory", "存货", None, "asset", StatementType.BALANCE, L1),
        _d("fixed_assets", "固定资产", None, "asset", StatementType.BALANCE, L1),
        _d("short_term_borrowings", "短期借款", None, "liability", StatementType.BALANCE, L1),
        _d("accounts_payable", "应付账款", None, "liability", StatementType.BALANCE, L1),
        _d("current_assets_total", "流动资产合计", None, "asset", StatementType.BALANCE, L1),
        _d("current_liabilities_total", "流动负债合计", None, "liability", StatementType.BALANCE, L1),
        _d("total_liabilities", "负债合计", None, "liability", StatementType.BALANCE, L1),
        # 02e §4.2：补「实收资本(或股本)」
        _d("share_capital", "股本", "实收资本(或股本)", "equity", StatementType.BALANCE, L1),
        _d("undistributed_profit", "未分配利润", None, "equity", StatementType.BALANCE, L1),
        # 02e §4.2：补括号变体
        _d("equity_total", "所有者权益合计", "所有者权益(或股东权益)合计", "equity", StatementType.BALANCE, L1),
        _d("gross_margin", "毛利率", None, "profitability", StatementType.RATIO, L2, "(revenue-cost_of_revenue)/revenue*100", "%"),
        _d("net_margin", "净利率", None, "profitability", StatementType.RATIO, L2, "net_profit_attributable/revenue*100", "%"),
        _d("debt_to_asset_ratio", "资产负债率", None, "solvency", StatementType.RATIO, L2, "total_liabilities/total_assets*100", "%"),
        _d("current_ratio", "流动比率", None, "solvency", StatementType.RATIO, L2, "current_assets_total/current_liabilities_total", "倍"),
        _d("ocf_to_net_profit", "经营现金流/净利润", None, "cashflow", StatementType.RATIO, L2, "net_operating_cashflow/net_profit_attributable*100", "%"),
        # avg_* 依赖 t 与 t-1 期末，本节点不物化；缺依赖会跳过（docs/02e §7.2）
        _d("roe", "净资产收益率", None, "profitability", StatementType.RATIO, L2, "net_profit_attributable/avg_equity_attributable*100", "%"),
        _d("roa", "总资产收益率", None, "profitability", StatementType.RATIO, L2, "net_profit/avg_total_assets*100", "%"),
    ]
    for d in rows:
        repo.upsert(d)
    return len(rows)


if __name__ == "__main__":
    # 真实全量：600628 新世界 report_tables(ok) → S6 → financial_metrics，写真实 MySQL。
    # 验收对照 docs/02e §11：revenue 2024=1120106782.08 唯一；gross_margin≈41.73%。
    import asyncio
    from collections import Counter

    from langgraph.runtime import Runtime

    from app.clients.mysql_client import MySQLClientManager
    from app.models.base import Base
    from app.repositories.financial_metric_repository import FinancialMetricRepository
    from app.repositories.metric_definition_repository import MetricDefinitionRepository
    from app.repositories.report_table_repository import ReportTableRepository

    DOC_ID = "4ffe5afef59df4342e46bfd86b41b62992f642c7"  # 600628 新世界 PDF sha1

    mgr = MySQLClientManager()
    mgr.init()
    Base.metadata.create_all(mgr.engine)
    session = mgr.session()

    def_repo = MetricDefinitionRepository(session)
    table_repo = ReportTableRepository(session)
    metric_repo = FinancialMetricRepository(session)
    n_defs = _seed_definitions(def_repo)
    session.commit()
    print(f"== 字典种子 {n_defs} 条（含 02e §4.2 别名补丁）==")

    ok_tables = table_repo.list_ok_by_document(DOC_ID)
    by_type = Counter((t.statement_type.value if t.statement_type else "?") for t in ok_tables)
    n_items = sum(len(t.extracted_metrics or []) for t in ok_tables)
    print(f"== 输入 report_tables ok={len(ok_tables)} {dict(by_type)} items={n_items} ==")
    assert ok_tables, "report_tables 无 ok 表，先跑 table_extract"

    state: ImportState = {
        "document_id": DOC_ID,
        "stock_code": "600628",
        "stock_name": "新世界",
        "report_type": "annual",
        "report_period": "2024",
        "metrics_rows": 0,
        "unmapped_names": [],
    }  # type: ignore[typeddict-item]

    ctx = DataAgentContext(
        report_table_repository=table_repo,
        metric_definition_repository=def_repo,
        financial_metric_repository=metric_repo,
    )
    patch = asyncio.run(node_metric_normalize(state, Runtime(context=ctx)))  # type: ignore[arg-type]
    print(f"== state patch == {patch}")
    assert patch["metrics_rows"] > 0

    rows = metric_repo.list_by_document(DOC_ID)
    by_code = {}
    for r in rows:
        by_code.setdefault(r.metric_code, []).append(r)

    print(f"== financial_metrics {len(rows)} 行 / {len(by_code)} 个 code ==")
    print(f"== unmapped {len(patch['unmapped_names'])} == {patch['unmapped_names'][:15]}")

    # 02e §11 验收
    print("== 02e §11 验收 ==")
    rev = by_code.get("revenue") or []
    rev_2024 = [r for r in rev if r.metric_period == "2024"]
    print(f"  revenue 2024 rows={len(rev_2024)} values={[str(r.metric_value) for r in rev_2024]}")
    assert len(rev_2024) == 1, "revenue consolidated 2024 必须唯一"
    assert abs(float(rev_2024[0].metric_value) - 1120106782.08) < 0.01, rev_2024[0].metric_value
    print("  ✓ revenue 2024 = 1120106782.08 唯一")

    gm = by_code.get("gross_margin") or []
    gm_2024 = [r for r in gm if r.metric_period == "2024"]
    if gm_2024:
        v = float(gm_2024[0].metric_value)
        print(f"  gross_margin 2024 = {v:.4f}% source={gm_2024[0].extract_source.value}")
        assert abs(v - 41.73) < 0.1, v
        print("  ✓ gross_margin ≈ 41.73%")
    else:
        print("  ✗ gross_margin 未产出（缺 cost_of_revenue 映射？）")

    for code in ("net_margin", "debt_to_asset_ratio", "current_ratio", "ocf_to_net_profit"):
        hit = [r for r in (by_code.get(code) or []) if r.metric_period == "2024"]
        if hit:
            print(f"  {code} 2024 = {float(hit[0].metric_value):.4f}{hit[0].unit or ''}")
        else:
            print(f"  {code} 未产出（缺依赖，02e §7 允许跳过）")

    # period current/prior（新世界 id=179）
    periods = Counter(r.metric_period for r in rows)
    print(f"  metric_period 分布: {dict(periods)}")

    # 幂等：重跑行数不变
    patch2 = asyncio.run(node_metric_normalize(state, Runtime(context=ctx)))  # type: ignore[arg-type]
    rows2 = metric_repo.list_by_document(DOC_ID)
    print(f"== 幂等重跑 == rows {len(rows)} -> {len(rows2)} patch2={patch2}")
    assert len(rows) == len(rows2), "重跑行数必须不变"

    by_src = Counter(r.extract_source.value for r in rows2)
    print(f"== extract_source 分布: {dict(by_src)} ==")
    print(f"== SQL 复核: SELECT metric_code,metric_period,statement_scope,metric_value,unit,extract_source FROM financial_metrics WHERE document_id='{DOC_ID}' ORDER BY metric_code,metric_period; ==")
    session.close()
    print("== 完成 ==")
