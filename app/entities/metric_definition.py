from dataclasses import dataclass
from enum import IntEnum, StrEnum

from app.entities.report_table import StatementType


class MetricLevel(IntEnum):
    L1 = 1
    L2 = 2
    L3 = 3


class MetricCategory(StrEnum):
    HEADLINE = "headline"
    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"
    INCOME = "income"
    COST = "cost"
    CASHFLOW = "cashflow"
    PROFITABILITY = "profitability"
    SOLVENCY = "solvency"
    EFFICIENCY = "efficiency"
    GROWTH = "growth"


@dataclass
class MetricDefinitionDbData:
    metric_code: str
    metric_name: str
    aliases: str | None
    category: MetricCategory | None
    statement_type: StatementType | None
    metric_level: MetricLevel
    calc_formula: str | None
    unit_default: str | None
    description: str | None
