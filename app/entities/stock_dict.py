"""stock_dict 实体（docs/04 §4.1 / 04a §4）。

名称→代码真相源。写入时机：导入图 node_metadata 解析文件名后 upsert（幂等）；
别名人工 SQL 补录；全 A 股底仓一期不做（YAGNI）。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StockDictDbData:
    stock_code: str
    stock_name: str
    full_name: str = ""
    aliases: list[str] = field(default_factory=list)
    market: str = ""  # SH | SZ | BJ
    status: int = 1  # 1=正常 0=停用
