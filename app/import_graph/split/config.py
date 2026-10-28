from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitConfig:
    """切分参数（docs/02a §2）。

    为什么是 500 token 而不是 2000 字符：
    业界基准递归切分 400–512 token + 10–20% overlap 对检索精度最稳；
    2000 字符 ≈ 1500+ token 粒度过粗，命中后噪音大。
    """

    max_token: int = 500
    min_token: int = 150
    # 500 × 15%：只用于叙述细切；结构边界（标题/表格）处重叠只会制造重复命中
    overlap_token: int = 75
    # 合并/行组允许微超的上界（max 的 110%），防长句、大表失控
    hard_max_token: int = 550
