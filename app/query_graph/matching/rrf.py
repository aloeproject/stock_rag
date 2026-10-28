"""RRF 融合（docs/04 §5.8）。

权重 dense=1.0 / sparse=0.9 / hyde=0.8。
**全量合并→打分→截断 top_k**，勿先截断再排序（避坑）。
"""

# TODO(P3): rrf_fuse(hits_by_source: dict[str, list[dict]], top_k: int) -> list[dict]
#   score = Σ weight / (60 + rank)；返回统一证据块（带 ref/page_no/source/score）
