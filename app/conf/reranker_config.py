"""bge-reranker-large 本地模型配置（docs/04i §2）。

对齐 dataset_rag 的 env 名；默认 cpu——4 核 Intel Mac 无 GPU，fp16 无加速收益。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class RerankerConfig:
    bge_reranker_large: str
    bge_reranker_device: str
    bge_reranker_fp16: bool


reranker_config = RerankerConfig(
    bge_reranker_large=os.getenv("BGE_RERANKER_LARGE", "models/BAAI/bge-reranker-large"),
    bge_reranker_device=os.getenv("BGE_RERANKER_DEVICE", "cpu"),
    bge_reranker_fp16=os.getenv("BGE_RERANKER_FP16") in ("1", "True", "true"),
)
