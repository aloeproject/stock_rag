"""查询图 Runtime 共享上下文（docs/04 §10 编码规约 2）。

仿 DataAgentContext：Repository / Redis / Qdrant 从外部注入，节点内不自建。
单测手工构造同型对象，不 mock 节点内部逻辑。
"""
from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from app.clients.qdrant_client import QdrantClientManager
from app.repositories.business_fact_repository import BusinessFactRepository
from app.repositories.financial_metric_repository import FinancialMetricRepository
from app.repositories.metric_definition_repository import MetricDefinitionRepository
from app.repositories.stock_dict_repository import StockDictRepository


class LLMClient(Protocol):
    """LangChain ChatOpenAI 同型接口；单测注入 FakeLLM。"""

    def invoke(self, prompt: str) -> Any: ...


class QueryAgentContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    qdrant: Any | None = None  # QdrantClientManager，search_dense/search_sparse；单测注入 Fake
    stock_dict_repository: StockDictRepository | None = None
    financial_metric_repository: FinancialMetricRepository | None = None
    metric_definition_repository: MetricDefinitionRepository | None = None
    # 业务切片事实表（docs/02g）：未映射业务词扫 dim_name 用
    business_fact_repository: BusinessFactRepository | None = None
    llm: Any | None = None  # 自由文本：generate 流式 / HyDE 假设文档
    json_llm: Any | None = None  # json_mode：intent/plan 的 json.loads 依赖 response_format
