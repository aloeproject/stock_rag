from pydantic import BaseModel, ConfigDict

from app.clients.qdrant_client import QdrantClientManager
from app.repositories.business_fact_repository import BusinessFactRepository
from app.repositories.financial_metric_repository import FinancialMetricRepository
from app.repositories.metric_definition_repository import MetricDefinitionRepository
from app.repositories.report_document_repository import ReportDocumentRepository
from app.repositories.report_table_repository import ReportTableRepository
from app.repositories.stock_dict_repository import StockDictRepository


class DataAgentContext(BaseModel):
    """LangGraph Runtime 共享上下文——仓储/连接从外部注入，节点内不散建客户端。

    参考 personal-assistant DataAgentContext：Repository 实例直接挂字段。
    qdrant 同理由 API dependencies / 测试夹具注入，节点禁止回落全局单例。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    qdrant: QdrantClientManager | None = None
    report_document_repository: ReportDocumentRepository | None = None
    report_table_repository: ReportTableRepository | None = None
    metric_definition_repository: MetricDefinitionRepository | None = None
    financial_metric_repository: FinancialMetricRepository | None = None
    # 业务切片事实表（docs/02g）：与 financial_metrics 独立生命周期
    business_fact_repository: BusinessFactRepository | None = None
    stock_dict_repository: StockDictRepository | None = None
