"""共享测试夹具。

节点统一签名为 (state, runtime, writer)，runtime 是 LangGraph 注入的
DataAgentContext 单例——单测里手工构造同型对象，不 mock 节点内部逻辑。
"""
import pytest
from langgraph.runtime import Runtime

from app.import_graph.context import DataAgentContext


@pytest.fixture()
def runtime() -> Runtime[DataAgentContext]:
    return Runtime(context=DataAgentContext())
