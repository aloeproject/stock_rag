"""run_ingest 事件桥接测试：graph custom 事件 → ProgressHub。

用最小 StateGraph 验证桥接行为，不跑真实导入节点（MinerU/MySQL/Qdrant）。
"""
import asyncio

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import StreamWriter

from app.import_graph.context import DataAgentContext
from app.import_graph.main_graph import bridge_stream_to_hub
from app.import_graph.progress_hub import ProgressHub


def _writer_node(state: dict, runtime: Runtime[DataAgentContext], writer: StreamWriter) -> dict:
    writer({"type": "progress", "step": "测试步骤", "status": "running"})
    writer({"type": "progress", "step": "测试步骤", "status": "done"})
    return {"result": "ok"}


def _build_tiny_graph():
    g = StateGraph(state_schema=dict, context_schema=DataAgentContext)
    g.add_node("writer_node", _writer_node)
    g.add_edge(START, "writer_node")
    g.add_edge("writer_node", END)
    return g.compile()


def test_bridge_publishes_custom_events_to_hub():
    async def scenario():
        hub = ProgressHub()
        graph = _build_tiny_graph()
        ctx = DataAgentContext()
        final = await bridge_stream_to_hub(
            graph=graph,
            state={},
            context=ctx,
            document_id="doc1",
            hub=hub,
        )
        return hub.history("doc1"), final

    history, final = asyncio.run(scenario())
    assert history == [
        {"type": "progress", "step": "测试步骤", "status": "running"},
        {"type": "progress", "step": "测试步骤", "status": "done"},
    ]
    assert final == {"result": "ok"}


def test_bridge_without_hub_still_returns_state():
    async def scenario():
        graph = _build_tiny_graph()
        return await bridge_stream_to_hub(
            graph=graph,
            state={},
            context=DataAgentContext(),
            document_id="doc1",
            hub=None,
        )

    assert asyncio.run(scenario()) == {"result": "ok"}


def test_bridge_passes_context_into_nodes():
    """节点 runtime.context 必须是调用方传入的那个（DI 装配的仓储要能到节点）。"""

    seen: list[object] = []

    def probe_node(state: dict, runtime: Runtime[DataAgentContext], writer: StreamWriter) -> dict:
        seen.append(runtime.context)
        return {}

    g = StateGraph(state_schema=dict, context_schema=DataAgentContext)
    g.add_node("probe", probe_node)
    g.add_edge(START, "probe")
    g.add_edge("probe", END)

    ctx = DataAgentContext()
    asyncio.run(
        bridge_stream_to_hub(
            graph=g.compile(),
            state={},
            context=ctx,
            document_id="doc1",
            hub=ProgressHub(),
        )
    )
    assert seen and seen[0] is ctx
