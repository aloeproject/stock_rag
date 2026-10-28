"""查询图装配（docs/04 §7）。

11 节点短事务图：不持久化中间态，不支持断点续跑（与导入图相反）。
fan-out 用元组并行后必须在 node_fuse 汇合。
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.core.logger import logger
from app.core.timing import timed_node
from app.query_graph.context import QueryAgentContext
from app.query_graph.nodes import (
    node_calc,
    node_clarify,
    node_entity,
    node_fuse,
    node_generate,
    node_intent,
    node_metric_query,
    node_plan,
    node_rerank,
    node_retrieve,
    node_validate,
)
from app.query_graph.routes import (
    route_after_entity,
    route_after_intent,
    route_after_metric_query,
    route_after_plan,
)
from app.query_graph.state import QueryState, default_state
from app.repositories.redis_chat_history import get_redis_chat_history, get_window_messages

NODE_INTENT = "intent"
NODE_ENTITY = "entity"
NODE_CLARIFY = "clarify"
NODE_PLAN = "plan"
NODE_RETRIEVE = "retrieve"
NODE_METRIC_QUERY = "metric_query"
NODE_CALC = "calc"
NODE_FUSE = "fuse"
NODE_RERANK = "rerank"
NODE_GENERATE = "generate"
NODE_VALIDATE = "validate"

ALL_NODES = (
    NODE_INTENT,
    NODE_ENTITY,
    NODE_CLARIFY,
    NODE_PLAN,
    NODE_RETRIEVE,
    NODE_METRIC_QUERY,
    NODE_CALC,
    NODE_FUSE,
    NODE_RERANK,
    NODE_GENERATE,
    NODE_VALIDATE,
)


def build_query_graph() -> StateGraph:
    """组装查询图：11 节点 + 4 条件边 + 两路 fan-out。

    intent ──out_of_scope──► END
           └──────────────► entity ──clarify──► END
                                  └──────────► plan ─┬─∥→ retrieve ──────────┐
                                                     └─∥→ metric_query ─calc─┤
                                                                            ▼
                                              validate ← generate ← rerank ← fuse

    并行段只有 retrieve ∥ metric_query；calc 串行在 metric_query 后（04g §5.1），
    否则拿不到 metric_rows 里的基期值。
    """
    g = StateGraph(state_schema=QueryState, context_schema=QueryAgentContext)

    g.add_node(NODE_INTENT, timed_node(NODE_INTENT, node_intent))
    g.add_node(NODE_ENTITY, timed_node(NODE_ENTITY, node_entity))
    g.add_node(NODE_CLARIFY, timed_node(NODE_CLARIFY, node_clarify))
    g.add_node(NODE_PLAN, timed_node(NODE_PLAN, node_plan))
    g.add_node(NODE_RETRIEVE, timed_node(NODE_RETRIEVE, node_retrieve))
    g.add_node(NODE_METRIC_QUERY, timed_node(NODE_METRIC_QUERY, node_metric_query))
    g.add_node(NODE_CALC, timed_node(NODE_CALC, node_calc))
    g.add_node(NODE_FUSE, timed_node(NODE_FUSE, node_fuse))
    g.add_node(NODE_RERANK, timed_node(NODE_RERANK, node_rerank))
    g.add_node(NODE_GENERATE, timed_node(NODE_GENERATE, node_generate))
    g.add_node(NODE_VALIDATE, timed_node(NODE_VALIDATE, node_validate))

    g.add_edge(START, NODE_INTENT)
    g.add_conditional_edges(
        NODE_INTENT,
        route_after_intent,
        {"entity": NODE_ENTITY, "end": END},
    )
    g.add_conditional_edges(
        NODE_ENTITY,
        route_after_entity,
        {"clarify": NODE_CLARIFY, "plan": NODE_PLAN},
    )
    g.add_edge(NODE_CLARIFY, END)
    # path_map 键 = route_after_plan 返回的元组元素，缺键运行时炸（避坑清单）
    g.add_conditional_edges(
        NODE_PLAN,
        route_after_plan,
        {
            NODE_RETRIEVE: NODE_RETRIEVE,
            NODE_METRIC_QUERY: NODE_METRIC_QUERY,
        },
    )
    # calc 串行在 metric_query 后取基期值（04g §5.1）；不满足时直接 fuse
    g.add_conditional_edges(
        NODE_METRIC_QUERY,
        route_after_metric_query,
        {"calc": NODE_CALC, "fuse": NODE_FUSE},
    )
    g.add_edge(NODE_RETRIEVE, NODE_FUSE)
    g.add_edge(NODE_CALC, NODE_FUSE)
    g.add_edge(NODE_FUSE, NODE_RERANK)
    g.add_edge(NODE_RERANK, NODE_GENERATE)
    g.add_edge(NODE_GENERATE, NODE_VALIDATE)
    g.add_edge(NODE_VALIDATE, END)

    return g


async def run_query_stream(
    query: str,
    session_id: str,
    request_id: str,
    context: QueryAgentContext | None = None,
):
    """编译查询图并流式产出 custom 事件（docs/05 §4）。

    为什么 astream 不是 ainvoke：ainvoke 拿不到 stream_writer 的自定义事件，
    而 token/clarify/done 全靠这些事件推送（04 §8）。
    为什么只收 custom 不收 values：查询不持久化中间态（04 §7），
    service 只需事件不需要终态快照。
    """
    # history 必须在起图前从 Redis 灌进 state：intent/entity 的指代消解读它，
    # 不加载则「它/该公司」永远解析不到历史主体（恒空列表 → out_of_scope 误拒）
    history: list[dict] = []
    if session_id:
        try:
            history = get_window_messages(get_redis_chat_history(session_id))
        except Exception as e:
            # Redis 不可用不阻断问答：无历史只是指代消解退化，不是致命错
            logger.warning(f"[run_query_stream] 加载会话历史失败 session_id={session_id}: {e}")
    state = default_state(
        query=query,
        request_id=request_id,
        session_id=session_id,
        history=history,
    )
    graph = build_query_graph().compile()
    # stream_mode 必须用 ["custom"] 列表形式：字符串 "custom" 产出裸 chunk 不是
    # (mode, chunk) 元组，而 astream 元组解包是按列表模式写的（05 §4）
    async for _mode, chunk in graph.astream(
        state,
        context=context if context is not None else QueryAgentContext(),
        stream_mode=["custom"],
    ):
        yield chunk
