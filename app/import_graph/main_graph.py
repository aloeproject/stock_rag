from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.core.timing import timed_node
from app.import_graph.context import DataAgentContext
from app.import_graph.nodes import (
    node_commit_status,
    node_embed_store,
    node_metadata,
    node_metric_normalize,
    node_parse,
    node_sectionize,
    node_split,
    node_table_extract,
)
from app.import_graph.routes import (
    route_after_metadata,
    route_after_parse,
    route_after_sectionize,
)
from app.import_graph.progress_hub import ProgressHub
from app.import_graph.state import ImportState, initial_state
from app.utils.file import sha1_file
from app.utils.path_util import PROJECT_ROOT

NODE_PARSE = "node_parse"
NODE_METADATA = "node_metadata"
NODE_SPLIT = "node_split"
NODE_SECTIONIZE = "node_sectionize"
NODE_EMBED_STORE = "node_embed_store"
NODE_TABLE_EXTRACT = "node_table_extract"
NODE_METRIC_NORMALIZE = "node_metric_normalize"
NODE_COMMIT_STATUS = "node_commit_status"

ALL_NODES = (
    NODE_PARSE,
    NODE_METADATA,
    NODE_SPLIT,
    NODE_SECTIONIZE,
    NODE_EMBED_STORE,
    NODE_TABLE_EXTRACT,
    NODE_METRIC_NORMALIZE,
    NODE_COMMIT_STATUS,
)


def build_import_graph(entry_node: str = NODE_PARSE) -> StateGraph:
    if entry_node not in ALL_NODES:
        raise ValueError(f"unknown entry_node: {entry_node!r}, expected one of {ALL_NODES}")
    """组装 StateGraph：8 节点 + 条件边 + 并行 fan-out。

    parse ─ok→ metadata ─ok→ split → sectionize ─┬─∥→ embed_store ──────────────┐
            │                 │                 └─∥→ table_extract → normalize ─┤
            └fail─────────────┴fail──────────────────→ commit_status ←──────────┘ → END
    """
    g = StateGraph(state_schema=ImportState, context_schema=DataAgentContext)

    g.add_node(NODE_PARSE, timed_node(NODE_PARSE, node_parse))
    g.add_node(NODE_METADATA, timed_node(NODE_METADATA, node_metadata))
    g.add_node(NODE_SPLIT, timed_node(NODE_SPLIT, node_split))
    g.add_node(NODE_SECTIONIZE, timed_node(NODE_SECTIONIZE, node_sectionize))
    g.add_node(NODE_EMBED_STORE, timed_node(NODE_EMBED_STORE, node_embed_store))
    g.add_node(NODE_TABLE_EXTRACT, timed_node(NODE_TABLE_EXTRACT, node_table_extract))
    g.add_node(NODE_METRIC_NORMALIZE, timed_node(NODE_METRIC_NORMALIZE, node_metric_normalize))
    g.add_node(NODE_COMMIT_STATUS, timed_node(NODE_COMMIT_STATUS, node_commit_status))

    g.add_edge(START, entry_node)

    g.add_conditional_edges(
        NODE_PARSE,
        route_after_parse,
        {"ok": NODE_METADATA, "fail": NODE_COMMIT_STATUS},
    )
    g.add_conditional_edges(
        NODE_METADATA,
        route_after_metadata,
        {"ok": NODE_SPLIT, "fail": NODE_COMMIT_STATUS},
    )

    g.add_edge(NODE_SPLIT, NODE_SECTIONIZE)

    # S5a 向量与 S5b 指标并行；路由返回节点名列表即 fan-out
    g.add_conditional_edges(
        NODE_SECTIONIZE,
        route_after_sectionize,
        [NODE_EMBED_STORE, NODE_TABLE_EXTRACT],
    )

    g.add_edge(NODE_TABLE_EXTRACT, NODE_METRIC_NORMALIZE)

    # 汇合：两分支均完成后执行 commit（多源 start_key 列表）
    g.add_edge([NODE_EMBED_STORE, NODE_METRIC_NORMALIZE], NODE_COMMIT_STATUS)

    g.add_edge(NODE_COMMIT_STATUS, END)

    return g


async def bridge_stream_to_hub(
        graph,
        state: Mapping[str, Any] | dict,
        context: DataAgentContext,
        document_id: str,
        hub: ProgressHub | None,
) -> ImportState:
    """astream custom 事件桥接到 ProgressHub，返回最终 state。

    为什么 astream 而不是 ainvoke：ainvoke 拿不到 stream_writer 的自定义事件，
    SSE 进度推送依赖这些事件（docs 上传接口设计）。
    values 模式取终态，custom 模式取节点 writer 的进度。
    """
    final: ImportState | None = None
    async for mode, chunk in graph.astream(
            state,
            context=context,
            stream_mode=["custom", "values"],
    ):
        if mode == "custom" and hub is not None:
            hub.publish(document_id, chunk)
        elif mode == "values":
            final = chunk
    return final if final is not None else dict(state)  # type: ignore[return-value]


async def run_ingest(
        pdf_path: str,
        start_from: str | None = None,
        prior_state: Mapping[str, Any] | None = None,
        context: DataAgentContext | None = None,
        progress_hub: ProgressHub | None = None,
        document_id: str | None = None,
) -> ImportState:
    """编译并执行导入图；start_from 支持节点级重跑。

    start_from 为节点名（如 node_table_extract）时，从该节点起执行，
    上游字段用 prior_state（或 initial_state 默认值）填充。
    context 从 API 依赖注入传入；progress_hub 非空时推送 SSE 进度事件。
    document_id 由上传落盘时算好传入则不重算，避免后台任务再读一遍 PDF。
    """
    if not document_id:
        document_id = sha1_file(pdf_path)
    state = initial_state(pdf_path=pdf_path, document_id=document_id)
    if prior_state:
        for key, value in prior_state.items():
            if value is not None:
                state[key] = value  # type: ignore[literal-required]

    entry_node = start_from or NODE_PARSE
    graph = build_import_graph(entry_node=entry_node).compile()
    final = await bridge_stream_to_hub(
        graph=graph,
        state=state,
        context=context or DataAgentContext(),
        document_id=document_id,
        hub=progress_hub,
    )
    # 终态事件：SSE 订阅端靠 finished 关流（import_router._event_stream）。
    # 不发则订阅端永远等不到结束信号，前端导入页挂住。
    if progress_hub is not None:
        progress_hub.publish(
            document_id,
            {
                "type": "finished",
                "parse_status": (final or {}).get("parse_status"),
            },
        )
    return final


if __name__ == "__main__":
    async def test():
        pdf_path = PROJECT_ROOT / "report" / "000006_2024_深振业A_2024年年度报告_2025-04-26.pdf"
        result = await run_ingest(str(pdf_path))
        print(result)


    asyncio.run(test())
