"""查询服务：编译图 + 转发 custom 事件（docs/05 §6）。

为什么 service 与 routes 分层：路由只做 HTTP，业务调度（图编译、事件转发、
异常兜底）进 service——单测可绕开 FastAPI 直接测 service（05 §2）。
为什么不重试整个图：generate/hyde 已内建重试（04 §9）；service 层重跑整图
会把 token 重复推给前端（05 §6）。
"""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any, Awaitable, Callable

from app.query_graph.context import QueryAgentContext

# 终态事件发完关流（04 §8）；clarify/out_of_scope 是可行动提示不是故障
TERMINAL_TYPES = frozenset({"done", "error", "clarify", "out_of_scope"})


class QueryService:
    def __init__(
        self,
        run_query_stream: Callable[..., AsyncIterator[dict]],
        context: QueryAgentContext | None = None,
    ) -> None:
        self._run_query_stream = run_query_stream
        self._context = context or QueryAgentContext()

    async def stream_events(
        self, query: str, session_id: str, request_id: str
    ) -> AsyncIterator[dict]:
        # 空 session_id 由服务端生成并透传给图（05 §3.1/§6）：前端从首帧拿到，下一轮带回
        if not session_id:
            session_id = uuid.uuid4().hex
        yield {"type": "session", "session_id": session_id, "request_id": request_id}

        try:
            async for event in self._run_query_stream(
                query=query,
                session_id=session_id,
                request_id=request_id,
                context=self._context,
            ):
                yield event
                # 终态事件发完关流（04 §8）；clarify/out_of_scope 不补发 error
                if event.get("type") in TERMINAL_TYPES:
                    return
        except Exception as e:
            # 兜底「节点没来得及发 error 就炸了」的缺口；必须产出事件不吞异常（05 §6）
            yield {
                "type": "error",
                "code": "INTERNAL",
                "message": str(e),
                "stage": "api",
            }
