"""查询 SSE 接口（docs/05 §7）。

POST 本身就是 SSE 流——token 边生成边推；与导入侧两段式差异见 05 §1。
"""
from __future__ import annotations

import json
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.query_graph.api.dependencies import get_query_service
from app.query_graph.api.schemas.query_schema import QueryRequest
from app.query_graph.api.services.query_service import QueryService

router = APIRouter()


async def _event_stream(
    service: QueryService, body: QueryRequest, request_id: str
):
    """SSE 序列化：data: {json}\\n\\n，无 event: 行（对齐导入侧）。

    default=str 兜底非 JSON 原生类型（datetime/Decimal），不炸流（05 §7）。
    """
    try:
        async for event in service.stream_events(
            body.query, body.session_id, request_id
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
    except Exception as e:
        # 兜底序列化/网络层异常；service 已兜底图内异常，这里只堵最后一道
        yield f"data: {json.dumps({'type': 'error', 'code': 'INTERNAL', 'message': str(e), 'stage': 'api'}, ensure_ascii=False)}\n\n"


@router.post("/api/query/stream")
async def query_stream(
    body: QueryRequest,
    service: Annotated[QueryService, Depends(get_query_service)],
) -> StreamingResponse:
    # request_id 在路由层生成：HTTP 进入即分配追踪 id，service 失败前也能带 id 记日志（05 §7）
    request_id = f"q_{uuid.uuid4().hex[:12]}"
    return StreamingResponse(
        _event_stream(service, body, request_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
