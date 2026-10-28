"""年报上传 + SSE 进度接口。

POST 上传立刻返回 document_id，工作流丢后台跑；
GET 进度用 SSE，前端 EventSource 直连即可。
"""
from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile
from fastapi.responses import StreamingResponse

from app.entities.response_data import ResponseData
from app.import_graph.api.dependencies import get_import_service, get_progress_hub
from app.import_graph.api.services.import_service import ImportService
from app.import_graph.progress_hub import ProgressHub

router = APIRouter()


@router.post("/api/import/report")
async def upload_report(
    background_tasks: BackgroundTasks,
    service: Annotated[ImportService, Depends(get_import_service)],
    file: UploadFile = File(...),
) -> ResponseData:
    payload = await file.read()
    try:
        saved = service.save_upload(payload, filename=file.filename or "")
    except ValueError as e:
        return ResponseData(code=400, message=str(e))

    background_tasks.add_task(
        service.start_ingest,
        pdf_path=saved["pdf_path"],
        document_id=saved["document_id"],
    )
    return ResponseData(
        data={"document_id": saved["document_id"], "source_file": saved["source_file"]}
    )


async def _event_stream(hub: ProgressHub, document_id: str):
    # 用 has() 而非 history 是否为空：上传即 register，后台首事件可能还没到，
    # 「暂无事件」不等于「id 不存在」
    if not hub.has(document_id):
        yield f"data: {json.dumps({'type': 'error', 'message': 'unknown document_id'}, ensure_ascii=False)}\n\n"
        return

    try:
        async for event in hub.subscribe(document_id, from_index=0):
            yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
            if event.get("type") == "finished":
                break
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"


@router.get("/api/import/progress/{document_id}")
async def import_progress(
    document_id: str,
    hub: Annotated[ProgressHub, Depends(get_progress_hub)],
) -> StreamingResponse:
    return StreamingResponse(
        _event_stream(hub, document_id),
        media_type="text/event-stream",
    )
