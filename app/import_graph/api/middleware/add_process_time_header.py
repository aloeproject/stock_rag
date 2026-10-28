import uuid
from app.core.context import request_id_ctx_var
from fastapi import Request

from starlette.middleware.base import BaseHTTPMiddleware


class add_process_time_header(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 调用路径函数之前
        request_id_ctx_var.set(uuid.uuid4())
        # 调用路径函数
        return await call_next(request)
