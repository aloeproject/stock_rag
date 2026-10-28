"""查询 API 请求 Schema（docs/05 §3.1）。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="用户本轮原话")
    session_id: str = Field(default="", max_length=64, description="多轮会话 id，空则服务端生成")
