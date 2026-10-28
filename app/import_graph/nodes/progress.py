"""节点进度事件工具：统一 writer 事件格式，SSE 前端靠它渲染进度。

事件格式与 my-data-agent 对齐：{"type":"progress","step","status"}。
writer 可能为 None（直接单测调节点时），统一判空避免到处 if。
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from langgraph.types import StreamWriter


def emit_progress(writer: StreamWriter | None, step: str, status: str) -> None:
    if writer is None:
        return
    writer({"type": "progress", "step": step, "status": status})


@contextmanager
def progress_step(writer: StreamWriter | None, step: str) -> Iterator[dict]:
    """节点入口 running、出口 done/failed。

    outcome["ok"]=False 用于节点吞异常后返回 error dict 的场景（如 parse），
    这类节点不抛异常，但进度上应记 failed 而非 done。异常路径记 failed 并继续抛出。
    """
    emit_progress(writer, step, "running")
    outcome: dict = {"ok": True}
    try:
        yield outcome
    except Exception:
        emit_progress(writer, step, "failed")
        raise
    else:
        emit_progress(writer, step, "done" if outcome.get("ok", True) else "failed")
