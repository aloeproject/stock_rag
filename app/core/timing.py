"""节点耗时中间件：包装图节点统一打印各阶段耗时。

为什么包装节点而不是 FastAPI 中间件：HTTP 层只能量到整请求总耗时，
看不出慢在 generate 还是 rerank；按节点打点才能定位瓶颈。
"""
from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Callable

from app.core.logger import logger


def _log(name: str, state: Any, t0: float) -> None:
    rid = ""
    if isinstance(state, dict):
        rid = state.get("request_id") or state.get("document_id") or ""
    logger.info(f"[timing] {name} rid={rid} 耗时 {time.perf_counter() - t0:.2f}s")


def timed_node(name: str, fn: Callable) -> Callable:
    """包装节点函数，执行完打印耗时；兼容 sync / async 节点。"""
    if asyncio.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def aw(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                _log(name, args[0] if args else None, t0)

        return aw

    @functools.wraps(fn)
    def w(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _log(name, args[0] if args else None, t0)

    return w
