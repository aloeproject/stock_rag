"""导入进度事件中枢：按 document_id 缓冲 writer 事件，供 SSE 订阅。

为什么历史缓冲：前端可能导入中途才连 SSE（刷新页面），只推实时会丢前面步骤。
为什么进程内 dict 而不是 Redis：单机 FastAPI，YAGNI；多实例再换。
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator


class ProgressHub:
    def __init__(self) -> None:
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._conditions: dict[str, asyncio.Condition] = {}

    def _condition(self, document_id: str) -> asyncio.Condition:
        if document_id not in self._conditions:
            self._conditions[document_id] = asyncio.Condition()
        return self._conditions[document_id]

    def register(self, document_id: str) -> None:
        """登记 document_id（上传落盘时调用）。

        为什么需要 register：POST 上传立刻返回，后台 start_ingest 还没 publish
        第一个事件时前端已来订阅。若用「history 是否为空」判断 id 是否存在，
        会把「已登记但暂无事件」误判成 unknown 关流。register 让 has() 区分二者。
        """
        self._events.setdefault(document_id, [])

    def has(self, document_id: str) -> bool:
        return document_id in self._events

    def publish(self, document_id: str, event: dict[str, Any]) -> None:
        self._events.setdefault(document_id, []).append(event)
        cond = self._condition(document_id)
        # Condition.notify 必须在持锁状态；publish 可能在同步上下文调用，
        # 无运行中 loop 时只 append（无订阅者需要唤醒）。
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._notify(document_id, event))

    async def _notify(self, document_id: str, event: dict[str, Any]) -> None:
        cond = self._condition(document_id)
        async with cond:
            cond.notify_all()

    def history(self, document_id: str) -> list[dict[str, Any]]:
        return list(self._events.get(document_id, []))

    async def subscribe(self, document_id: str, from_index: int = 0) -> AsyncIterator[dict[str, Any]]:
        """先补发 [from_index:] 历史，再阻塞等实时事件。"""
        index = from_index
        while True:
            events = self._events.get(document_id, [])
            while index < len(events):
                yield events[index]
                index += 1
            cond = self._condition(document_id)
            async with cond:
                # 二次检查：避免「检查后、等待前」publish 的通知丢失
                events = self._events.get(document_id, [])
                if index < len(events):
                    continue
                await cond.wait()
