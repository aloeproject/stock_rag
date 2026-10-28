"""ProgressHub 测试：历史缓冲、实时订阅、后订阅补发。

为什么需要历史缓冲：SSE 客户端可能在导入中途才连上（刷新页面），
若只推实时事件会丢掉前面步骤的进度。
"""
import asyncio

from app.import_graph.progress_hub import ProgressHub


def test_publish_and_collect_history():
    hub = ProgressHub()
    hub.publish("doc1", {"type": "progress", "step": "解析PDF", "status": "running"})
    hub.publish("doc1", {"type": "progress", "step": "解析PDF", "status": "done"})

    assert hub.history("doc1") == [
        {"type": "progress", "step": "解析PDF", "status": "running"},
        {"type": "progress", "step": "解析PDF", "status": "done"},
    ]


def test_history_unknown_document_is_empty():
    hub = ProgressHub()
    assert hub.history("nope") == []


def test_register_marks_known_without_events():
    """register 后 has=True，但 history 仍空——区分「已登记没事件」和「id 不存在」。"""
    hub = ProgressHub()
    hub.register("doc1")

    assert hub.has("doc1") is True
    assert hub.history("doc1") == []
    assert hub.has("nope") is False


def test_documents_are_isolated():
    hub = ProgressHub()
    hub.publish("doc1", {"type": "progress", "step": "A", "status": "running"})
    hub.publish("doc2", {"type": "progress", "step": "B", "status": "running"})

    assert [e["step"] for e in hub.history("doc1")] == ["A"]
    assert [e["step"] for e in hub.history("doc2")] == ["B"]


def test_subscribe_replays_history_then_live():
    async def scenario():
        hub = ProgressHub()
        hub.publish("doc1", {"type": "progress", "step": "解析PDF", "status": "running"})

        received: list[dict] = []

        async def consume():
            async for event in hub.subscribe("doc1", from_index=0):
                received.append(event)
                if len(received) == 2:
                    break

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)  # 让消费者先进入订阅
        hub.publish("doc1", {"type": "progress", "step": "解析PDF", "status": "done"})
        await asyncio.wait_for(task, timeout=1)
        return received

    assert asyncio.run(scenario()) == [
        {"type": "progress", "step": "解析PDF", "status": "running"},
        {"type": "progress", "step": "解析PDF", "status": "done"},
    ]


def test_subscribe_without_history_waits_for_live():
    async def scenario():
        hub = ProgressHub()
        received: list[dict] = []

        async def consume():
            async for event in hub.subscribe("doc1", from_index=0):
                received.append(event)
                break

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        hub.publish("doc1", {"type": "finished", "parse_status": "done"})
        await asyncio.wait_for(task, timeout=1)
        return received

    assert asyncio.run(scenario()) == [{"type": "finished", "parse_status": "done"}]


def test_register_document_is_isolated_from_unregistered():
    hub = ProgressHub()
    hub.register("doc1")

    assert hub.has("doc1") is True
    assert hub.has("doc2") is False
