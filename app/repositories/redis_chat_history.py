"""Redis 多轮对话历史（docs/04 §4.2 / 04a §3.6）。

用法对齐 personal-assistant/app/repositories/redis_chat_history.py：
  history = get_redis_chat_history(session_id)   # 取实例
  window  = get_window_messages(history)         # 读窗口（旧→新）
  history.add_user_message(...) / add_ai_message(...)  # 写

key   : stock-rag:chat:{session_id}     # LIST
写入  : LPUSH 新消息 JSON + LTRIM 0 29  # 保留 30 条 ≈ 15 轮
TTL   : 7 天无活动过期

消息结构（解析结果回填，供下轮指代消解）：
  user 条:      {role, content, stock_codes, periods}
  assistant 条: {role, content, rewritten_query}

content 存原话不存改写句——审计要能看到用户真正输入了什么。
"""
from __future__ import annotations

import json

import redis

from app.conf.redis_config import redis_config

WINDOW_SIZE = 30
KEY_PREFIX = "stock-rag:chat:"
TTL_SECONDS = 7 * 24 * 3600


class TrimmedRedisChatHistory:
    """带自动裁剪的 Redis 消息历史（LPUSH 后 LTRIM，保头部即保最新）。"""

    def __init__(
        self,
        session_id: str,
        url: str,
        key_prefix: str = KEY_PREFIX,
        ttl: int = TTL_SECONDS,
        max_messages: int = WINDOW_SIZE,
    ) -> None:
        self.key = key_prefix + session_id
        self.ttl = ttl
        self.max_messages = max_messages
        self.redis_client = redis.Redis.from_url(url)

    def add_user_message(
        self,
        content: str,
        stock_codes: list[str] | None = None,
        periods: list[str] | None = None,
    ) -> None:
        self._append(
            {
                "role": "user",
                "content": content,
                "stock_codes": list(stock_codes or []),
                "periods": list(periods or []),
            }
        )

    def add_ai_message(self, content: str, rewritten_query: str = "") -> None:
        self._append(
            {
                "role": "assistant",
                "content": content,
                "rewritten_query": rewritten_query,
            }
        )

    def _append(self, message: dict) -> None:
        self.redis_client.lpush(self.key, json.dumps(message, ensure_ascii=False))
        self.redis_client.ltrim(self.key, 0, self.max_messages - 1)
        self.redis_client.expire(self.key, self.ttl)

    @property
    def messages(self) -> list[dict]:
        """全部保留条，旧→新。LPUSH 后 lrange 是新→旧，需反转。"""
        raw = self.redis_client.lrange(self.key, 0, self.max_messages - 1)
        items: list[dict] = []
        for blob in raw or []:
            if isinstance(blob, bytes):
                blob = blob.decode("utf-8", errors="ignore")
            try:
                items.append(json.loads(blob))
            except (TypeError, ValueError):
                continue
        return list(reversed(items))


def get_redis_chat_history(session_id: str) -> TrimmedRedisChatHistory:
    """获取 Redis 持久化的消息历史实例。"""
    if redis_config.password:
        url = (
            f"redis://:{redis_config.password}"
            f"@{redis_config.host}:{redis_config.port}/{redis_config.db}"
        )
    else:
        url = f"redis://{redis_config.host}:{redis_config.port}/{redis_config.db}"
    return TrimmedRedisChatHistory(
        session_id=session_id,
        url=url,
        key_prefix=KEY_PREFIX,
        ttl=TTL_SECONDS,
        max_messages=WINDOW_SIZE,
    )


def get_window_messages(
    history: TrimmedRedisChatHistory, k: int = WINDOW_SIZE
) -> list[dict]:
    """获取窗口内的最近 k 条消息（旧→新）。"""
    all_messages = history.messages
    return all_messages[-k:] if len(all_messages) > k else all_messages
