"""generate_embeddings 远程 DashScope 实现测试（docs 性能改造：本地 BGE-M3 CPU → text-embedding-v4）。

为什么 mock TextEmbedding.call：测试不打真实 API，只锁返回结构与错误语义。
为什么断言稀疏必须带 token 语义：旧实现「字符下标→1.0」是假稀疏，检索端 lexical 匹配完全失效。
"""
from __future__ import annotations

import pytest

from app.lm import embedding_utils


def _fake_resp(texts: list[str]):
    """按真实 dashscope 返回结构造响应：embedding + sparse_embedding[{index,token,value}]。"""

    class _Resp:
        status_code = 200

        def __init__(self):
            self.output = {
                "embeddings": [
                    {
                        "text_index": i,
                        "embedding": [0.1] * 1024,
                        "sparse_embedding": [
                            {"index": 102214, "token": "衣服", "value": 2.0781},
                            {"index": 3837, "token": "的", "value": 0.895},
                        ],
                    }
                    for i in range(len(texts))
                ]
            }
            self.usage = {"total_tokens": len(texts)}

    return _Resp()


def test_generate_embeddings_returns_1024_dense_and_token_sparse(monkeypatch):
    captured = {}

    def fake_call(model, input, **kwargs):
        captured["model"] = model
        captured["input"] = input
        captured["kwargs"] = kwargs
        return _fake_resp(input)

    monkeypatch.setattr(embedding_utils.TextEmbedding, "call", fake_call)

    result = embedding_utils.generate_embeddings(["衣服质量好", "营收增长"])

    assert len(result["dense"]) == 2
    assert all(len(d) == 1024 for d in result["dense"])
    # 稀疏必须是 token 语义的 {index: weight}，不是字符下标 0,1,2...
    assert result["sparse"][0] == {102214: 2.0781, 3837: 0.895}
    assert captured["model"] == "text-embedding-v4"
    assert captured["kwargs"].get("output_type") == "dense&sparse"
    assert captured["kwargs"].get("dimension") == 1024


def test_generate_embeddings_rejects_empty_or_non_list():
    with pytest.raises(ValueError):
        embedding_utils.generate_embeddings([])
    with pytest.raises(ValueError):
        embedding_utils.generate_embeddings("not a list")


def test_generate_embeddings_raises_on_api_failure(monkeypatch):
    def boom(model, input, **kwargs):
        raise RuntimeError("dashscope down")

    monkeypatch.setattr(embedding_utils.TextEmbedding, "call", boom)

    with pytest.raises(RuntimeError):
        embedding_utils.generate_embeddings(["x"])


def test_generate_embeddings_batches_over_api_limit(monkeypatch):
    """DashScope 单次 input 上限 10 条；超限自动分批且结果按原顺序拼回。"""
    calls: list[int] = []

    def fake_call(model, input, **kwargs):
        calls.append(len(input))
        return _fake_resp(input)

    monkeypatch.setattr(embedding_utils.TextEmbedding, "call", fake_call)

    texts = [f"t{i}" for i in range(25)]
    result = embedding_utils.generate_embeddings(texts)

    assert calls == [10, 10, 5]
    assert len(result["dense"]) == 25
    assert len(result["sparse"]) == 25


def test_generate_embeddings_empty_sparse_is_empty_dict(monkeypatch):
    class _Resp:
        status_code = 200

        def __init__(self):
            self.output = {
                "embeddings": [
                    {
                        "text_index": 0,
                        "embedding": [0.2] * 1024,
                        "sparse_embedding": [],
                    }
                ]
            }

    monkeypatch.setattr(
        embedding_utils.TextEmbedding, "call", lambda model, input, **kw: _Resp()
    )

    result = embedding_utils.generate_embeddings(["纯数字 2024"])
    assert result["sparse"] == [{}]
