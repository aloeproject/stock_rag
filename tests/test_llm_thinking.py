"""LLM 思考模式配置：默认关闭（减少延迟），可经 LLM_ENABLE_THINKING 打开。"""
from types import SimpleNamespace

import pytest

from app.lm import llm_utils
from app.lm.llm_utils import get_llm_client


@pytest.fixture(autouse=True)
def _clear_cache():
    llm_utils._llm_client_cache.clear()
    yield
    llm_utils._llm_client_cache.clear()


def _capture_client(monkeypatch, enable_thinking: bool):
    captured: dict = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("app.lm.llm_utils.ChatOpenAI", fake_chat)
    monkeypatch.setattr(
        "app.lm.llm_utils.llm_config",
        SimpleNamespace(
            api_key="k",
            base_url="http://x",
            llm_model="m",
            llm_temperature=0.1,
            llm_enable_thinking=enable_thinking,
        ),
    )
    return captured


def test_thinking_disabled_by_default(monkeypatch):
    captured = _capture_client(monkeypatch, enable_thinking=False)
    get_llm_client()
    assert captured["extra_body"]["enable_thinking"] is False


def test_thinking_enabled_when_configured(monkeypatch):
    captured = _capture_client(monkeypatch, enable_thinking=True)
    get_llm_client()
    assert captured["extra_body"]["enable_thinking"] is True


def test_config_default_is_thinking_off(monkeypatch):
    monkeypatch.setenv("LLM_ENABLE_THINKING", "false")
    from app.conf import llm_config as mod

    value = mod.LLMConfig(
        base_url="http://x",
        api_key="k",
        llm_model="m",
        llm_temperature=0.1,
        llm_enable_thinking=mod._env_bool("LLM_ENABLE_THINKING"),
    )
    assert value.llm_enable_thinking is False
