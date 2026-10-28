from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    llm_model: str
    llm_temperature: float
    # 思考链拖慢抽取，默认关；需要推理质量时用 LLM_ENABLE_THINKING=true 打开
    llm_enable_thinking: bool


# env 键与 docs/01 §4.1 一致：LLM_API_BASE / LLM_API_KEY / LLM_MODEL
llm_config = LLMConfig(
    base_url=os.getenv("LLM_API_BASE"),
    api_key=os.getenv("LLM_API_KEY"),
    llm_model=os.getenv("LLM_MODEL", "qwen-plus"),
    llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
    llm_enable_thinking=_env_bool("LLM_ENABLE_THINKING", default=False),
)
