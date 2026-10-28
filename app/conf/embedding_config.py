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
class EmbeddingConfig:
    # 远程 DashScope text-embedding-v4：本地 BGE-M3 CPU 单条 ~2.2s，年报 500+ chunk 要 20+ 分钟
    model: str
    api_key: str
    dimension: int
    # docs/02c §3.2：规则替换默认关，评测确认指代仍影响召回后再开
    resolve_pronoun: bool


embedding_config = EmbeddingConfig(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
    api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY"),
    dimension=int(os.getenv("EMBEDDING_DIMENSION", "1024")),
    resolve_pronoun=_env_bool("EMBEDDING_RESOLVE_PRONOUN", default=False),
)
