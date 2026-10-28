from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class QdrantConfig:
    host: str
    port: int
    embedding_size: int
    collection: str


def _host_only(raw: str) -> str:
    """env 可能写成 http://host:port（用户照抄浏览器地址）；build_url 只要主机名。"""
    raw = raw.strip()
    for scheme in ("http://", "https://"):
        if raw.startswith(scheme):
            raw = raw[len(scheme) :]
    return raw.split("/")[0].split(":")[0]


qdrant_config = QdrantConfig(
    host=_host_only(os.getenv("QDRANT_HOST", "localhost")),
    port=int(os.getenv("QDRANT_PORT", 6333)),
    embedding_size=int(os.getenv("QDRANT_EMBEDDING_SIZE", 1024)),
    collection=os.getenv("QDRANT_COLLECTION", "report_chunks"),
)
