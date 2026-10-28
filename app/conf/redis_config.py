from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class RedisConfig:
    host: str
    port: int
    password: str
    db: int


redis_config = RedisConfig(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    password=os.getenv("REDIS_PASSWORD", ""),
    db=int(os.getenv("REDIS_DB", "0")),
)
