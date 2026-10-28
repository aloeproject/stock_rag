from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class MysqlConfig:
    host: str
    port: int
    user: str
    password: str
    database: str


mysql_config = MysqlConfig(
    host=os.getenv("MYSQL_HOST", "localhost"),
    port=int(os.getenv("MYSQL_PORT", "3306")),
    user=os.getenv("MYSQL_USER", "root"),
    password=os.getenv("MYSQL_PASSWORD", ""),
    database=os.getenv("MYSQL_DATABASE", "stock_rag"),
)
