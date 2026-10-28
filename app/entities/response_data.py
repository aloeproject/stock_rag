from dataclasses import dataclass
from typing import Any


@dataclass
class ResponseData:
    code: int = 200
    message: str = "success"
    data: Any = None
