from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.entities.response_data import ResponseData


def register_exception_handlers(app: FastAPI):
    """注册自定义异常处理器"""

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        # 提取第一个错误信息
        first_error = errors[0] if errors else {}
        field = first_error.get("loc", ["unknown"])[-1]
        msg = first_error.get("msg", "参数错误")

        return JSONResponse(
            status_code=400,
            content=ResponseData(
                code=400,
                message=f"{field}: {msg}",
                data=errors
            ).__dict__
        )