import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.import_graph.api.middleware.add_process_time_header import add_process_time_header
from app.import_graph.api.middleware.cors import setup_cors
from app.import_graph.api.middleware.exception_handler import register_exception_handlers
from app.import_graph.api.routes.import_router import router as import_router
from app.import_graph.api.routes.lifespan import lifespan
from app.query_graph.api.routes.query_router import router as query_router

app = FastAPI(lifespan=lifespan)

setup_cors(app)
app.include_router(import_router)
app.include_router(query_router)
app.add_middleware(add_process_time_header)
register_exception_handlers(app)

# 前端静态页（frontend/*.html）与 API 同源，省去跨域
_FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
