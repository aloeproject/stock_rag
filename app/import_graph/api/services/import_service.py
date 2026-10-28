"""年报导入服务：落盘、算 document_id、调度后台工作流。

为什么 document_id 在这里算好传入：run_ingest 内 sha1_file 会再读一遍 PDF，
落盘后 hash 一次并传入，避免后台任务重复读大文件。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable

from app.import_graph.progress_hub import ProgressHub
from app.utils.file import sha1_file
from app.utils.path_util import PROJECT_ROOT


class ImportService:
    def __init__(
        self,
        run_ingest: Callable[..., Awaitable[Any]],
        progress_hub: ProgressHub | None = None,
        context_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._run_ingest = run_ingest
        self._progress_hub = progress_hub or ProgressHub()
        self._context_factory = context_factory

    @property
    def progress_hub(self) -> ProgressHub:
        return self._progress_hub

    def save_upload(self, payload: bytes, filename: str) -> dict[str, str]:
        """校验并落盘到 report/，返回 document_id / source_file / pdf_path。"""
        name = Path(filename or "").name  # 防路径穿越：只取 basename
        if not name.lower().endswith(".pdf"):
            raise ValueError("仅支持 .pdf 年报文件")
        if not payload:
            raise ValueError("上传文件为空")

        report_dir = PROJECT_ROOT / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = report_dir / name
        pdf_path.write_bytes(payload)

        document_id = sha1_file(str(pdf_path))
        # 落盘即登记：否则后台首事件前订阅会误判 unknown（POST 立刻返回的竞态）
        self._progress_hub.register(document_id)

        return {
            "document_id": document_id,
            "source_file": name,
            "pdf_path": str(pdf_path),
        }

    async def start_ingest(self, pdf_path: str, document_id: str) -> None:
        """后台执行导入工作流；事件推 progress_hub 供 SSE 订阅。"""
        context = self._context_factory() if self._context_factory else None
        await self._run_ingest(
            pdf_path=pdf_path,
            document_id=document_id,
            context=context,
            progress_hub=self._progress_hub,
        )
