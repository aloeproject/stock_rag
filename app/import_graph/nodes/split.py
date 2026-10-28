from __future__ import annotations

import json
import sys
from pathlib import Path

from langgraph.runtime import Runtime

from app.import_graph.context import DataAgentContext
from app.utils.file import sha1_file
from app.utils.path_util import PROJECT_ROOT

# 支持直接运行本文件时解析到项目根（与 metadata.py 同款约定）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from langgraph.types import StreamWriter

from app.core.logger import logger
from app.import_graph.nodes.progress import progress_step
from app.import_graph.split.config import SplitConfig
from app.import_graph.split.dual_track import split_document
from app.import_graph.state import ImportState


def _load_inputs(state: ImportState) -> tuple[str, list[dict], str]:
    md_path = state.get("markdown_path") or ""
    pages_path = state.get("pages_map_path") or ""
    document_id = state.get("document_id") or ""

    md_content = Path(md_path).read_text(encoding="utf-8") if md_path else ""
    pages: list[dict] = []
    if pages_path:
        if Path(pages_path).exists():
            pages = json.loads(Path(pages_path).read_text(encoding="utf-8"))
        else:
            # 指定了路径却读不到 = 配置错误。不告警的话 page_no 会静默全 0
            logger.warning(f"[node_split] pages 文件不存在，页码将全为 0: {pages_path}")
    return md_content, pages, document_id


def _backup_chunks(markdown_path: str, chunks: list[dict]) -> None:
    """落 chunks.json 备份：排查切分质量时不需要重跑上游。"""
    backup_path = Path(markdown_path).parent / "chunks.json"
    backup_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def node_split(state: ImportState, runtime: Runtime[DataAgentContext], writer: StreamWriter | None = None) -> dict:
    """S3 双轨切分：叙述 + 表格 → chunks。

    设计见 docs/02a-文本切分与Chunk设计.md。失败落到 State，不裸抛。
    """
    logger.info(f"[node_split] 开始处理文档: {state.get('document_id')}")
    with progress_step(writer, "切分"):
        md_content, pages, document_id = _load_inputs(state)

        if not md_content.strip():
            logger.error("[node_split] content.md 为空，无法切分")
            return {"chunks": [], "failed_chunks": 0, "error_stage": "split", "error": "content.md 为空"}

        try:
            chunks = split_document(
                md_content=md_content,
                pages=pages,
                document_id=document_id,
                file_title=state.get("source_file") or "default_file",
                cfg=SplitConfig(),
            )
        except (OSError, ValueError, KeyError) as e:
            # 只捕获可预期的输入/解析错误；编程错误继续抛出便于暴露
            logger.exception("[node_split] 切分失败")
            return {"chunks": [], "failed_chunks": 0, "error_stage": "split", "error": str(e)}

        failed = sum(1 for c in chunks if c.get("failed"))
        live = [c for c in chunks if not c.get("failed")]

        if state.get("markdown_path"):
            try:
                _backup_chunks(state["markdown_path"], live)
            except OSError:
                logger.warning("[node_split] chunks.json 备份失败，不影响入库")

        logger.info(f"[node_split] 产出 {len(live)} 个 chunk（失败 {failed}）")
        return {"chunks": live, "failed_chunks": failed}


if __name__ == "__main__":
    pdf_path = PROJECT_ROOT / "report" / "000006_2024_深振业A_2024年年度报告_2025-04-26.pdf"
    document_id = sha1_file(pdf_path)
    path_root = PROJECT_ROOT / "data" / "parse" / "45d1b670ffb64d02650010648076fdb14aa01d42"
    markdown_path = (
            path_root / "content.md"
    )
    # parse.py 落盘的是 pages.json，不是 pages_map.json（state 键名叫 pages_map_path）
    pages_map_path = path_root / "pages.json"
    state: ImportState = {
        "pdf_path": str(pdf_path),
        "markdown_path": str(markdown_path),
        "pages_map_path": str(pages_map_path),
        "document_id": document_id,
    }  # type: ignore[typeddict-item]
    print(node_split(state, None, None))
