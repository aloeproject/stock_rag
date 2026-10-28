"""node_split 节点测试——读 content.md + pages.json，返回 State 增量 patch。"""
import json
from pathlib import Path

from langgraph.runtime import Runtime

from app.core.logger import logger
from app.import_graph.context import DataAgentContext
from app.import_graph.nodes.split import node_split


def _write_parse_dir(tmp_path: Path) -> dict:
    md = (
        "# 第十节 财务报告\n"
        "前言说明文字。\n"
        "<table><tr><td>科目</td><td>本期数</td></tr>"
        "<tr><td>营业收入</td><td>1,120</td></tr></table>\n"
        "后文内容。\n"
    )
    md_path = tmp_path / "content.md"
    md_path.write_text(md, encoding="utf-8")
    pages_path = tmp_path / "pages.json"
    pages_path.write_text(
        json.dumps([{"char_start": 0, "char_end": len(md), "page_no": 10}]),
        encoding="utf-8",
    )
    return {
        "document_id": "doc1",
        "markdown_path": str(md_path),
        "pages_map_path": str(pages_path),
    }


def test_node_split_returns_patch_with_chunks(tmp_path, runtime: Runtime[DataAgentContext]):
    state = _write_parse_dir(tmp_path)
    patch = node_split(state, runtime)
    assert "chunks" in patch
    assert "failed_chunks" in patch
    assert len(patch["chunks"]) >= 2
    assert patch["failed_chunks"] == 0


def test_node_split_writes_backup(tmp_path, runtime: Runtime[DataAgentContext]):
    state = _write_parse_dir(tmp_path)
    node_split(state, runtime)
    backup = Path(state["markdown_path"]).parent / "chunks.json"
    assert backup.exists()


def test_node_split_empty_content_sets_error(tmp_path, runtime: Runtime[DataAgentContext]):
    md_path = tmp_path / "content.md"
    md_path.write_text("", encoding="utf-8")
    pages_path = tmp_path / "pages.json"
    pages_path.write_text("[]", encoding="utf-8")
    state = {
        "document_id": "doc1",
        "markdown_path": str(md_path),
        "pages_map_path": str(pages_path),
    }
    patch = node_split(state, runtime)
    assert patch.get("error_stage") == "split"


def test_node_split_maps_page_no_from_pages_file(tmp_path, runtime: Runtime[DataAgentContext]):
    """pages.json 的 page_no 必须落到 chunk 上，不能静默变 0。"""
    state = _write_parse_dir(tmp_path)  # fixture 写死 page_no=10
    patch = node_split(state, runtime)
    assert patch["chunks"]
    assert all(c["page_no"] == 10 for c in patch["chunks"])
    assert all(c["page_end"] == 10 for c in patch["chunks"])


def test_node_split_missing_pages_file_warns(tmp_path, runtime: Runtime[DataAgentContext]):
    """pages_map_path 指向不存在的文件要告警——否则 page_no 全 0 无人知晓。"""
    md_path = tmp_path / "content.md"
    md_path.write_text("# 标题\n正文内容。\n", encoding="utf-8")
    state = {
        "document_id": "doc1",
        "markdown_path": str(md_path),
        "pages_map_path": str(tmp_path / "not_exist_pages.json"),
    }
    records: list[str] = []
    sink_id = logger.add(lambda m: records.append(str(m)), level="WARNING")
    try:
        node_split(state, runtime)
    finally:
        logger.remove(sink_id)
    assert any("pages" in r.lower() for r in records), records
