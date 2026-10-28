from __future__ import annotations

from app.import_graph.state import ImportState


def route_after_parse(state: ImportState) -> str:
    """ok → node_metadata；fail → node_commit_status。"""
    return "ok" if state.get("parse_ok") else "fail"


def route_after_metadata(state: ImportState) -> str:
    """ok → node_split；fail → node_commit_status。"""
    return "fail" if state.get("error_stage") == "metadata" else "ok"


def route_after_sectionize(state: ImportState) -> list[str]:
    """fan-out → [node_embed_store, node_table_extract]（并行分支）。"""
    return ["node_embed_store", "node_table_extract"]
