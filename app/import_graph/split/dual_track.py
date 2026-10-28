from __future__ import annotations

from app.core.logger import logger
from app.import_graph.split.config import SplitConfig
from app.import_graph.split.narrative_refine import refine_sections
from app.import_graph.split.page_map import PageMap
from app.import_graph.split.table_extract import extract_tables, strip_tables
from app.import_graph.split.table_split import build_table_chunks, extract_caption
from app.import_graph.split.title_split import split_by_title
from app.import_graph.split.token_estimator import estimate_token_count

# chunk_type 二值：叙述 / 表格，供 S4 归类与 Qdrant payload 过滤
TYPE_NARRATIVE = "narrative"
TYPE_TABLE = "table"


def _has_narrative_body(section: dict) -> bool:
    """标题行之外还有实质文字才留——纯表题章节在此被过滤。"""
    content = section.get("content") or ""
    body = content.split("\n", 1)[1] if "\n" in content else ""
    return bool(body.strip())


def split_document(
    md_content: str,
    pages: list[dict],
    document_id: str,
    file_title: str,
    cfg: SplitConfig | None = None,
) -> list[dict]:
    """双轨切分总入口：轨道 A 叙述 + 轨道 B 表格，合并后统一编号。

    为什么双轨：表格混进递归分割器会切断表头-行语义，还会与 S5b 双向重复
    （docs/02a §1）。两轨在此汇合，下游节点拿到的是同构的 chunks 列表。
    """
    cfg = cfg or SplitConfig()
    page_map = PageMap(pages)

    tables = extract_tables(md_content)
    stripped = strip_tables(md_content)

    narrative_raw = split_by_title(stripped, file_title=file_title)
    # 纯表题章节（标题下只有被抠走的表格）丢弃：它们是表格 caption 的宿主，
    # 抠表后只剩空白，进向量库是噪音；标题信息由 extract_caption 给 table chunk
    narrative_raw = [s for s in narrative_raw if _has_narrative_body(s)]
    narrative = refine_sections(narrative_raw, cfg)

    chunks: list[dict] = []
    for sec in narrative:
        content = (sec.get("content") or "").strip()
        if not content:
            continue
        char_start = sec.get("char_start") or 0
        char_end = sec.get("char_end") or char_start
        page_no, page_end = page_map.page_span(char_start, char_end)
        chunks.append({
            "chunk_type": TYPE_NARRATIVE,
            "content": content,
            "token_count": estimate_token_count(content),
            "char_start": char_start,
            "char_end": char_end,
            "page_no": page_no,
            "page_end": page_end,
            "title": sec.get("title", ""),
            "parent_title": sec.get("parent_title", ""),
            "part": sec.get("part", 1),
            "section_path": sec.get("section_path", ""),
            "section": "",
            "table_id": "",
            "table_caption": "",
        })

    for table in tables:
        caption = extract_caption(md_content, table)
        page_no, page_end = page_map.page_span(table["char_start"], table["char_end"])
        # table_id 与 tables/ 目录、S5b report_tables 对齐：table_{page}_{n}
        table_id = f"table_{page_no}_{len([c for c in chunks if c['chunk_type'] == TYPE_TABLE]) + 1}"
        try:
            table_chunks = build_table_chunks(
                html=table["html"],
                table_id=table_id,
                caption=caption,
                section_path=_nearest_section(narrative, table["char_start"]),
                char_start=table["char_start"],
                char_end=table["char_end"],
                cfg=cfg,
            )
        except Exception:
            # 单表失败跳过并计数，正文切分不受影响（docs/02a §7）
            chunks.append(_failed_table_placeholder(table_id, table, page_no, page_end))
            continue

        for tc in table_chunks:
            tc.setdefault("token_count", estimate_token_count(tc["content"]))
            tc["page_no"] = page_no
            tc["page_end"] = page_end
            tc["section"] = ""
            # 不可约超限行：内容保留，上层按 02a §7 记 warning
            if tc.get("oversized"):
                logger.warning(
                    f"[split] 表格块超硬上限: {tc['table_id']} part={tc['part']} "
                    f"token={tc['token_count']}"
                )
            chunks.append(tc)

    return _assign_chunk_ids(chunks, document_id)


def _nearest_section(narrative: list[dict], char_start: int) -> str:
    """表格落在哪个章节区间就继承哪个的 section_path。

    为什么按 char_start 而不是 char_end：表往往在章节中间（表前有标题、表后还有正文），
    用 char_end 判等会误挂到上一章。
    """
    best = ""
    for sec in narrative:
        if (sec.get("char_start") or 0) <= char_start:
            best = sec.get("section_path", best)
        else:
            break
    return best


def _failed_table_placeholder(table_id: str, table: dict, page_no: int, page_end: int) -> dict:
    return {
        "chunk_type": TYPE_TABLE,
        "content": "",
        "token_count": 0,
        "char_start": table["char_start"],
        "char_end": table["char_end"],
        "page_no": page_no,
        "page_end": page_end,
        "title": table_id,
        "parent_title": "",
        "part": 1,
        "section_path": "",
        "section": "",
        "table_id": table_id,
        "table_caption": "",
        "failed": True,
    }


def _assign_chunk_ids(chunks: list[dict], document_id: str) -> list[dict]:
    """chunk_id = {document_id}:{seq:04d}。

    为什么不用 uuid：同文档重跑 id 必须不变，否则 Qdrant 先删后写会留下旧点
    （幂等要求，docs/02a §3.3）。
    """
    for i, c in enumerate(chunks, start=1):
        c["chunk_id"] = f"{document_id}:{i:04d}"
        c["document_id"] = document_id
    return chunks
