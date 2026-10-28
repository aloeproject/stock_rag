"""S5b 表格过滤与 LLM 抽取（docs/02d）。

为什么不全表 LLM：四报 905 表只有 44 张核心报表，逐表调用会产出假指标。
为什么 skip 表也落 raw：规则会演进（如放开补充资料），补抽不回 PDF。
为什么先删后写：重导时表数量可能变少，只 upsert 留残行。
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langgraph.runtime import Runtime
from langgraph.types import StreamWriter

from app.core.logger import logger
from app.entities.report_table import (
    ExtractStatus,
    ReportTableDbData,
    StatementScope,
    StatementType,
)
from app.import_graph.context import DataAgentContext
from app.import_graph.nodes.progress import emit_progress, progress_step
from app.import_graph.split.table_extract import extract_tables, html_table_to_markdown
from app.import_graph.split.page_map import PageMap
from app.import_graph.split.table_split import extract_caption
from app.import_graph.state import ImportState
from app.import_graph.table_classify import classify_table
from app.core.load_prompt import load_prompt
from app.lm.llm_utils import get_llm_client
from app.utils.path_util import PROJECT_ROOT


# 业务切片表额外纪律（docs/02g §5.3）：dim 字段是开放维度的定位键，
# 缺了就无法路由到 business_facts；不另开 prompt 文件避免与主表纪律漂移
_SEGMENT_RULES = """
8. 本表是业务切片表（主营业务分行业/分产品/分地区/分销售模式等）
9. 每个 item 必须带 dim_type 与 dim_name
10. dim_type 只准取：industry / product / region / channel / customer / other
11. dim_name 填行首原文（含「合计」），禁止意译、禁止合并同名不同段的行
12. 段头行（「分行业」「分产品」…）只用来定 dim_type，不作为数据行
13. 「比上年增减(%)」列照抄成独立 metric_name，不要重命名成同比
"""


def llm_extract_items(
    caption: str, table_md: str, statement_type: str, statement_scope: str
) -> dict:
    """调 LLM 抽行项目；返回 {"items": [...], "statement_scope": str}。

    Prompt 外置走 load_prompt（docs/02d §5.3）；json_mode 保证可解析。
    segment 时追加 dim 抽取纪律（docs/02g §5.3），items 多 dim_type/dim_name。
    分段计时：prompt / get_client / invoke / parse，定位慢在编排还是模型。
    """
    t0 = time.perf_counter()
    prompt = load_prompt(
        "table_extract",
        statement_type=statement_type,
        statement_scope=statement_scope or "unknown",
        caption=caption or "(无标题)",
        table_md=table_md,
        segment_rules=_SEGMENT_RULES if statement_type == "segment" else "",
    )
    t1 = time.perf_counter()
    client = get_llm_client(json_mode=True)
    t2 = time.perf_counter()
    resp = client.invoke(prompt)
    t3 = time.perf_counter()
    data = json.loads(resp.content)
    t4 = time.perf_counter()
    logger.info(
        f"[table_extract] LLM耗时 prompt={t1 - t0:.2f}s get_client={t2 - t1:.2f}s "
        f"invoke={t3 - t2:.2f}s parse={t4 - t3:.2f}s total={t4 - t0:.2f}s "
        f"prompt_len={len(prompt)} model={getattr(client, 'model_name', None) or getattr(client, 'model', '?')}"
    )
    if not isinstance(data, dict):
        raise ValueError(f"LLM 输出不是对象: {type(data)}")
    return data


def _page_of(pages_map: list[dict], offset: int) -> int | None:
    """offset → page_no。空 map 返 None；间隙归上一页（同 PageMap 语义）。

    为什么不能精确包含才返页：pages.json 历史上是碎片 span，表 offset
    落在空洞里会全 NULL；间隙属于上一页正文后的空白。
    """
    if not pages_map:
        return None
    return PageMap(pages_map).page_at(offset)


def step_1_load_tables(state: ImportState) -> list[dict]:
    """取数：content.md 的 <table> + 正上方 caption（docs/02d §5.1 步骤 1）。

    为什么不用 table_paths 当主源：caption 在 content.md 表格上方，
    table 文件只存表体，拿不到标题就无法走 title 通道。
    """
    md_path = state.get("markdown_path") or ""
    if not md_path or not Path(md_path).exists():
        # 退回 table_paths：无 caption，只能走 sig 通道
        rows = []
        for p in state.get("table_paths") or []:
            body = Path(p).read_text(encoding="utf-8")
            rows.append({"caption": "", "html": body, "raw": body, "page_no": None})
        return rows

    md = Path(md_path).read_text(encoding="utf-8")
    pages_map: list[dict] = []
    pm = state.get("pages_map_path") or ""
    if pm and Path(pm).exists():
        pages_map = json.loads(Path(pm).read_text(encoding="utf-8"))

    rows = []
    for t in extract_tables(md):
        caption = extract_caption(md, t)
        rows.append({
            "caption": caption,
            "html": t["html"],
            "raw": t["html"],
            "page_no": _page_of(pages_map, t["char_start"]),
        })
    return rows


def step_2_classify(rows: list[dict]) -> list[dict]:
    for r in rows:
        kind, how, scope = classify_table(r["caption"], r["html"])
        r["statement_type"] = kind
        r["how"] = how
        r["scope"] = scope
    return rows


def step_3_llm_extract(row: dict) -> dict:
    """核心报表 LLM 抽取；返回 items 或抛异常。

    空 items 视为失败——「抽不出」与「没有」不作区分，宁可重跑（docs/02d §6）。
    """
    table_md = html_table_to_markdown(row["html"]) if "<table" in row["html"] else row["html"]
    data = llm_extract_items(
        caption=row.get("caption") or "",
        table_md=table_md,
        statement_type=row.get("statement_type") or "",
        statement_scope=row.get("scope") or "",
    )
    items = data.get("items")
    if not items:
        raise ValueError("LLM 输出 items 为空")
    return data


# 并行只用于 LLM I/O；写库仍单线程短事务（session 非线程安全）
LLM_MAX_WORKERS = 4


def _llm_extract_with_retry(row: dict) -> dict | None:
    for attempt in range(2):
        try:
            return step_3_llm_extract(row)
        except Exception as e:
            if attempt == 0:
                logger.warning(f"[node_table_extract] LLM 失败，重试: {e}")
                continue
            logger.error(f"[node_table_extract] LLM 失败，标记 failed: {e}")
    return None


async def node_table_extract(
    state: ImportState,
    runtime: Runtime[DataAgentContext],
    writer: StreamWriter | None = None,
) -> dict:
    """S5b: 规则过滤后仅核心报表 LLM 抽取（分支 B）。

    事务边界：raw 先删后写并立即 commit，LLM 完全在事务外，每条结果一个短事务。
    不让 DB 事务跨模型调用——秒级 LLM × 多表会占死 MySQL 连接/行锁。
    失败不污染 State：单表失败计 tables_failed 并继续其他表（docs/02d §6）。
    仓储由 DataAgentContext 注入，节点内不自建 ReportTableRepository。
    """
    emit_progress(writer, "表格抽取", "running")
    document_id = state.get("document_id") or ""
    repo = runtime.context.report_table_repository
    logger.info(f"[node_table_extract] 开始处理文档: {document_id}")
    if repo is None:
        raise ValueError("[node_table_extract] DataAgentContext.report_table_repository 未注入")
    session = repo.session

    # 短事务 1：raw 落库（pending/skipped），提交后才允许碰 LLM
    try:
        t0 = time.perf_counter()
        repo.delete_by_document(document_id)
        rows = step_1_load_tables(state)
        t1 = time.perf_counter()
        rows = step_2_classify(rows)
        t2 = time.perf_counter()
        jobs: list[tuple[int, dict]] = []
        skipped = 0
        for row in rows:
            kind = row.get("statement_type")
            scope = row.get("scope") or None
            status = ExtractStatus.PENDING if kind else ExtractStatus.SKIPPED
            if scope == "":
                scope = None
            table_id = repo.insert(
                ReportTableDbData(
                    document_id=document_id,
                    page_no=row.get("page_no"),
                    table_title=next(
                        (ln[:255] for ln in (row.get("caption") or "").splitlines() if ln),
                        None,
                    ),
                    statement_type=StatementType(kind) if kind else None,
                    statement_scope=StatementScope(scope) if scope else None,
                    raw_markdown=row.get("raw") or "",
                    extracted_metrics=None,
                    extract_status=status,
                )
            )
            if not kind:
                skipped += 1
                continue
            jobs.append((table_id, row))
        t3 = time.perf_counter()
        session.commit()
        t4 = time.perf_counter()
        logger.info(
            f"[table_extract] raw阶段 load={t1 - t0:.2f}s classify={t2 - t1:.2f}s "
            f"insert={t3 - t2:.2f}s commit={t4 - t3:.2f}s total={t4 - t0:.2f}s "
            f"tables={len(rows)} core={len(jobs)}"
        )
    except Exception as e:
        logger.error(f"[node_table_extract] raw 落库失败: {e}", exc_info=True)
        session.rollback()
        emit_progress(writer, "表格抽取", "failed")
        return {
            "tables_total": 0,
            "tables_ok": 0,
            "tables_skipped": 0,
            "tables_failed": state.get("tables_failed") or 0,
        }

    ok = failed = 0
    llm_wall = 0.0
    if jobs:
        wall0 = time.perf_counter()
        # LLM 并行：只并行网络 I/O；写库在主线程逐条短事务（session 非线程安全）
        with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as pool:
            future_map = {
                pool.submit(_llm_extract_with_retry, row): (idx, table_id, row, time.perf_counter())
                for idx, (table_id, row) in enumerate(jobs, 1)
            }
            for fut in as_completed(future_map):
                idx, table_id, row, tl = future_map[fut]
                data = fut.result()

                # 短事务 2..n：单表结果独立提交，失败不拖垮已落库的 raw
                try:
                    if data is None:
                        repo.update_extract(table_id, ExtractStatus.FAILED)
                        failed += 1
                    else:
                        # sig 通道 scope 为空，用 LLM 回填；title 通道规则优先（docs/02d §5.2）
                        llm_scope = (data.get("statement_scope") or "").strip()
                        final_scope = row.get("scope") or llm_scope or "consolidated"
                        if not row.get("scope") and not llm_scope:
                            logger.warning(f"[node_table_extract] scope 回填失败，默认 consolidated: table_id={table_id}")
                        if row.get("scope") and llm_scope and llm_scope != row.get("scope"):
                            logger.warning(
                                f"[node_table_extract] LLM scope 与规则不一致，信规则: "
                                f"{llm_scope} != {row.get('scope')} table_id={table_id}"
                            )
                        repo.update_extract(
                            table_id,
                            ExtractStatus.OK,
                            extracted_metrics=data.get("items"),
                            statement_scope=final_scope,
                        )
                        ok += 1
                    td0 = time.perf_counter()
                    session.commit()
                    logger.info(
                        f"[table_extract] 表进度 {idx}/{len(jobs)} table_id={table_id} "
                        f"llm={time.perf_counter() - tl:.2f}s commit={time.perf_counter() - td0:.2f}s "
                        f"title={row.get('caption', '')[:40]!r}"
                    )
                except Exception as e:
                    logger.error(f"[node_table_extract] 结果落库失败: {e}", exc_info=True)
                    session.rollback()
                    failed += 1
                    if data is not None:
                        ok -= 1
        llm_wall = time.perf_counter() - wall0

    logger.info(
        f"[table_extract] 完成 total={len(rows)} ok={ok} skipped={skipped} failed={failed} "
        f"llm_wall={llm_wall:.2f}s workers={LLM_MAX_WORKERS}"
    )
    emit_progress(writer, "表格抽取", "done" if failed == 0 else "failed")
    return {
        "tables_total": len(rows),
        "tables_ok": ok,
        "tables_skipped": skipped,
        "tables_failed": failed,
    }


if __name__ == "__main__":
    # 真实全量：600628 新世界完整 content.md → classify → 核心表全部 LLM → 写真实 MySQL。
    # 不裁样本、不造假 md；document_id=PDF sha1、page_no 来自 parse/pages.json，与线上同构。
    import asyncio
    import json
    from collections import Counter

    from app.clients.mysql_client import MySQLClientManager
    from app.models.base import Base
    from app.repositories.report_table_repository import ReportTableRepository
    from app.utils.file import sha1_file

    REPORTS = {
        "45d1b670ffb64d02650010648076fdb14aa01d42": "000006 深振业A",
        "0b16b6765819aa7b3037c14c163a0a85fbe5a5ef": "000007 全新好",
        "013d1633320c62a34fd359a6fa5bb27f1e765355": "000008 神州高铁",
        "4ffe5afef59df4342e46bfd86b41b62992f642c7": "600628 新世界",
    }
    base = PROJECT_ROOT / "data" / "parse"
    LLM_SHA = "4ffe5afef59df4342e46bfd86b41b62992f642c7"
    PDF = PROJECT_ROOT / "report" / "600628_2024_新世界_新世界2024年年度报告_2025-04-12.pdf"
    DOC_ID = sha1_file(PDF)
    assert DOC_ID == LLM_SHA, f"document_id 与 parse 目录不一致: {DOC_ID}"

    print("== 四报分类统计（对照 docs/02d §3.4，不调 LLM）==")
    for sha, name in REPORTS.items():
        md = (base / sha / "content.md").read_text(encoding="utf-8")
        hits: Counter = Counter()
        for t in extract_tables(md):
            caption = extract_caption(md, t)
            kind, how, _ = classify_table(caption, t["html"])
            hits[f"{kind or 'skip'}{'/' + how if kind else ''}"] += 1
        n_tables = sum(hits.values())
        n_core = sum(v for k, v in hits.items() if not k.startswith("skip"))
        print(f"  {name}: 进LLM {n_core}/{n_tables} ({100 * n_core / n_tables:.1f}%) {dict(hits)}")

    mgr = MySQLClientManager()
    mgr.init()
    Base.metadata.create_all(mgr.engine)
    session = mgr.session()

    state: ImportState = {
        "document_id": DOC_ID,
        "markdown_path": str(base / LLM_SHA / "content.md"),
        "table_paths": [],
        "pages_map_path": str(base / LLM_SHA / "pages.json"),
        "stock_code": "600628",
        "stock_name": "新世界",
        "report_type": "annual",
        "report_period": "2024",
        "tables_total": 0,
        "tables_ok": 0,
        "tables_skipped": 0,
        "tables_failed": 0,
    }  # type: ignore[typeddict-item]

    repo = ReportTableRepository(session)
    ctx = DataAgentContext(report_table_repository=repo)
    patch = asyncio.run(node_table_extract(state, Runtime(context=ctx)))  # type: ignore[arg-type]
    print("== state patch（新世界全部核心表走 LLM）==")
    print({k: patch[k] for k in ("tables_total", "tables_ok", "tables_skipped", "tables_failed")})
    assert patch["tables_total"] > 0 and patch["tables_ok"] > 0

    print(f"== MySQL 落库抽查 document_id={DOC_ID} ==")
    by_status: Counter = Counter()
    for r in repo.list_by_document(DOC_ID):
        by_status[r.extract_status.value] += 1
        metrics = r.extracted_metrics or []
        if r.extract_status == ExtractStatus.SKIPPED:
            continue  # skip 行太多，只看核心表明细
        sample = json.dumps(metrics[0], ensure_ascii=False) if metrics else None
        print(
            f"  type={r.statement_type} scope={r.statement_scope} status={r.extract_status.value} "
            f"page={r.page_no} title={r.table_title!r} items={len(metrics)} "
            f"raw_len={len(r.raw_markdown or '')} sample={sample}"
        )
        if r.extract_status == ExtractStatus.OK:
            assert r.extracted_metrics, "核心表必须抽出 items"
            assert r.page_no is not None, "核心表必须有 page_no"
    print(f"  status 分布: {dict(by_status)}")
    print(f"== SQL 复核：SELECT id,page_no,statement_type,statement_scope,extract_status,table_title FROM report_tables WHERE document_id='{DOC_ID}'; ==")
    session.close()
    print("== 完成 ==")
