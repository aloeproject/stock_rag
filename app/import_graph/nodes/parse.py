from __future__ import annotations

import json
import re
import shutil
import time
import zipfile
from pathlib import Path

import requests
from langgraph.runtime import Runtime
from langgraph.types import StreamWriter

from app.conf.mineru_config import mineru_config
from app.core.logger import logger
from app.import_graph.context import DataAgentContext
from app.import_graph.nodes.progress import progress_step
from app.import_graph.state import ImportState
from app.utils.file import sha1_file
from app.utils.path_util import PROJECT_ROOT

# 单次 HTTP 超时 / 轮询总超时（约 1s/页，年报 200 页左右给足余量）
HTTP_TIMEOUT = 180
POLL_TIMEOUT = 600
POLL_INTERVAL = 3


def step_1_validate_paths(state: ImportState) -> Path:
    pdf_path = state.get("pdf_path") or ""
    if not pdf_path:
        raise ValueError("[step_1_validate_paths] 没有发现传入文件，无法解析")

    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"[step_1_validate_paths] 文件不存在: {pdf_path}")

    return pdf_path_obj


def step_2_upload_and_poll(pdf_path_obj: Path) -> str:
    """上传 PDF 到 MinerU 云端并轮询，返回结果 zip 下载地址。"""
    if not mineru_config.base_url or not mineru_config.api_key:
        raise RuntimeError("[step_2_upload_and_poll] MINERU_API_BASE / MINERU_API_KEY 未配置")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {mineru_config.api_key or ''}",
    }
    data = {
        "files": [{"name": pdf_path_obj.name}],
        "model_version": "vlm",
    }

    url = f"{mineru_config.base_url}/file-urls/batch"
    response = requests.post(url, headers=headers, json=data, timeout=HTTP_TIMEOUT)
    if response.status_code != 200 or response.json().get("code") != 0:
        raise RuntimeError(
            f"[step_2_upload_and_poll] 申请上传地址失败: status={response.status_code}, body={response.text[:200]}"
        )

    response_json = response.json()
    batch_id = response_json["data"]["batch_id"]
    uploaded_url = response_json["data"]["file_urls"][0]

    http_session = requests.Session()
    http_session.trust_env = False
    try:
        with open(pdf_path_obj, "rb") as f:
            file_data = f.read()
        upload_response = http_session.put(uploaded_url, data=file_data, timeout=HTTP_TIMEOUT)
        if upload_response.status_code != 200:
            raise RuntimeError(
                f"[step_2_upload_and_poll] 上传 PDF 失败: status={upload_response.status_code}"
            )
    finally:
        http_session.close()

    poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    start_time = time.time()
    while True:
        if time.time() - start_time > POLL_TIMEOUT:
            raise TimeoutError(f"[step_2_upload_and_poll] 轮询超时({POLL_TIMEOUT}s): {pdf_path_obj.name}")

        res = requests.get(poll_url, headers=headers, timeout=HTTP_TIMEOUT)
        if res.status_code != 200:
            if 500 <= res.status_code < 600:
                time.sleep(POLL_INTERVAL)
                continue
            raise RuntimeError(
                f"[step_2_upload_and_poll] 查询解析结果失败: status={res.status_code}, body={res.text[:200]}"
            )

        json_data = res.json()
        if json_data.get("code") != 0:
            raise RuntimeError(f"[step_2_upload_and_poll] 解析业务失败: {json_data}")

        extract_result = json_data["data"]["extract_result"][0]
        state = extract_result.get("state")
        if state == "done":
            full_zip_url = extract_result["full_zip_url"]
            logger.info(f">>> [step_2_upload_and_poll] 解析完成，耗时 {time.time() - start_time:.1f}s")
            return full_zip_url
        if state == "failed":
            raise RuntimeError(f"[step_2_upload_and_poll] 云端解析失败: {extract_result}")
        time.sleep(POLL_INTERVAL)


def step_3_download_and_extract(zip_url: str, out_dir: Path, stem: str) -> Path:
    """下载 zip 并解压到 out_dir，返回主 markdown 文件路径。"""
    response = requests.get(zip_url, timeout=HTTP_TIMEOUT)
    if response.status_code != 200:
        raise RuntimeError(f"[step_3_download_and_extract] 下载 zip 失败: status={response.status_code}")

    zip_save_path = out_dir / f"{stem}_result.zip"
    zip_save_path.write_bytes(response.content)

    extract_target_dir = out_dir / stem
    if extract_target_dir.exists():
        shutil.rmtree(extract_target_dir)
    extract_target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_save_path, "r") as zip_file_object:
        zip_file_object.extractall(extract_target_dir)

    md_file_list = list(extract_target_dir.rglob("*.md"))
    if not md_file_list:
        raise RuntimeError("[step_3_download_and_extract] zip 中没有找到 md 文件")

    target_md_file = None
    for md_file in md_file_list:
        if md_file.name == f"{stem}.md":
            target_md_file = md_file
            break
    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name.lower() == "full.md":
                target_md_file = md_file
                break
    if not target_md_file:
        target_md_file = md_file_list[0]

    final_md_path = target_md_file.resolve()
    logger.info(f">>> [step_3_download_and_extract] md 文件: {final_md_path}")
    return final_md_path


def step_4_organize_outputs(md_path: Path, out_dir: Path) -> tuple[str, str, list[str]]:
    """整理 S1 产物：content.md、tables/table_{page}_{n}.md、pages.json。

    MinerU 解压目录里通常有 *_content_list.json（含 page_idx）与 markdown 表格；
    页码映射优先用 content_list，其次用正文分页标记，兜底整篇第 1 页。
    """
    document_dir = out_dir
    document_dir.mkdir(parents=True, exist_ok=True)

    content_path = document_dir / "content.md"
    shutil.copyfile(md_path, content_path)

    # 页码映射：character offset -> page_no
    content = content_path.read_text(encoding="utf-8", errors="ignore")
    pages_map_path = document_dir / "pages.json"
    pages_map = _build_pages_map(content, md_path.parent, pages_map_path)

    # 表格落盘
    tables_dir = document_dir / "tables"
    if tables_dir.exists():
        shutil.rmtree(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    table_paths = _extract_tables(content, md_path.parent, tables_dir, pages_map)

    logger.info(
        f">>> [step_4_organize_outputs] content.md={content_path}, "
        f"tables={len(table_paths)}, pages_map={pages_map_path}"
    )
    return str(content_path), str(pages_map_path), [str(p) for p in table_paths]


def _find_text_offset(content: str, text: str, cursor: int) -> int:
    """定位 content_list 文本在 content.md 中的偏移；markdown 会改写空白/换行。

    精确 find 失配时退回「空白可变」正则：真实年报 2274 条里 1908 条精确失配，
    是 page_no 全空的根因之一。
    """
    idx = content.find(text, cursor)
    if idx >= 0:
        return idx
    parts = [re.escape(p) for p in text.split() if p]
    if not parts:
        return -1
    m = re.search(r"\s+".join(parts), content[cursor:])
    return cursor + m.start() if m else -1


# 页锚点最短长度：page_number 文本是 '1'/'2'，短标题如「表」「释义」在正文里随处可中，
# 会把后页起点钉进前页（真实数据短锚点让 229/239 张表页码偏 1-2 页）。
MIN_ANCHOR_LEN = 15


def _anchor_text(item: dict) -> str:
    """content_list 项里可当页锚点的文本；page_number/image 不可靠，直接丢弃。"""
    if item.get("type") in ("page_number", "image"):
        return ""
    text = item.get("text") or item.get("table_body") or item.get("caption") or ""
    if not isinstance(text, str):
        return ""
    return text if len(text.strip()) >= MIN_ANCHOR_LEN else ""


def _build_pages_map(content: str, raw_dir: Path, pages_map_path: Path) -> list[dict]:
    """生成 [{char_start, char_end, page_no}]，优先 content_list.json。

    为什么按页首锚点铺连续区间：content_list 只能命中部分文本片段，
    若按片段记 span，片段之间的空洞会让 table_extract 查不到页码
    （真实数据 89 个碎片 span / 88 处空洞，235/239 张表 page_no=NULL）。
    为什么只用长文本锚点：见 MIN_ANCHOR_LEN。
    """
    spans: list[dict] = []

    content_list_path = None
    for p in raw_dir.rglob("*_content_list.json"):
        content_list_path = p
        break
    if content_list_path is None:
        for p in raw_dir.rglob("*content_list.json"):
            content_list_path = p
            break

    if content_list_path and content_list_path.exists():
        try:
            items = json.loads(content_list_path.read_text(encoding="utf-8"))
            cursor = 0
            # page_no → 该页最早命中的偏移，作为页起点
            page_starts: dict[int, int] = {}
            for item in items:
                text = _anchor_text(item)
                if not text:
                    continue
                idx = _find_text_offset(content, text, cursor)
                if idx < 0:
                    continue
                page_no = int(item.get("page_idx", 0)) + 1
                if page_no not in page_starts or idx < page_starts[page_no]:
                    page_starts[page_no] = idx
                cursor = idx + 1

            ordered = sorted(page_starts.items(), key=lambda kv: kv[1])
            for i, (page_no, start) in enumerate(ordered):
                end = ordered[i + 1][1] - 1 if i + 1 < len(ordered) else len(content) - 1
                spans.append({"char_start": start, "char_end": end, "page_no": page_no})
            if spans:
                # 文首到首个锚点归第一页，保证 [0, len) 无空洞
                spans[0]["char_start"] = 0
        except Exception as e:
            logger.warning(f">>> [_build_pages_map] 解析 content_list 失败，回退分页标记: {e}")
            spans = []

    if not spans:
        # 兼容形如 <!-- page 5 --> / [PAGE 5] 的分页标记
        marks = list(re.finditer(r"<!--\s*page\s+(\d+)\s*-->|\[PAGE\s+(\d+)]", content, re.I))
        if marks:
            for i, m in enumerate(marks):
                page_no = int(m.group(1) or m.group(2))
                start = m.end()
                end = marks[i + 1].start() if i + 1 < len(marks) else len(content)
                spans.append({"char_start": start, "char_end": end, "page_no": page_no})
        else:
            spans = [{"char_start": 0, "char_end": len(content), "page_no": 1}]

    pages_map_path.write_text(
        json.dumps(spans, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return spans


def _extract_tables(content: str, raw_dir: Path, tables_dir: Path, pages_map: list[dict]) -> list[Path]:
    """从 content_list / 正文 markdown 中抽出表格，写入 tables/table_{page}_{n}.md。"""
    results: list[Path] = []
    seq_by_page: dict[int, int] = {}

    def _page_of(offset: int) -> int:
        for span in pages_map:
            if span["char_start"] <= offset <= span["char_end"]:
                return int(span["page_no"])
        return 1

    def _write_table(page_no: int, body: str) -> None:
        if not body or not body.strip():
            return
        n = seq_by_page.get(page_no, 0) + 1
        seq_by_page[page_no] = n
        path = tables_dir / f"table_{page_no}_{n}.md"
        path.write_text(body, encoding="utf-8")
        results.append(path)

    # 1) content_list 中的表格（html 或 markdown）
    for p in raw_dir.rglob("*_content_list.json"):
        try:
            items = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in items:
            if item.get("type") != "table":
                continue
            body = item.get("table_body") or item.get("text") or ""
            page_no = int(item.get("page_idx", 0)) + 1
            _write_table(page_no, body)
        break

    # 2) 正文 GFM 表格（|---|），按字符偏移定位页码
    table_pat = re.compile(r"(?:^\|.*\|$\n){2,}", re.M)
    for m in table_pat.finditer(content):
        _write_table(_page_of(m.start()), m.group(0))

    return results


def node_parse(state: ImportState,runtime: Runtime[DataAgentContext], writer: StreamWriter) -> dict:
    """S1: MinerU 云端解析 PDF → markdown/tables/pages。

    流程：
      1. 校验 pdf_path
      2. 申请上传地址并 PUT PDF，轮询解析结果
      3. 下载 zip、解压出 md
      4. 整理为 data/parse/{document_id}/：content.md、tables/、pages.json
    失败：返回 parse_ok=False + error_stage=parse，不吞异常语义。
    """
    logger.info(f"[node_parse] 开始处理文档: {state.get('document_id')}")
    with progress_step(writer, "解析PDF") as outcome:
        document_id = state.get("document_id") or ""
        pdf_path = state.get("pdf_path") or ""

        try:
            pdf_path_obj = step_1_validate_paths(state)

            out_dir = PROJECT_ROOT / "data" / "parse" / (document_id or pdf_path_obj.stem)
            out_dir.mkdir(parents=True, exist_ok=True)

            zip_url = step_2_upload_and_poll(pdf_path_obj)
            md_path = step_3_download_and_extract(zip_url, out_dir, pdf_path_obj.stem)
            markdown_path, pages_map_path, table_paths = step_4_organize_outputs(md_path, out_dir)

            logger.info(f">>> [node_parse] 解析成功: document_id={document_id}, pdf={pdf_path}")
            return {
                "parse_ok": True,
                "markdown_path": markdown_path,
                "pages_map_path": pages_map_path,
                "table_paths": table_paths,
                "error": None,
                "error_stage": None,
            }
        except TimeoutError as e:
            # 超时重试 1 次
            logger.warning(f">>> [node_parse] 超时，重试 1 次: {e}")
            try:
                pdf_path_obj = step_1_validate_paths(state)
                out_dir = PROJECT_ROOT / "data" / "parse" / (document_id or pdf_path_obj.stem)
                out_dir.mkdir(parents=True, exist_ok=True)
                zip_url = step_2_upload_and_poll(pdf_path_obj)
                md_path = step_3_download_and_extract(zip_url, out_dir, pdf_path_obj.stem)
                markdown_path, pages_map_path, table_paths = step_4_organize_outputs(md_path, out_dir)
                return {
                    "parse_ok": True,
                    "markdown_path": markdown_path,
                    "pages_map_path": pages_map_path,
                    "table_paths": table_paths,
                    "error": None,
                    "error_stage": None,
                }
            except Exception as retry_err:
                logger.error(f">>> [node_parse] 重试后仍失败: {retry_err}")
                outcome["ok"] = False
                return {
                    "parse_ok": False,
                    "error": str(retry_err),
                    "error_stage": "parse",
                }
        except Exception as e:
            logger.error(f">>> [node_parse] 解析失败: {e}")
            outcome["ok"] = False
            return {
                "parse_ok": False,
                "error": str(e),
                "error_stage": "parse",
            }


if __name__ == '__main__':
    pdf_path = PROJECT_ROOT / "report" / "000008_2024_神州高铁_2024年年度报告_2025-04-19.pdf"
    document_id = sha1_file(pdf_path)
    state = ImportState(
        document_id=document_id,
        pdf_path=str(pdf_path),
    )
    node_parse(state,None)