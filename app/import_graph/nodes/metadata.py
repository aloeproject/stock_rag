from __future__ import annotations

import re
import sys
from pathlib import Path

from langgraph.runtime import Runtime

from app.import_graph.context import DataAgentContext
from app.utils.file import sha1_file

# 支持直接运行本文件（python app/import_graph/nodes/metadata.py）时解析到项目根
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from langgraph.types import StreamWriter

from app.core.logger import logger
from app.entities.stock_dict import StockDictDbData
from app.utils.path_util import PROJECT_ROOT
from app.import_graph.nodes.progress import progress_step
from app.import_graph.state import ImportState

# ---------------------------------------------------------------------------
# S2 元数据提取规则（见 docs/02-数据构建.md）
#
# 来源优先级：
#   ① 文件名（约定：{stock_code}_{fiscal_year}_{stock_name}_{报告类型描述}_{publish_date}.pdf）
#   ② content.md 首页/封面文本（公司代码/公司简称/标题中的报告类型）
#
# 失败条件（任一缺失即拒绝入库）：
#   stock_code 或 report_period 无法确定 → error_stage="metadata"
# ---------------------------------------------------------------------------

# 文件名完整模式，例：
#   000006_2024_深振业A_2024年年度报告_2025-04-26.pdf
#   600628_2024_新世界_新世界2024年年度报告_2025-04-12.pdf
FILENAME_RE = re.compile(
    r"^(?P<stock_code>\d{6})_"
    r"(?P<fiscal_year>\d{4})_"
    r"(?P<stock_name>.+?)_"
    r"(?P<type_desc>.+)_"
    r"(?P<publish_date>\d{4}-\d{2}-\d{2})$"
)

# 宽松模式：允许缺少发布日期，至少保证 代码_年份_... 结构
FILENAME_LOOSE_RE = re.compile(
    r"^(?P<stock_code>\d{6})_(?P<fiscal_year>\d{4})_(?P<rest>.+)$"
)

# content.md 封面字段
CONTENT_CODE_RE = re.compile(r"公司代码\s*[：:]\s*(\d{6})")
CONTENT_NAME_RE = re.compile(r"公司简称\s*[：:]\s*([^\s，,。；;]+)")
CONTENT_YEAR_RE = re.compile(r"(\d{4})\s*年")

# 中文季度数字 → 阿拉伯数字
CN_QUARTER_MAP = {"一": "1", "二": "2", "三": "3", "四": "4"}


def _derive_market(stock_code: str) -> str:
    """A 股代码首位定市场：6→SH，0/3→SZ，4/8→BJ。"""
    if not stock_code:
        return ""
    head = stock_code[0]
    if head == "6":
        return "SH"
    if head in ("0", "3"):
        return "SZ"
    if head in ("4", "8"):
        return "BJ"
    return ""


def _parse_report_type(text: str, fiscal_year: int) -> tuple[str, str]:
    """从文本识别报告类型，并拼装 report_period。

    report_type 枚举（与 report_documents 表一致）：
      annual / q1 / q2 / q3 / interim / prospectus

    report_period 拼装规则：
      annual   → "2024"
      q1~q3    → "2024Q1" 等
      interim  → "2024H1"（半年度，避免与 annual 的 "2024" 混淆）

    返回 (report_type, report_period)；无法识别时两者均为空串。
    """
    # 注意：必须先判断「半年度报告」，因为「年度报告」是其子串
    if "半年度报告" in text or "半年报" in text:
        return "interim", f"{fiscal_year}H1"

    if "招股说明书" in text:
        return "prospectus", str(fiscal_year)

    if "年度报告" in text or "年报" in text:
        return "annual", str(fiscal_year)

    # 季度报告：第一季度报告 / 一季度报告 / 第1季度报告 / 一季报 / 2024Q1
    q_match = re.search(
        r"第?\s*([一二三四1-4])\s*季度|第?\s*([一二三四1-4])\s*季报|[Qq]([1-4])",
        text,
    )
    if q_match:
        raw_q = q_match.group(1) or q_match.group(2) or q_match.group(3)
        q_no = CN_QUARTER_MAP.get(raw_q, raw_q)
        return f"q{q_no}", f"{fiscal_year}Q{q_no}"

    # 仅出现「季度报告」但没有季度号时，无法拼出具体期间
    return "", ""


def _parse_filename(pdf_path: str) -> dict:
    """按文件名约定提取元数据，返回已解析字段的 dict（可能不全）。

    优先严格模式（含发布日期），失败再用宽松模式；
    文件名中的报告类型描述单独用 _parse_report_type 判定。
    """
    stem = Path(pdf_path).stem
    result: dict = {"source_file": Path(pdf_path).name}

    m = FILENAME_RE.match(stem)
    if m:
        result["stock_code"] = m.group("stock_code")
        result["fiscal_year"] = int(m.group("fiscal_year"))
        result["stock_name"] = m.group("stock_name")
        report_type, report_period = _parse_report_type(m.group("type_desc"), result["fiscal_year"])
        if report_type:
            result["report_type"] = report_type
            result["report_period"] = report_period
        return result

    m = FILENAME_LOOSE_RE.match(stem)
    if m:
        result["stock_code"] = m.group("stock_code")
        result["fiscal_year"] = int(m.group("fiscal_year"))
        rest = m.group("rest")
        # rest 形如「深振业A_2024年年度报告」或「深振业A」
        parts = rest.split("_")
        if parts:
            result["stock_name"] = parts[0]
        report_type, report_period = _parse_report_type(rest, result["fiscal_year"])
        if report_type:
            result["report_type"] = report_type
            result["report_period"] = report_period

    return result


def _parse_content_head(markdown_path: str) -> dict:
    """从 content.md 首页文本兜底提取元数据（封面：公司代码/简称/报告标题）。"""
    result: dict = {}
    if not markdown_path:
        return result

    try:
        # 首页/封面信息足够，取前 2KB 避免整篇年报拖慢解析
        content = Path(markdown_path).read_text(encoding="utf-8", errors="ignore")[:2048]
    except OSError as e:
        logger.warning(f">>> [node_metadata] 读取 content.md 失败，跳过内容兜底: {e}")
        return result

    code_m = CONTENT_CODE_RE.search(content)
    if code_m:
        result["stock_code"] = code_m.group(1)

    name_m = CONTENT_NAME_RE.search(content)
    if name_m:
        result["stock_name"] = name_m.group(1)

    year_m = CONTENT_YEAR_RE.search(content)
    fiscal_year = int(year_m.group(1)) if year_m else 0

    report_type, report_period = _parse_report_type(content, fiscal_year) if fiscal_year else ("", "")
    if fiscal_year:
        result["fiscal_year"] = fiscal_year
    if report_type:
        result["report_type"] = report_type
        result["report_period"] = report_period

    return result


def node_metadata(state: ImportState,runtime: Runtime[DataAgentContext], writer: StreamWriter) -> dict:
    """S2: 提取股票代码/简称/报告类型/报告期。

    流程：
      1. 优先解析 PDF 文件名（约定：代码_年份_简称_报告类型描述_发布日期.pdf）
      2. 文件名缺失字段再用 content.md 封面文本兜底
      3. 校验 stock_code / report_period 必填，缺失则短路至 commit_status

    成功：返回 stock_code/stock_name/report_type/report_period/fiscal_year/source_file
    失败：error_stage="metadata" + parse_status="failed"，由 route_after_metadata 短路。
    """
    logger.info(f"[node_metadata] 开始处理文档: {state.get('document_id')}")
    with progress_step(writer, "元数据抽取") as outcome:
        pdf_path = state.get("pdf_path") or ""
        markdown_path = state.get("markdown_path") or ""

        try:
            # ① 文件名优先（设计约定，样例文件名已满足校验建议）
            from_filename = _parse_filename(pdf_path)
            # ② 封面文本兜底，仅填补文件名未解析出的字段
            from_content = _parse_content_head(markdown_path)

            merged: dict = {}
            merged.update(from_content)
            merged.update({k: v for k, v in from_filename.items() if v not in ("", None)})

            stock_code = str(merged.get("stock_code") or "").strip()
            stock_name = str(merged.get("stock_name") or "").strip()
            report_type = str(merged.get("report_type") or "").strip()
            report_period = str(merged.get("report_period") or "").strip()
            fiscal_year = int(merged.get("fiscal_year") or 0)
            source_file = str(merged.get("source_file") or Path(pdf_path).name)

            # 必填校验：缺代码或期间 → METADATA_INVALID，不进后续阶段
            missing = []
            if not stock_code:
                missing.append("stock_code")
            if not report_period:
                missing.append("report_period")
            if missing:
                error_msg = f"METADATA_INVALID: 缺少必填字段 {missing}, pdf={source_file}"
                logger.error(f">>> [node_metadata] {error_msg}")
                outcome["ok"] = False
                return {
                    "parse_status": "failed",
                    "error": error_msg,
                    "error_stage": "metadata",
                }

            logger.info(
                f">>> [node_metadata] 提取成功: code={stock_code}, name={stock_name}, "
                f"type={report_type}, period={report_period}, year={fiscal_year}"
            )

            # 导入种子写入 stock_dict（04 §4.1）：名称→代码真相源，查询侧四级匹配依赖它。
            # 仅在 code+name 齐全时 upsert——空名进字典会污染四级匹配。
            # repo 未注入（旧测试/未装配）时跳过，不炸。
            stock_repo = getattr(runtime.context, "stock_dict_repository", None) if runtime else None
            if stock_repo is not None and stock_code and stock_name:
                stock_repo.upsert(
                    StockDictDbData(
                        stock_code=stock_code,
                        stock_name=stock_name,
                        market=_derive_market(stock_code),
                        status=1,
                    )
                )
                logger.info(
                    f">>> [node_metadata] stock_dict upserted: {stock_code} {stock_name}"
                )

            return {
                "stock_code": stock_code,
                "stock_name": stock_name,
                "report_type": report_type,
                "report_period": report_period,
                "fiscal_year": fiscal_year,
                "source_file": source_file,
                "error": None,
                "error_stage": None,
            }
        except Exception as e:
            logger.error(f">>> [node_metadata] 元数据提取失败: {e}")
            outcome["ok"] = False
            return {
                "parse_status": "failed",
                "error": f"METADATA_INVALID: {e}",
                "error_stage": "metadata",
            }


if __name__ == "__main__":
    # 本地冒烟：用 report/ 下样例 PDF + 已生成的 parse 产物验证字段
    pdf_path = PROJECT_ROOT / "report" / "000008_2024_神州高铁_2024年年度报告_2025-04-19.pdf"
    markdown_path = (
        PROJECT_ROOT / "data" / "parse" / "013d1633320c62a34fd359a6fa5bb27f1e765355" / "content.md"
    )
    state: ImportState = {
        "pdf_path": str(pdf_path),
        "markdown_path": str(markdown_path),
    }  # type: ignore[typeddict-item]
    print(node_metadata(state, None))
