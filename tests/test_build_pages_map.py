"""pages.json 构建测试——真实根因：content_list 文本在 content.md 里 find 不到，
pages.json 只剩碎片 span，table_extract._page_of 落空 → report_tables.page_no 全 NULL。

第二层根因：page_number/'表' 等短文本误匹配，把 cursor 带飞或把后页起点钉进前页，
页码整体偏 1-2 页。锚点只用 ≥15 字长文本（见 parse.MIN_ANCHOR_LEN）。
"""
from __future__ import annotations

import json

from app.import_graph.nodes.parse import _build_pages_map
from app.import_graph.nodes.table_extract import _page_of
from app.import_graph.split.page_map import PageMap


def _write_content_list(raw_dir, items):
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "xxx_content_list.json").write_text(
        json.dumps(items, ensure_ascii=False), encoding="utf-8"
    )


def test_pages_map_covers_entire_content_without_gaps(tmp_path):
    """span 必须连续覆盖 [0, len(content))——碎片 span 是 page_no 全空的根因。"""
    content = (
        "第一节重要提示目录和释义完整正文\n\n"
        "## 第二节公司简介与主要财务指标\n\n第二节公司简介与主要财务指标正文内容\n\n"
        "## 第三节管理层讨论与分析\n\n第三节管理层讨论与分析完整正文内容"
    )
    raw = tmp_path / "raw"
    _write_content_list(
        raw,
        [
            {"type": "text", "text": "第一节重要提示目录和释义完整正文", "page_idx": 0},
            {"type": "text", "text": "第二节公司简介与主要财务指标正文内容", "page_idx": 1},
            {"type": "text", "text": "第三节管理层讨论与分析完整正文内容", "page_idx": 2},
        ],
    )
    out = tmp_path / "pages.json"
    spans = _build_pages_map(content, raw, out)

    # 连续覆盖：任意偏移都能命中某个 span
    for off in range(len(content)):
        assert any(s["char_start"] <= off <= s["char_end"] for s in spans), off
    assert PageMap(spans).page_at(0) == 1
    assert PageMap(spans).page_at(len(content) - 1) == 3


def test_pages_map_matches_markdown_rewritten_text(tmp_path):
    """content.md 的标题带 ## 前缀、空白被改写，content_list 原文 find 不到——要能定位。"""
    content = (
        "## 备查文件目录与公司基本情况一览表\n\n"
        "1、会计报表审计报告全文\n\n"
        "## 释义与重要名词解释完整详细说明\n\n<table><tr><td>释义项解释内容</td></table>"
    )
    raw = tmp_path / "raw"
    _write_content_list(
        raw,
        [
            # 不带 markdown 前缀的原文（MinerU content_list 就是这样）
            {"type": "text", "text": "备查文件目录与公司基本情况一览表", "page_idx": 0},
            {"type": "text", "text": "释义与重要名词解释完整详细说明", "page_idx": 1},
        ],
    )
    out = tmp_path / "pages.json"
    spans = _build_pages_map(content, raw, out)

    pm = PageMap(spans)
    assert pm.page_at(content.index("会计报表审计报告")) == 1
    assert pm.page_at(content.index("<table>")) == 2


def test_pages_map_matches_whitespace_rewritten_block(tmp_path):
    """多行文本在 content.md 里换行/空格被改写，精确 find 失配（真实日志 1908/2274）。"""
    content = (
        "第一节 重要提示、目录和释义....1\n第二节 公司简介....5\n\n"
        "第三节管理层讨论与分析完整正文内容"
    )
    raw = tmp_path / "raw"
    _write_content_list(
        raw,
        [
            {
                "type": "text",
                "text": "第一节 重要提示、目录和释义....1 第二节 公司简介....5",
                "page_idx": 0,
            },
            {"type": "text", "text": "第三节管理层讨论与分析完整正文内容", "page_idx": 1},
        ],
    )
    out = tmp_path / "pages.json"
    spans = _build_pages_map(content, raw, out)

    pm = PageMap(spans)
    assert pm.page_at(content.index("第二节")) == 1
    assert pm.page_at(content.index("第三节")) == 2


def test_pages_map_fills_gap_between_matched_texts(tmp_path):
    """只命中两页的文本时，中间的大段空白必须落在前一页而不是查不到。"""
    content = "第一节重要提示目录和释义完整正文" + "x" * 100 + "第三节管理层讨论与分析完整正文内容"
    raw = tmp_path / "raw"
    _write_content_list(
        raw,
        [
            {"type": "text", "text": "第一节重要提示目录和释义完整正文", "page_idx": 0},
            {"type": "text", "text": "这页文本在content的md里根本找不到所以没有锚点", "page_idx": 1},
            {"type": "text", "text": "第三节管理层讨论与分析完整正文内容", "page_idx": 2},
        ],
    )
    out = tmp_path / "pages.json"
    spans = _build_pages_map(content, raw, out)

    assert _page_of(spans, 50) == 1  # 间隙归上一页
    assert _page_of(spans, content.index("第三节")) == 3


def test_page_of_returns_none_only_when_no_map():
    assert _page_of([], 10) is None


def test_page_number_digit_must_not_consume_later_page_text(tmp_path):
    """page_number '1' 匹配到后文 '2021' 的数字，cursor 跳过第 2 页正文，该页锚点丢失、页码偏 1-2。

    真实年报 page_number 文本就是 '1'/'2'/'1 / 158'，属于最不可靠的锚点。
    """
    content = (
        "第一节重要提示目录和释义完整正文\n\n"
        "公司简介与2021年经营情况分析报告完整说明\n\n"
        "合并利润表主要项目数据明细完整列示"
    )
    raw = tmp_path / "raw"
    _write_content_list(
        raw,
        [
            {"type": "text", "text": "第一节重要提示目录和释义完整正文", "page_idx": 0},
            {"type": "page_number", "text": "1", "page_idx": 0},
            {"type": "text", "text": "公司简介与2021年经营情况分析报告完整说明", "page_idx": 1},
            {"type": "text", "text": "合并利润表主要项目数据明细完整列示", "page_idx": 2},
        ],
    )
    out = tmp_path / "pages.json"
    spans = _build_pages_map(content, raw, out)

    pm = PageMap(spans)
    assert pm.page_at(content.index("公司简介与2021年")) == 2
    assert pm.page_at(content.index("合并利润表")) == 3


def test_short_generic_text_must_not_anchor_pages(tmp_path):
    """短文本（<15 字）在正文里随处可中，会把后页起点钉进前页、页序压缩。

    「表」作为 page_idx=1 的锚点会命中第 1 页「资产负债表」里的表字，
    第 1 页后半截被错划给第 2 页。
    """
    content = (
        "合并资产负债表编制基础说明完整披露\n\n"
        "第二节公司简介与业务情况完整说明\n\n"
        "投资收益明细表数据完整列示内容"
    )
    raw = tmp_path / "raw"
    _write_content_list(
        raw,
        [
            {"type": "text", "text": "合并资产负债表编制基础说明完整披露", "page_idx": 0},
            {"type": "text", "text": "表", "page_idx": 1},  # 短文本，命中第 1 页「资产负债表」
            {"type": "text", "text": "第二节公司简介与业务情况完整说明", "page_idx": 1},
            {"type": "text", "text": "投资收益明细表数据完整列示内容", "page_idx": 2},
        ],
    )
    out = tmp_path / "pages.json"
    spans = _build_pages_map(content, raw, out)

    pm = PageMap(spans)
    # 第 1 页后半截（表之后）必须仍是第 1 页，不能被短锚点划走
    assert pm.page_at(content.index("编制基础说明")) == 1
    assert pm.page_at(content.index("第二节")) == 2
    assert pm.page_at(content.index("投资收益明细")) == 3


def test_image_item_without_text_is_ignored(tmp_path):
    """image 项 content 为空，不得参与锚定（连空串 find 都会把 cursor 钉在 0）。"""
    content = (
        "第一节重要提示目录和释义完整正文\n\n"
        "第二节公司简介主要内容完整说明"
    )
    raw = tmp_path / "raw"
    _write_content_list(
        raw,
        [
            {"type": "image", "img_path": "images/x", "content": "", "page_idx": 1},
            {"type": "text", "text": "第一节重要提示目录和释义完整正文", "page_idx": 0},
            {"type": "text", "text": "第二节公司简介主要内容完整说明", "page_idx": 1},
        ],
    )
    out = tmp_path / "pages.json"
    spans = _build_pages_map(content, raw, out)

    pm = PageMap(spans)
    assert pm.page_at(content.index("第二节")) == 2
