"""node_metadata 测试：元数据解析 + stock_dict 字典写入（04 §4.1）。

假 PDF 文件名 / content.md 注入，断言字段解析与 stock_dict upsert。
不碰真实 MinerU / LLM。
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.entities.stock_dict import StockDictDbData
from app.import_graph.nodes.metadata import node_metadata
from app.models.base import Base
from app.repositories.stock_dict_repository import StockDictRepository


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    yield s
    s.close()


@pytest.fixture()
def stock_repo(session):
    return StockDictRepository(session)


def _run(state, runtime, writer=None):
    # node_metadata 是同步节点（不起异步 IO），直接调用
    return node_metadata(state, runtime, writer)


def _state(pdf_name: str, markdown_path: str = ""):
    return {
        "pdf_path": f"/fake/report/{pdf_name}",
        "markdown_path": markdown_path,
        "document_id": "doc1",
    }


# ---------- 字段解析基线 ----------


def test_parse_filename_success(runtime):
    out = _run(
        _state("601225_2024_陕西煤业_2024年年度报告_2025-04-26.pdf"),
        runtime,
    )
    assert out["stock_code"] == "601225"
    assert out["stock_name"] == "陕西煤业"
    assert out["report_type"] == "annual"
    assert out["report_period"] == "2024"
    assert out["error"] is None


def test_missing_stock_code_fails(runtime):
    out = _run(_state("badname.pdf"), runtime)
    assert out["parse_status"] == "failed"
    assert out["error_stage"] == "metadata"


# ---------- stock_dict 写入 ----------


def test_stock_dict_upserted_on_success(runtime, stock_repo, session):
    """解析成功 → stock_dict 落库（04 §4.1 导入种子主路径）。"""
    runtime.context.stock_dict_repository = stock_repo
    _run(_state("601225_2024_陕西煤业_2024年年度报告_2025-04-26.pdf"), runtime)

    rows = stock_repo.list_all()
    assert len(rows) == 1
    assert rows[0].stock_code == "601225"
    assert rows[0].stock_name == "陕西煤业"
    assert rows[0].status == 1


def test_market_derived_from_code(runtime, stock_repo):
    """市场前缀：6→SH，0/3→SZ，4/8→BJ。"""
    runtime.context.stock_dict_repository = stock_repo
    cases = [
        ("601225_2024_陕西煤业_2024年年度报告_2025-04-26.pdf", "601225", "SH"),
        ("000008_2024_神州高铁_2024年年度报告_2025-04-19.pdf", "000008", "SZ"),
        ("300750_2024_宁德时代_2024年年度报告_2025-04-20.pdf", "300750", "SZ"),
        ("430047_2024_诺思兰德_2024年年度报告_2025-04-20.pdf", "430047", "BJ"),
        ("832000_2024_安徽凤凰_2024年年度报告_2025-04-20.pdf", "832000", "BJ"),
    ]
    for pdf_name, code, market in cases:
        _run(_state(pdf_name), runtime)
        row = next(r for r in stock_repo.list_all() if r.stock_code == code)
        assert row.market == market, f"{code} expected {market}, got {row.market}"


def test_upsert_idempotent_updates_name(runtime, stock_repo):
    """同 code 再导入 → 不新增行，只覆盖 stock_name（04 §4.1 ON DUPLICATE KEY UPDATE）。"""
    runtime.context.stock_dict_repository = stock_repo
    stock_repo.upsert(StockDictDbData(stock_code="601225", stock_name="旧名", market="SH"))
    _run(_state("601225_2024_陕西煤业_2024年年度报告_2025-04-26.pdf"), runtime)

    rows = stock_repo.list_all()
    assert len(rows) == 1
    assert rows[0].stock_name == "陕西煤业"


def test_missing_name_skips_upsert(runtime, stock_repo, tmp_path):
    """有 code 无 name → 不写脏行（空名进字典会污染四级匹配）。

    严格文件名要求 stock_name ≥1 字符，缺名场景只能走 content.md 兜底：
    只给「公司代码」不给「公司简称」。
    """
    md = tmp_path / "content.md"
    md.write_text("公司代码：601225\n2024年年度报告", encoding="utf-8")
    runtime.context.stock_dict_repository = stock_repo
    out = _run(_state("unknown.pdf", markdown_path=str(md)), runtime)
    assert out.get("stock_code") == "601225"
    assert out.get("stock_name") in ("", None)
    assert stock_repo.list_all() == []


def test_no_repo_does_not_crash(runtime):
    """未注入 stock_dict_repository（旧测试）→ 跳过写入，不炸。"""
    out = _run(_state("601225_2024_陕西煤业_2024年年度报告_2025-04-26.pdf"), runtime)
    assert out["error"] is None


def test_fail_path_does_not_upsert(runtime, stock_repo):
    """metadata 失败 → 不写 stock_dict。"""
    runtime.context.stock_dict_repository = stock_repo
    _run(_state("badname.pdf"), runtime)
    assert stock_repo.list_all() == []


def test_content_fallback_also_upserts(runtime, stock_repo, tmp_path):
    """content.md 兜底解析出的 code/name 同样进字典。"""
    md = tmp_path / "content.md"
    md.write_text("公司代码：601225\n公司简称：陕西煤业\n2024年年度报告", encoding="utf-8")
    runtime.context.stock_dict_repository = stock_repo
    out = _run(_state("unknown.pdf", markdown_path=str(md)), runtime)
    assert out["stock_code"] == "601225"
    rows = stock_repo.list_all()
    assert len(rows) == 1
    assert rows[0].stock_name == "陕西煤业"
