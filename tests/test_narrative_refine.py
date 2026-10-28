"""细化切分测试——docs/02a §3.2：超长段落递归切、过短同级合并。"""
from app.import_graph.split.config import SplitConfig
from app.import_graph.split.narrative_refine import (
    split_long_section,
    merge_short_sections,
    refine_sections,
)


def _section(content: str, title: str = "标题", parent: str = "父标题") -> dict:
    return {"title": title, "content": content, "parent_title": parent, "section_path": parent}


def test_short_section_passes_through():
    sec = _section("短内容")
    out = split_long_section(sec, SplitConfig())
    assert len(out) == 1
    assert out[0] is sec


def test_long_section_splits_with_paragraph_priority():
    # 3 个段落，每段 ~30 字 → 总 ~90 token；max=20 强制切
    para = "这是第一段内容，用于测试段落优先切分。" * 2
    content = f"{para}\n\n{para}\n\n{para}"
    sec = _section(content)
    cfg = SplitConfig(max_token=20, min_token=5, overlap_token=3)
    out = split_long_section(sec, cfg)
    assert len(out) > 1
    # 子块命名 parent_i，parent_title 保留
    assert out[0]["title"].startswith("标题_")
    assert out[0]["parent_title"] == "父标题"
    assert all(s["part"] == i for i, s in enumerate(out, start=1))


def test_long_section_respects_overlap():
    para_a = "甲" * 30
    para_b = "乙" * 30
    sec = _section(f"{para_a}\n\n{para_b}")
    cfg = SplitConfig(max_token=25, min_token=5, overlap_token=10)
    out = split_long_section(sec, cfg)
    # overlap 使相邻块有重叠文本
    assert any(a["content"][-5:] in b["content"] for a, b in zip(out, out[1:]))


def test_merge_short_same_parent():
    cfg = SplitConfig(max_token=50, min_token=20, overlap_token=5)
    s1 = _section("短A", title="标题_1")
    s2 = _section("短B", title="标题_2")
    out = merge_short_sections([s1, s2], cfg)
    assert len(out) == 1
    assert "短A" in out[0]["content"]
    assert "短B" in out[0]["content"]


def test_merge_rejects_different_parent():
    cfg = SplitConfig(max_token=50, min_token=20, overlap_token=5)
    s1 = _section("短A", title="标题_1", parent="父1")
    s2 = _section("短B", title="标题_2", parent="父2")
    out = merge_short_sections([s1, s2], cfg)
    assert len(out) == 2


def test_merge_respects_hard_max():
    cfg = SplitConfig(max_token=50, min_token=30, overlap_token=5, hard_max_token=40)
    s1 = _section("甲" * 20, title="标题_1")  # 20 token < min(30)
    s2 = _section("乙" * 20, title="标题_2")  # 20+20=40 == hard_max，允许合并
    out = merge_short_sections([s1, s2], cfg)
    assert len(out) == 1
    # 超过硬上限则不并（20+25=45 > 40）
    out2 = merge_short_sections(
        [_section("甲" * 20, title="标题_1"), _section("丙" * 25, title="标题_3")],
        cfg,
    )
    assert len(out2) == 2


def test_refine_sections_end_to_end():
    long = "段落一。" * 40 + "\n\n" + "段落二。" * 40
    secs = [_section(long), _section("短尾", title="标题_2")]
    cfg = SplitConfig(max_token=50, min_token=15, overlap_token=5)
    out = refine_sections(secs, cfg)
    assert all(len(s["content"]) > 0 for s in out)
    assert all("part" in s for s in out)


def test_refine_strips_table_placeholder_whitespace():
    """strip_tables 留下的等长空白垫要在细化前清掉，否则污染合并与可读性。"""
    content = "标题\n正文一句。\n" + " " * 200
    sec = _section(content)
    out = refine_sections([sec], SplitConfig())
    assert "  " not in out[0]["content"]
    assert out[0]["content"].endswith("正文一句。")


def test_refine_sets_parent_title_before_merge():
    """合并前就要有真实父标题——否则 setdefault 写成自身，同级短块永远并不上。"""
    secs = [
        _section("短A", title="（1）营业收入构成", parent=""),
        _section("短B", title="（2）收入情况", parent=""),
    ]
    for s in secs:
        s["section_path"] = "2、收入与成本/" + s["title"].lstrip("# ")
        del s["parent_title"]
    cfg = SplitConfig(max_token=50, min_token=20, overlap_token=5)
    out = refine_sections(secs, cfg)
    assert len(out) == 1
    assert out[0]["parent_title"] == "2、收入与成本"
    assert "短A" in out[0]["content"] and "短B" in out[0]["content"]


def test_merge_absorbs_short_tail_into_long_prev():
    """长块后面的短尾巴也要能并进去，不能只让「短+短」合并。"""
    cfg = SplitConfig(max_token=50, min_token=20, overlap_token=5, hard_max_token=40)
    long = _section("长" * 30, title="标题_1")  # 30 token ≥ min
    tail = _section("尾巴", title="标题_2")  # 2 token < min
    out = merge_short_sections([long, tail], cfg)
    assert len(out) == 1
    assert "尾巴" in out[0]["content"]
