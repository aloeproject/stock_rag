"""递归文本切分测试——docs/02a §3.2：段落优先切口，相邻块带 overlap。"""
from app.import_graph.split.text_split import split_text_recursive


def test_paragraph_split_applies_overlap():
    """段落一次切开时也要 overlap——否则续块边界是硬切口，检索接不上。"""
    para_a = "甲" * 40
    para_b = "乙" * 40
    para_c = "丙" * 40
    text = f"{para_a}\n\n{para_b}\n\n{para_c}"
    pieces = split_text_recursive(text, max_chars=50, overlap_chars=10)
    assert len(pieces) == 3
    # 相邻块必须共享上一块的尾巴
    for a, b in zip(pieces, pieces[1:]):
        tail = a[-10:]
        assert tail in b, f"缺少 overlap: tail={tail!r} not in {b!r}"


def test_hard_split_applies_overlap():
    """无分隔符硬切按 step=max-overlap 前进，天然带重叠。"""
    text = "x" * 200
    pieces = split_text_recursive(text, max_chars=50, overlap_chars=10)
    assert len(pieces) >= 2
    for a, b in zip(pieces, pieces[1:]):
        assert a[-10:] in b


def test_single_chunk_no_overlap_needed():
    """整段不超限就单块，无所谓 overlap。"""
    pieces = split_text_recursive("短内容", max_chars=100, overlap_chars=10)
    assert pieces == ["短内容"]