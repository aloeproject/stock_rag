"""token 估算器测试——见 docs/02a §6：CJK≈1、ASCII 词≈1.3，确定性零依赖。"""
from math import ceil

from app.import_graph.split.token_estimator import estimate_token_count


def test_pure_cjk_counts_one_per_char():
    assert estimate_token_count("财务报告") == 4


def test_ascii_word_counts_1_3():
    # 1 个 ASCII 词 → ceil(1.3) = 2
    assert estimate_token_count("revenue") == 2


def test_mixed_cjk_and_ascii():
    # 2 个 CJK + 2 个 ASCII 词 → 2 + ceil(2.6) = 5
    assert estimate_token_count("营收 revenue cost") == 5


def test_fullwidth_punct_counts_as_cjk():
    # 收入增长 4 字 + ：。 2 个全角标点 = 6
    assert estimate_token_count("收入：增长。") == 6


def test_empty_text_is_zero():
    assert estimate_token_count("") == 0


def test_long_paragraph_is_stable():
    text = "本年度公司实现营业收入112,010.68万元，同比减少1.21%。"
    # 非空即有正 token，且重复调用结果一致（确定性）
    assert estimate_token_count(text) > 0
    assert estimate_token_count(text) == estimate_token_count(text)
