"""页码映射测试——pages.json 的 char 区间 → page_no，见 docs/02a §3.3。"""
from app.import_graph.split.page_map import PageMap


def test_page_at_start_of_range():
    pm = PageMap([{"char_start": 0, "char_end": 10, "page_no": 1},
                  {"char_start": 12, "char_end": 20, "page_no": 2}])
    assert pm.page_at(0) == 1
    assert pm.page_at(10) == 1


def test_page_at_second_range():
    pm = PageMap([{"char_start": 0, "char_end": 10, "page_no": 1},
                  {"char_start": 12, "char_end": 20, "page_no": 2}])
    assert pm.page_at(12) == 2
    assert pm.page_at(20) == 2


def test_gap_falls_back_to_previous_page():
    # pages.json 覆盖率约 96%，有间隙；间隙应落到上一页而不是 0
    pm = PageMap([{"char_start": 0, "char_end": 10, "page_no": 1},
                  {"char_start": 15, "char_end": 20, "page_no": 2}])
    assert pm.page_at(13) == 1


def test_before_first_range_returns_zero():
    pm = PageMap([{"char_start": 10, "char_end": 20, "page_no": 3}])
    assert pm.page_at(5) == 0


def test_span_returns_start_and_end_page():
    pm = PageMap([{"char_start": 0, "char_end": 10, "page_no": 1},
                  {"char_start": 11, "char_end": 30, "page_no": 2}])
    start, end = pm.page_span(5, 25)
    assert start == 1
    assert end == 2
