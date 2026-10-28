"""表格 section_path 继承——表在章节中间时应挂当前章节，不是上一章。"""


from app.import_graph.split.dual_track import _nearest_section


def test_table_inside_section_uses_that_section():
    narrative = [
        {"section_path": "A章", "char_start": 0, "char_end": 50},
        {"section_path": "B章/一、子节", "char_start": 51, "char_end": 200},
        {"section_path": "C章", "char_start": 201, "char_end": 300},
    ]
    # 表落在 B 章区间内（之前误判成 A 章）
    assert _nearest_section(narrative, 120) == "B章/一、子节"


def test_table_between_sections_uses_previous():
    narrative = [
        {"section_path": "A章", "char_start": 0, "char_end": 50},
        {"section_path": "B章", "char_start": 60, "char_end": 100},
    ]
    assert _nearest_section(narrative, 55) == "A章"
