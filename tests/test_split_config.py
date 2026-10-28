"""切分参数测试——docs/02a §2：500/150/75/550。"""
from app.import_graph.split.config import SplitConfig


def test_default_params_match_design():
    cfg = SplitConfig()
    assert cfg.max_token == 500
    assert cfg.min_token == 150
    assert cfg.overlap_token == 75
    assert cfg.hard_max_token == 550


def test_overlap_is_15_percent_of_max():
    cfg = SplitConfig()
    assert cfg.overlap_token == int(cfg.max_token * 0.15)
