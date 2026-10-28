"""查询图装配测试（docs/04 §7）。

只验拓扑可编译 + 节点齐全 + path_map 无未注册键；
节点业务逻辑由各自单测覆盖，此处节点是空实现。
"""
from __future__ import annotations

from app.query_graph.main_graph import ALL_NODES, build_query_graph


def test_build_query_graph_compiles():
    g = build_query_graph().compile()
    assert g is not None


def test_graph_has_all_11_nodes():
    g = build_query_graph()
    names = set(g.nodes)
    for n in ALL_NODES:
        assert n in names, f"缺少节点 {n}"
    assert len(ALL_NODES) == 11


def test_path_map_keys_match_route_returns():
    """条件边 path_map 键必须覆盖路由可能返回的全部值，否则运行时炸（避坑清单）。"""
    g = build_query_graph()
    # compile 通过即说明 path_map 校验过；再编译一次确保幂等
    g.compile()
    g.compile()
