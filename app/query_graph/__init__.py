"""查询图（P3，docs/04-查询图节点设计.md）。

11 节点短事务 DAG：intent→entity→clarify?→plan→(retrieve/metric_query/calc)→
fuse→rerank→generate→validate。API 层经 build_query_graph().compile() 调用。
"""
