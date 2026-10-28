"""查询图 State（docs/04-查询图节点设计.md §3）。

状态默认值一律走 default_state() 工厂，禁止可变默认参数共享，防跨请求污染。
"""
from __future__ import annotations

from typing import Any, TypedDict


class Entities(TypedDict):
    stock_names: list[str]          # LLM 抽出的公司原词（含未解析的）
    stock_codes: list[str]          # stock_dict 解析后的 6 位代码，下游 filters 用
    unresolved_names: list[str]     # 四级匹配失败的原词 → clarify 追问
    periods: list[str]              # 已绝对化报告期，如 ["2022","2023","2024"]
    report_types: list[str]         # 年报/中报/季报等类型过滤
    metrics: list[str]              # 指标编码，如 ["gross_margin","roe"]


class SubQuery(TypedDict):
    sub_id: str                     # 子问题编号，evidence 对齐用
    question: str                   # 子问题原文（改写后）
    route: str                      # vector | metric_sql | calc | mixed
    filters: dict[str, Any]         # stock_code/period/sections 等检索过滤条件
    hyde_used: bool                 # 是否走了 HyDE 假设文档检索

class QueryState(TypedDict):
    request_id: str                 # 全链路追踪 id，日志/SSE 关联
    session_id: str                 # 多轮会话 id，Redis 历史 key 的一段
    query: str                      # 用户本轮原话，审计/Redis content 用
    history: list[dict]             # Redis 窗口快照（近 30 条），指代消解读
    intent: str                     # compare 对比 | metric_calc 单期指标取数| trend 多期趋势| simple_qa 非指标问答| out_of_scope 与股票投资无关
    intent_raw: dict[str, Any]      # node_intent 共享的 LLM JSON，entity 只读复用
    rewritten_query: str            # 指代补全后的完整问句，plan/generate 用
    entities: Entities              # 实体解析结果（04a §3）
    resolved_names: dict[str, str]  # name → code 明细，审计命中了哪级
    missing_fields: list[str]       # 缺关键字段名，非异常，走 clarify 追问
    sub_queries: list[SubQuery]     # plan 拆出的子问题 + 路由
    vector_hits: list[dict]         # dense 向量命中（chunk_id/content/page_no/score）
    sparse_hits: list[dict]         # sparse 稀疏命中，与 dense 同构
    hyde_hits: list[dict]           # HyDE 假设文档命中，与 dense 同构
    metric_rows: list[dict]         # 指标 SQL 查回的行（含 page_no，供引用）
    calc_results: list[dict]        # L3 计算结果（趋势/同比环比，含 formula）
    evidence: list[dict]            # fuse 统一证据块，rerank 后为最终上下文
    answer: str                     # generate 最终回答（含 [n] 引用标记）
    citations: list[dict]           # 引用清单（page_no/snippet/type），validate 校验
    error: str | None               # 人可读错误信息
    error_code: str | None          # 机器可读错误码，如 INTERNAL / DEPENDENCY_UNAVAILABLE
    error_stage: str | None         # 出错节点名，定位用


def default_entities() -> Entities:
    return {
        "stock_names": [],
        "stock_codes": [],
        "unresolved_names": [],
        "periods": [],
        "report_types": [],
        "metrics": [],
    }


def default_state(
    query: str = "",
    request_id: str = "",
    session_id: str = "",
    history: list[dict] | None = None,
) -> QueryState:
    return {
        "request_id": request_id,
        "session_id": session_id,
        "query": query,
        "history": history or [],
        "intent": "",
        "intent_raw": {},
        "rewritten_query": query,
        "entities": default_entities(),
        "resolved_names": {},
        "missing_fields": [],
        "sub_queries": [],
        "vector_hits": [],
        "sparse_hits": [],
        "hyde_hits": [],
        "metric_rows": [],
        "calc_results": [],
        "evidence": [],
        "answer": "",
        "citations": [],
        "error": None,
        "error_code": None,
        "error_stage": None,
    }
