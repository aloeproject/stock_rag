"""Q5 node_retrieve 分支 A（docs/04e）。

Qdrant 混合检索：dense+sparse 必开，HyDE 按条件开；
filter 先过滤后向量；按 (stock_code, report_period) fan-out。
为什么 retrieve 不做 RRF：融合/截断/降权都在 fuse（04e §1.3），先融合会让权重失效。
为什么 dense+sparse 必开：术语词面 + 语义各有盲区（04e §3.3）。
为什么 HyDE 放在首轮检索之后：触发条件 3 依赖首轮 top1 得分（04e §3.4）。
为什么单路失败不断流：fuse 能用剩余路凑证据，只有连接级失败才 DEPENDENCY（04e §8）。
"""
from __future__ import annotations

from langgraph.runtime import Runtime
from langgraph.types import StreamWriter

from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.query_graph.context import QueryAgentContext
from app.query_graph.matching.retrieve_rules import (
    build_payload_filter,
    fanout_units,
    should_use_hyde,
)
from app.query_graph.state import QueryState

VECTOR_ROUTES = {"vector", "mixed"}
TOP_K_PER_ROUTE = 20
HYDE_MAX_RETRIES = 1
HYDE_MAX_CHARS = 200


def step_1_vector_sub_queries(state: QueryState) -> list[dict]:
    """只取 route ∈ {vector, mixed} 的子问题；metric_sql/calc 归 metric_query/calc。"""
    return [sq for sq in (state.get("sub_queries") or []) if sq.get("route") in VECTOR_ROUTES]


def step_3_search_hits(
    qdrant, question: str, units: list[tuple[str, str | None]], base_filters: dict, sub_id: str
) -> tuple[list[dict], list[dict], list[float]]:
    """dense+sparse 必开；单路失败 WARN 继续另路。返回 (vector_hits, sparse_hits, top_scores)。"""
    embs = generate_embeddings([question])
    dense_vec = embs["dense"][0]
    sparse_vec = embs["sparse"][0]

    vector_hits: list[dict] = []
    sparse_hits: list[dict] = []
    top_scores: list[float] = []

    for code, period in units:
        unit_f = dict(base_filters)
        unit_f["stock_codes"] = [code]
        unit_f["report_periods"] = [period] if period else []
        qfilter = build_payload_filter(unit_f)

        try:
            hits = qdrant.search_dense(dense_vec, qfilter, TOP_K_PER_ROUTE)
            vector_hits.extend(_tag(h, "dense", sub_id) for h in hits)
            if hits:
                top_scores.append(float(hits[0].get("score") or 0.0))
        except ConnectionError:
            raise  # 连接级失败不是单路降级，上抛 → DEPENDENCY_UNAVAILABLE
        except Exception as e:
            logger.warning(f"[node_retrieve] dense 检索失败，继续 sparse: {e}")

        try:
            hits = qdrant.search_sparse(sparse_vec, qfilter, TOP_K_PER_ROUTE)
            sparse_hits.extend(_tag(h, "sparse", sub_id) for h in hits)
            if hits:
                top_scores.append(float(hits[0].get("score") or 0.0))
        except ConnectionError:
            raise
        except Exception as e:
            logger.warning(f"[node_retrieve] sparse 检索失败，继续: {e}")

    return vector_hits, sparse_hits, top_scores


def step_4_hyde_hits(
    qdrant, question: str, units: list[tuple[str, str | None]], base_filters: dict, sub_id: str, llm
) -> list[dict] | None:
    """生成假设文档再走 dense。返回 None=未执行（LLM 失败跳过）；[]=执行了但没命中。

    hyde_used 语义是「HyDE 路跑过」不是「有命中」——空结果也该让 fuse 知道补过路。
    """
    hypothesis = _generate_hypothesis(llm, question)
    if not hypothesis:
        return None
    embs = generate_embeddings([hypothesis])
    hyde_vec = embs["dense"][0]
    out: list[dict] = []
    for code, period in units:
        unit_f = dict(base_filters)
        unit_f["stock_codes"] = [code]
        unit_f["report_periods"] = [period] if period else []
        qfilter = build_payload_filter(unit_f)
        try:
            hits = qdrant.search_dense(hyde_vec, qfilter, TOP_K_PER_ROUTE)
            out.extend(_tag(h, "hyde", sub_id) for h in hits)
        except Exception as e:
            logger.warning(f"[node_retrieve] hyde 检索失败，跳过: {e}")
    return out


def _tag(hit: dict, source: str, sub_id: str) -> dict:
    h = dict(hit)
    h["source"] = source
    h["sub_id"] = sub_id
    return h


def _generate_hypothesis(llm, question: str) -> str | None:
    """假设文档：禁止具体数字（prompt 硬约束）；超时/空文本重试 1 次。"""
    if llm is None:
        return None
    prompt = load_prompt("hyde", question=question or "")
    for _ in range(HYDE_MAX_RETRIES + 1):
        try:
            resp = llm.invoke(prompt)
            text = (getattr(resp, "content", None) or "").strip()
            if text:
                return text[:HYDE_MAX_CHARS]
        except Exception as e:
            logger.warning(f"[node_retrieve] HyDE LLM 调用失败，重试/跳过: {e}")
    return None


async def node_retrieve(
    state: QueryState,
    runtime: Runtime[QueryAgentContext],
    writer: StreamWriter = None,
) -> dict:
    sub_queries = step_1_vector_sub_queries(state)
    if not sub_queries:
        logger.warning(f"[node_retrieve] request_id={state.get('request_id')} 无 vector/mixed 子问题")
        return {
            "vector_hits": [],
            "sparse_hits": [],
            "hyde_hits": [],
            "error": None,
            "error_code": None,
            "error_stage": None,
        }

    qdrant = runtime.context.qdrant
    if qdrant is None:
        return {
            "vector_hits": [],
            "sparse_hits": [],
            "hyde_hits": [],
            "error": "Qdrant 不可用",
            "error_code": "DEPENDENCY_UNAVAILABLE",
            "error_stage": "retrieve",
        }

    llm = runtime.context.llm
    vector_hits: list[dict] = []
    sparse_hits: list[dict] = []
    hyde_hits: list[dict] = []
    updated: list[dict] = []
    dep_down = False

    for sq in sub_queries:
        question = sq.get("question") or ""
        filters = sq.get("filters") or {}
        sub_id = sq.get("sub_id") or ""
        stock_codes = list(filters.get("stock_codes") or [])
        if not stock_codes:
            logger.error(f"[node_retrieve] sub_id={sub_id} stock_codes 为空")
            return {
                "vector_hits": [],
                "sparse_hits": [],
                "hyde_hits": [],
                "error": "stock_codes 为空",
                "error_code": "INTERNAL",
                "error_stage": "retrieve",
            }

        units = fanout_units(stock_codes, list(filters.get("report_periods") or []))
        logger.info(
            f"[node_retrieve] sub_id={sub_id} question={question[:40]!r} "
            f"units={units} filters={filters}"
        )

        try:
            v_hits, s_hits, top_scores = step_3_search_hits(qdrant, question, units, filters, sub_id)
            # sections filter 命中 0 时降级重搜：plan LLM 标的 section 名可能与
            # sectionize 归类值对不上（如 management_discussion vs md_narrative），
            # 硬过滤会把全部 chunk 滤光（好当家 600467 实测）
            if not (v_hits or s_hits) and (filters.get("sections") or []):
                logger.info(f"[node_retrieve] sub_id={sub_id} sections 过滤空，降级去掉 sections 重搜")
                relaxed = {**filters, "sections": []}
                v_hits, s_hits, top_scores = step_3_search_hits(qdrant, question, units, relaxed, sub_id)
        except ConnectionError as e:
            logger.error(f"[node_retrieve] Qdrant 连接失败: {e}")
            return {
                "vector_hits": [],
                "sparse_hits": [],
                "hyde_hits": [],
                "error": str(e),
                "error_code": "DEPENDENCY_UNAVAILABLE",
                "error_stage": "retrieve",
            }
        except Exception as e:
            # embedding/API 级失败：本条子问题跳过，其余继续
            logger.warning(f"[node_retrieve] sub_id={sub_id} 检索失败，跳过: {e}")
            dep_down = dep_down or isinstance(e, ConnectionError)
            updated.append({**sq, "hyde_used": False})
            continue

        vector_hits.extend(v_hits)
        sparse_hits.extend(s_hits)

        top1 = max(top_scores) if top_scores else None
        use_hyde = should_use_hyde(
            sq.get("route") or "",
            list(filters.get("sections") or []),
            question,
            top1,
        )
        this_hyde: list[dict] | None = None
        if use_hyde:
            this_hyde = step_4_hyde_hits(qdrant, question, units, filters, sub_id, llm)
            if this_hyde:
                hyde_hits.extend(this_hyde)

        updated.append({**sq, "hyde_used": this_hyde is not None})

    if dep_down and not (vector_hits or sparse_hits or hyde_hits):
        return {
            "vector_hits": [],
            "sparse_hits": [],
            "hyde_hits": [],
            "error": "Qdrant 不可用",
            "error_code": "DEPENDENCY_UNAVAILABLE",
            "error_stage": "retrieve",
        }

    if writer is not None:
        for sq in updated:
            writer(
                {
                    "type": "sub_query",
                    "sub_id": sq["sub_id"],
                    "status": "done",
                    "hyde_used": sq["hyde_used"],
                }
            )

    logger.info(
        f"[node_retrieve] request_id={state.get('request_id')} "
        f"vector={len(vector_hits)} sparse={len(sparse_hits)} hyde={len(hyde_hits)}"
    )
    return {
        "vector_hits": vector_hits,
        "sparse_hits": sparse_hits,
        "hyde_hits": hyde_hits,
        "sub_queries": updated,
        "error": None,
        "error_code": None,
        "error_stage": None,
    }


if __name__ == "__main__":
    # 真实链路冒烟：真实 embedding API + 真实 Qdrant report_chunks + 真实 HyDE LLM。
    # 前置：embed_store 已把年报写进 Qdrant（000008 神州高铁）；.env 有 EMBEDDING/LLM key。
    # 覆盖 3 条典型子问题：风险题（条件1触发 HyDE）/ 短句无指标（条件2）/ 含指标长句（不触发）。
    import asyncio

    from langgraph.runtime import Runtime

    from app.clients.qdrant_client import QdrantClientManager
    from app.lm.llm_utils import get_llm_client
    from app.query_graph.context import QueryAgentContext
    from app.query_graph.state import default_state

    mgr = QdrantClientManager()
    mgr.init()
    print(f"== Qdrant {mgr.build_url(mgr.config)} collection={mgr.config.collection} ==")

    cases = [
        {
            "sub_id": "sq1",
            "question": "神州高铁2024年亏损的原因是什么？",
            "route": "vector",
            "filters": {
                "stock_codes": ["000008"],
                "report_periods": ["2024"],
                "report_types": ["annual"],
                "sections": ["management_discussion"],
            },
        },
        {
            "sub_id": "sq2",
            "question": "公司面临哪些风险？",
            "route": "vector",
            "filters": {
                "stock_codes": ["000008"],
                "report_periods": ["2024"],
                "report_types": ["annual"],
                "sections": ["risk"],
            },
        },
        {
            "sub_id": "sq3",
            "question": "毛利率下降的经营原因是什么情况呢",
            "route": "vector",
            "filters": {
                "stock_codes": ["000008"],
                "report_periods": ["2024"],
                "report_types": ["annual"],
                "sections": ["notes"],
            },
        },
    ]

    async def _main() -> None:
        runtime = Runtime(  # type: ignore[arg-type]
            context=QueryAgentContext(qdrant=mgr, llm=get_llm_client())
        )
        state = default_state(
            query="神州高铁2024年亏损的原因是什么？",
            request_id="smoke-retrieve",
            session_id="smoke",
        )
        state["sub_queries"] = [
            {**c, "hyde_used": False} for c in cases
        ]  # type: ignore[typeddict-item]
        patch = await node_retrieve(state, runtime, None)

        print(f"== error = {patch.get('error_code')} ==")
        for key in ("vector_hits", "sparse_hits", "hyde_hits"):
            hits = patch.get(key) or []
            print(f"== {key}: {len(hits)} 条 ==")
            for h in hits[:3]:
                print(
                    f"   [{h.get('sub_id')}] {h.get('source')} "
                    f"score={float(h.get('score') or 0):.3f} "
                    f"page={h.get('page_no')} section={h.get('section')} "
                    f"content={(h.get('content') or '')[:50]!r}"
                )
        print("== hyde_used ==")
        for sq in patch.get("sub_queries") or []:
            print(f"   {sq['sub_id']}: hyde_used={sq['hyde_used']}")
        assert patch.get("error_code") is None, patch.get("error")
        assert patch.get("vector_hits"), "dense 至少应有命中（库里已有 000008 年报）"
        # sq1/sq2 走条件 1（management_discussion/risk）必触发；sq3 含指标词看首轮得分决定条件 3
        hyde_map = {sq["sub_id"]: sq["hyde_used"] for sq in patch.get("sub_queries") or []}
        assert hyde_map.get("sq1") is True, "归因题应触发 HyDE"
        assert hyde_map.get("sq2") is True, "risk 章节应触发 HyDE"

    asyncio.run(_main())
    print("== smoke ok ==")
