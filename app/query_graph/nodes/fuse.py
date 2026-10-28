"""Q8 node_fuse（docs/04h）。

多路清单 → 唯一 evidence 列表：三路 hits 按 chunk_id RRF 去重融合，
metric/calc 归一为结构化证据按 sub_id 对齐，统一 ref 供 generate 引用。
为什么融合在汇合点：retrieve fan-out 跨单元拼接，去重打分只能在 merge 处做（04h §1.2）。
为什么 metric/calc 不进 RRF：查表/算式没有检索排名，硬造 rank 是编造可比性（04h §4.2）。
为什么全量合并再截断：先截断会丢跨路共识 chunk（04h §4.1）。
"""
from __future__ import annotations

from langgraph.runtime import Runtime
from langgraph.types import StreamWriter

from app.core.logger import logger
from app.query_graph.context import QueryAgentContext
from app.query_graph.state import QueryState

RRF_K = 60
RRF_WEIGHTS = {"dense": 1.0, "sparse": 0.9, "hyde": 0.8}
FUSE_TOP_K = 15
MAX_STRUCT_RESULTS = 20
_SNIPPET_LEN = 120

_SOURCE_ORDER = ("dense", "sparse", "hyde")


def rrf_score(rank_by_source: dict[str, int]) -> float:
    """score = Σ weight/(k+rank)。rank 是本路 1-based 名次。"""
    return sum(
        RRF_WEIGHTS[src] / (RRF_K + rank)
        for src, rank in rank_by_source.items()
        if src in RRF_WEIGHTS
    )


def step_1_rrf_merge(
    vector_hits: list[dict], sparse_hits: list[dict], hyde_hits: list[dict]
) -> list[dict]:
    """按 chunk_id 聚合三路：累加 RRF 分，source 合并命中路，保留首见元数据。

    原始 score 不可比（dense 余弦 vs sparse 匹配分），丢弃只留 RRF 分（04h §3.1）。
    保留副本会稀释 ref——RRF 累加本身就是多路共识加权（04h §3.1）。
    """
    merged: dict[str, dict] = {}
    for source, hits in (
        ("dense", vector_hits),
        ("sparse", sparse_hits),
        ("hyde", hyde_hits),
    ):
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.get("chunk_id")
            if not chunk_id:
                logger.warning(f"[node_fuse] 跳过缺 chunk_id 的 hit source={source}")
                continue
            if chunk_id not in merged:
                merged[chunk_id] = {
                    "chunk_id": chunk_id,
                    "content": hit.get("content") or "",
                    "page_no": hit.get("page_no"),
                    "document_id": hit.get("document_id"),
                    "stock_code": hit.get("stock_code"),
                    "report_period": hit.get("report_period"),
                    "type": hit.get("chunk_type") or "narrative",
                    "sub_id": hit.get("sub_id"),
                    "status": None,
                    "_ranks": {},
                }
            merged[chunk_id]["_ranks"][source] = rank

    items: list[dict] = []
    for chunk_id, item in merged.items():
        ranks: dict[str, int] = item.pop("_ranks")
        item["score"] = rrf_score(ranks)
        item["source"] = "+".join(s for s in _SOURCE_ORDER if s in ranks)
        items.append(item)
    return items


def step_2_sort_truncate(items: list[dict]) -> list[dict]:
    """全量排序后截断 FUSE_TOP_K；平分按 chunk_id 字典序保稳定（04h §3.2）。

    截到 15 而不是 rerank 的 10：给断崖算法留对比池，防相关 chunk 被误杀（04h §4.3）。
    """
    items.sort(key=lambda x: (-x["score"], x["chunk_id"]))
    if len(items) > FUSE_TOP_K:
        logger.warning(f"[node_fuse] 向量证据 {len(items)} 超限截尾至 {FUSE_TOP_K}")
        items = items[:FUSE_TOP_K]
    return items


def _stock_label(stock_code: str | None, code_to_name: dict | None) -> str:
    """证据行公司前缀：名称(代码)。

    compare 题两家公司同码指标只靠数值区分时，LLM 必串引用
    （600467 归母 43,427,553.57 被标成扣非那条的 [2]，validate 拒）。
    """
    code = stock_code or ""
    name = (code_to_name or {}).get(code)
    if name and code:
        return f"{name}({code})"
    return name or code


def _metric_content(row: dict, label: str) -> str:
    prefix = f"{label} " if label else ""
    # 切片行必须带 dim_name：compare 两家同码指标已会串引用，再叠维度不带 dim 串得更狠（04h §3.3）
    dim = f"{row.get('dim_name')} " if row.get("dim_name") else ""
    return f"{prefix}{dim}{row.get('metric_name')} {row.get('metric_period')}：{row.get('metric_value')}{row.get('unit') or ''}"


def _calc_content(row: dict, label: str) -> str:
    prefix = f"{label} " if label else ""
    if row.get("values") is not None:
        val_part = f"：{row['values']}"
    elif row.get("value") is not None:
        val_part = f"：{row['value']}{row.get('unit') or ''}"
    else:
        val_part = "："
    content = (
        f"{prefix}{row.get('metric_name')} {row.get('kind')}（{row.get('periods') or []}）"
        f"{val_part}；formula={row.get('formula') or ''}"
    )
    if row.get("status") != "ok":
        # status≠ok 仍要写进 content：generate 得说「缺期/不支持」而不是装没查（04h §3.3）
        content += f"；status={row.get('status')}：{row.get('reason')}"
    return content


def _calc_page_no(calc_row: dict, metric_rows: list[dict]):
    """calc 无 page_no，跟随同基期码的 metric 行（04h §3.3）。"""
    code = calc_row.get("metric_code") or ""
    for row in metric_rows:
        base = row.get("metric_code") or ""
        if code == base or code.startswith(base + "_"):
            return row.get("page_no")
    return None


def step_3_struct_evidence(
    metric_rows: list[dict], calc_results: list[dict], code_to_name: dict | None = None
) -> list[dict]:
    """metric/calc 归一为证据块。不参与 RRF：score=0.0，无排名可比（04h §4.2）。"""
    items: list[dict] = []
    for row in metric_rows:
        items.append(
            {
                "chunk_id": None,
                "type": "metric",
                "content": _metric_content(row, _stock_label(row.get("stock_code"), code_to_name)),
                "page_no": row.get("page_no"),
                "document_id": row.get("document_id"),
                "stock_code": row.get("stock_code"),
                "report_period": row.get("metric_period"),
                "source": "metric",
                "score": 0.0,
                "sub_id": row.get("sub_id"),
                "metric_code": row.get("metric_code"),
                "status": None,
            }
        )
    for row in calc_results:
        items.append(
            {
                "chunk_id": None,
                "type": "calc",
                "content": _calc_content(row, _stock_label(row.get("stock_code"), code_to_name)),
                "page_no": _calc_page_no(row, metric_rows),
                "document_id": row.get("document_id"),
                "stock_code": row.get("stock_code"),
                "report_period": None,
                "source": "calc",
                "score": 0.0,
                "sub_id": row.get("sub_id"),
                "metric_code": row.get("metric_code"),
                "status": row.get("status"),
            }
        )
    if len(items) > MAX_STRUCT_RESULTS:
        logger.warning(f"[node_fuse] 结构化证据 {len(items)} 超限截尾至 {MAX_STRUCT_RESULTS}")
        items = items[:MAX_STRUCT_RESULTS]
    return items


def step_4_align_refs(vector_items: list[dict], struct_items: list[dict]) -> list[dict]:
    """向量在前（RRF 序），后 metric 后 calc；ref 从 1 连续（04h §3.4）。

    连续编号是 generate `[n]` 引用对账的前提，跳号会让 validate 校验断链。
    """
    evidence: list[dict] = []
    for item in vector_items + struct_items:
        item["ref"] = len(evidence) + 1
        item["snippet"] = (item.get("content") or "")[:_SNIPPET_LEN]
        evidence.append(item)
    return evidence


async def node_fuse(
    state: QueryState,
    runtime: Runtime[QueryAgentContext],
    writer: StreamWriter = None,
) -> dict:
    try:
        vector_items = step_1_rrf_merge(
            state.get("vector_hits") or [],
            state.get("sparse_hits") or [],
            state.get("hyde_hits") or [],
        )
        vector_items = step_2_sort_truncate(vector_items)
        # code→name 反查：resolved_names 是 name→code；实体列表可能只有码
        resolved = state.get("resolved_names") or {}
        code_to_name = {str(c): str(n) for n, c in resolved.items() if n and c}
        struct_items = step_3_struct_evidence(
            state.get("metric_rows") or [],
            state.get("calc_results") or [],
            code_to_name,
        )
        evidence = step_4_align_refs(vector_items, struct_items)

        # SSE 只推向量类 + status=ok 的 metric/calc：流式层只展示可引用证据，
        # status≠ok 归最终回答话术（04h §3.5）
        ok_rows = [
            {
                "ref": e["ref"],
                "document_id": e.get("document_id"),
                "stock_code": e.get("stock_code"),
                "report_period": e.get("report_period"),
                "page_no": e.get("page_no"),
                "snippet": e.get("snippet"),
                "type": e.get("type"),
                "sub_id": e.get("sub_id"),
                "metric_code": e.get("metric_code"),
                "status": e.get("status"),
            }
            for e in evidence
            if e.get("type") != "calc" or e.get("status") == "ok"
        ]
        if writer is not None and ok_rows:
            writer({"type": "evidence", "rows": ok_rows})

        logger.info(
            f"[node_fuse] request_id={state.get('request_id')} "
            f"evidence={len(evidence)} sse={len(ok_rows)}"
        )
        return {"evidence": evidence, "error": None, "error_code": None, "error_stage": None}
    except Exception as e:
        logger.error(f"[node_fuse] 内部异常: {e}")
        return {
            "evidence": [],
            "error": str(e),
            "error_code": "INTERNAL",
            "error_stage": "fuse",
        }


if __name__ == "__main__":
    # 纯函数冒烟：不打库不调 LLM，造假 hits/metric/calc 走完整节点。
    # 覆盖：多路累加、去重 source、结构化归一、status≠ok 仍进、SSE 只推 ok、空合法。
    import asyncio

    from app.query_graph.state import default_state

    def _hit(cid, source, rank_score=0.9):
        return {
            "chunk_id": cid,
            "content": f"正文{cid} " + "长" * 200,
            "page_no": 10,
            "chunk_type": "narrative",
            "score": rank_score,
            "document_id": "doc1",
            "stock_code": "000008",
            "report_period": "2024",
            "sub_id": "sq1",
            "source": source,
        }

    class _W:
        def __init__(self):
            self.events = []

        def __call__(self, item):
            self.events.append(item)

    async def main():
        rt = Runtime(context=QueryAgentContext())

        print("== sq1 三路同 chunk 累加 ==")
        w = _W()
        patch = await node_fuse(
            {
                **default_state(query="q", request_id="smoke"),
                "vector_hits": [_hit("A", "dense")],
                "sparse_hits": [_hit("A", "sparse"), _hit("B", "sparse")],
                "hyde_hits": [_hit("A", "hyde")],
                "metric_rows": [],
                "calc_results": [],
            },
            rt,
            w,
        )
        for e in patch["evidence"]:
            print(f"  ref={e['ref']} chunk={e['chunk_id']} score={e['score']:.4f} source={e['source']}")

        print("== sq2 metric + calc 归一 ==")
        w2 = _W()
        patch2 = await node_fuse(
            {
                **default_state(query="q", request_id="smoke"),
                "vector_hits": [_hit("A", "dense")],
                "sparse_hits": [],
                "hyde_hits": [],
                "metric_rows": [
                    {
                        "stock_code": "000008",
                        "metric_code": "revenue",
                        "metric_name": "营业收入",
                        "metric_period": "2024",
                        "metric_value": 20.82,
                        "unit": "亿元",
                        "page_no": 45,
                        "sub_id": "sq1",
                    }
                ],
                "calc_results": [
                    {
                        "stock_code": "000008",
                        "metric_code": "revenue_yoy",
                        "metric_name": "营业收入",
                        "kind": "yoy",
                        "periods": ["2024", "2023"],
                        "value": 10.0,
                        "values": None,
                        "unit": "%",
                        "formula": "(c-p)/abs(p)*100",
                        "status": "ok",
                        "reason": None,
                        "sub_id": "sq1",
                    },
                    {
                        "stock_code": "000008",
                        "metric_code": "margin_vol",
                        "metric_name": "毛利率波动率",
                        "kind": "unsupported",
                        "periods": [],
                        "value": None,
                        "values": None,
                        "unit": "",
                        "formula": "",
                        "status": "unsupported",
                        "reason": "暂不支持波动率",
                        "sub_id": "sq1",
                    },
                ],
            },
            rt,
            w2,
        )
        for e in patch2["evidence"]:
            print(f"  ref={e['ref']} type={e['type']} source={e['source']} status={e['status']}")
            print(f"       content={e['content'][:60]}")
        sse_codes = [r.get("metric_code") for r in w2.events[0]["rows"]]
        print(f"  SSE rows codes={sse_codes}（unsupported 不该出现）")

        print("== sq3 空合法 ==")
        patch3 = await node_fuse(
            {**default_state(query="q", request_id="smoke"), "vector_hits": [], "sparse_hits": [], "hyde_hits": [], "metric_rows": [], "calc_results": []},
            rt,
            None,
        )
        print(f"  evidence={patch3['evidence']} error={patch3['error']}")

    asyncio.run(main())
