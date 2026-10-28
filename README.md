# Stock RAG — A 股财报 RAG 分析系统

面向 **A 股多公司、多年报/季报** 的检索增强生成（RAG）分析系统：把 PDF 财报解析为「带页码溯源的向量 chunk + 标准化财务指标库」，通过 LangGraph 双图（导入图 + 查询图）支撑**跨文档对比分析**与**指标计算推理**，输出**带引用、可对账、宁拒勿猜**的中文问答。

> 典型问题：
> - 「对比贵州茅台和五粮液 2023–2024 年毛利率变化，并说明原因」
> - 「600519 近三年 ROE 趋势如何？经营现金流能否覆盖净利润？」

---

## 目录

- [核心特性](#核心特性)
- [架构概览](#架构概览)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [API 一览](#api-一览)
- [数据模型](#数据模型)
- [项目结构](#项目结构)
- [开发与测试](#开发与测试)
- [路线图](#路线图)

---

## 核心特性

### 导入侧（PDF → 可查询数据）

| 能力 | 说明 |
|---|---|
| **双路入库** | 叙述性 chunk → Qdrant（dense + sparse）；结构化指标 → MySQL，两路互不阻塞 |
| **双轨切分** | 标题语义粗切 + 递归细切（max 2000 / min 500），保留 `page_no` 与章节路径 |
| **章节归类** | `management_discussion` / `risk` / `notes` 等 section 标签，供检索过滤 |
| **表格 LLM 抽取** | 规则过滤（title/sig 双通道）+ 核心报表结构化抽取，结果可单独重跑 |
| **口径映射 + L2 物化** | 原文名 → 标准 `metric_code`，单位统一，派生比率（毛利率/ROE…）按公式计算**不靠 LLM 心算** |
| **状态与幂等** | `parse_status` 断点续跑；sha1 去重、向量先删后写、指标 UNIQUE 防重 |
| **业务切片** | `business_facts` 开放维度（segment/产品/地区）+ share 占比抽取 |

### 查询侧（自然语言 → 带引用答案）

| 能力 | 说明 |
|---|---|
| **五分类意图** | `compare` / `metric_calc` / `trend` / `simple_qa` / `out_of_scope`，越界问题直接短路 |
| **实体解析** | 公司名四级匹配 → 股票代码；期间绝对化（「近三年」→ `2022/2023/2024`）；多轮指代消解 |
| **追问澄清** | 缺公司/期间时按优先级追问，不瞎猜 |
| **子问题拆解** | 复杂问题拆子问并标注路由（`vector` / `metric_sql` / `calc` / `mixed`） |
| **混合检索** | dense + sparse 必开，HyDE 按条件 auto 触发（叙述类/低特指问句），RRF 全量合并再截断 |
| **指标取数 + L3 计算** | SQL 查 L1/L2，纯代码算同比 / CAGR / 多期对比 |
| **证据融合与重排** | RRF 去重融合 + 证据对齐；bge-reranker 成对打分 + 断崖 TopK（失败降级原序） |
| **带引用生成** | 强制 CoT、`[n]` 引用编号、数字必须来自证据；**引用对账 + 数字校验**，不一致返回 `INSUFFICIENT_EVIDENCE` |
| **流式 SSE** | 全程可见：`intent → entities → evidence → metric → token → citation → done` |

---

## 架构概览

```
                    ┌─────────────────────────────────────┐
                    │         FastAPI REST / SSE           │
                    │   /api/import/*    /api/query/stream │
                    └──────────────┬──────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│  导入图 (LangGraph)│    │  查询图 (LangGraph)  │    │  进度/会话管理    │
│  MinerU→切分→元数据│    │  意图→拆解→fan-out   │    │  ProgressHub     │
│  →向量化→指标抽取  │    │  →融合→rerank→生成   │    │  Redis 会话历史   │
└────┬────────┬────┘     └───┬─────────┬─────┘     └─────────────────┘
     │        │              │         │
     ▼        ▼              ▼         ▼
┌─────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────┐
│ Qdrant  │ │  MySQL   │ │ Qdrant  │ │   MySQL      │
│ (chunk) │ │  指标库   │ │  混合检索 │ │  SQL 计算     │
└─────────┘ └──────────┘ └─────────┘ └──────────────┘
```

### 导入图节点

```
node_parse → node_metadata → node_split → node_sectionize
                                            │
              ┌─────────────────────────────┴────────────────────────┐
              ▼                                                      ▼
      node_embed_store                                    node_table_extract
      (chunk → Qdrant)                                    (表格 → LLM 抽取)
              │                                                      │
              │                                                      ▼
              │                                            node_metric_normalize
              │                                            (口径映射 + L2 计算)
              │                                                      │
              └──────────────────────┬───────────────────────────────┘
                                     ▼
                            node_commit_status
```

### 查询图节点（11 节点）

```
        node_intent → node_entity → node_clarify?
                          │              │ 缺关键实体 → 追问，短路 END
                          ▼              ▼
                    node_plan  ◄─────────┘
                    (子问题拆解 + route 标注)
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
   node_retrieve                node_metric_query
   (dense/sparse/hyde)             (SQL L1/L2)
          │                               │
          │                               ▼
          │                          node_calc (L3 趋势)
          │                               │
          └───────────────┬───────────────┘
                          ▼
                    node_fuse (RRF + 证据对齐)
                          ▼
                    node_rerank (断崖 TopK)
                          ▼
                    node_generate (CoT + 引用)
                          ▼
                    node_validate (页码/数字对账)
                          ▼
                        END
```

---

## 技术栈

| 组件 | 选型 | 说明 |
|---|---|---|
| 编排 | [LangGraph](https://github.com/langchain-ai/langgraph) | 双 StateGraph：导入图 + 查询图 |
| 向量库 | [Qdrant](https://qdrant.tech/) | dense + sparse 混合检索，先 filter 后向量 |
| 指标存储 | MySQL（SQLAlchemy + PyMySQL） | 财务指标结构化存储、SQL 聚合/对比 |
| 文档解析 | MinerU | PDF → Markdown + 表格 + 页码 |
| LLM | Qwen（OpenAI 兼容协议） | 意图、实体、拆解、表格抽取、生成 |
| Embedding | BGE-M3（FlagEmbedding） | dense + 真实 sparse；可切 DashScope `text-embedding-v3` |
| Rerank | bge-reranker-large | 成对打分、断崖 TopK |
| API | FastAPI + Uvicorn | REST + SSE |
| 会话 | Redis | 多轮对话历史、窗口裁剪 |
| 前端 | 静态 HTML（`frontend/`） | 导入进度页 + 对话页，与 API 同源挂载 |
| Python | 3.12 | 依赖管理 [uv](https://github.com/astral-sh/uv) / pip |

---

## 快速开始

### 环境要求

- Python 3.12
- MySQL 8.x、Qdrant（本地或远程）、Redis
- 可用的 Qwen API Key（OpenAI 兼容）
- MinerU 云端 API Key（PDF 解析）

### 1. 安装依赖

```bash
git clone <your-repo-url> stock_rag
cd stock_rag

# 使用 uv（推荐）
uv sync

# 或 pip
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 2. 配置环境变量

在项目根目录创建 `.env`（已被 gitignore），参考下表填写：

| 变量 | 说明 |
|---|---|
| `LLM_API_BASE` / `LLM_API_KEY` / `LLM_MODEL` | Qwen（OpenAI 兼容）接口 |
| `LLM_ENABLE_THINKING` | 思考链开关（默认关，延迟敏感） |
| `EMBEDDING_MODEL` / `EMBEDDING_DEVICE` | BGE-M3 模型路径与 `cpu` / `cuda` |
| `RERANK_MODEL` | bge-reranker 模型路径 |
| `QDRANT_HOST` / `QDRANT_PORT` / `QDRANT_EMBEDDING_SIZE` | 向量库 |
| `MYSQL_HOST` / `MYSQL_PORT` / `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_DATABASE` | 指标库 |
| `MINERU_API_BASE` / `MINERU_API_KEY` / `MINERU_TIMEOUT_S` | MinerU 云端解析 |
| `HYDE_ENABLED` / `HYDE_FALLBACK_THRESHOLD` | HyDE 触发策略（`auto` / `on` / `off`） |

### 3. 初始化存储

```bash
# MySQL 表结构（schema.sql 含 DDL + metric_definitions 种子数据）
mysql -u<user> -p<password> <database> < sql/schema.sql
# 或通过脚本封装执行
python scripts/init_db.py

# 创建 Qdrant collection
python scripts/init_qdrant.py
```

### 4. 启动服务

```bash
python -m app.main
# 或
uvicorn app.main:app --host 0.0.0.0 --port 8003
```

启动后：

- 对话页：<http://localhost:8003/>
- 导入页：<http://localhost:8003/import.html>
- API 文档：<http://localhost:8003/docs>

### 5. 导入一份财报

```bash
curl -X POST http://localhost:8003/api/import/report \
  -F "file=@report/000006_深振业A_2024.pdf"
```

进度查询：`GET /api/import/progress/{document_id}`（SSE 推送）。

### 6. 提问

```bash
curl -N -X POST http://localhost:8003/api/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "贵州茅台 2024 年毛利率是多少", "session_id": "s_demo"}'
```

---

## API 一览

### 导入

| Method | Path | 说明 |
|---|---|---|
| `POST` | `/api/import/report` | 上传 PDF，返回 `document_id` 并后台解析 |
| `GET` | `/api/import/progress/{document_id}` | SSE 订阅解析进度 |

### 查询

| Method | Path | 说明 |
|---|---|---|
| `POST` | `/api/query/stream` | 一段式 SSE 流式问答（推荐） |

**请求：**

```json
{
  "query": "对比贵州茅台和五粮液2023-2024年毛利率变化，并说明原因",
  "session_id": "s_abc123"
}
```

**SSE 事件类型：**

| event | data |
|---|---|
| `intent` / `entities` / `clarify` | 理解阶段结果与追问 |
| `sub_query` / `evidence` / `metric` | 子问执行、证据块（含页码）、指标取数 |
| `token` | 生成增量 |
| `citation` | 引用定稿 |
| `done` / `error` | 终态 |

**错误码约定：** `INVALID_PARAM` / `MISSING_ENTITY` / `OUT_OF_SCOPE` / `INSUFFICIENT_EVIDENCE` / `DEPENDENCY_UNAVAILABLE` / `INTERNAL`

---

## 数据模型

### 文档元数据（贯穿全链路）

```
document_id      # sha1(pdf bytes)，幂等去重键
stock_code       # "600519"
stock_name       # "贵州茅台"
report_type      # annual | q1 | q2 | q3 | interim | prospectus
report_period    # "2024" 或 "2024Q3"
fiscal_year      # 2024
page_no          # chunk/指标所在页码（引用溯源）
```

### Qdrant `report_chunks`

`chunk_id` · `document_id` · `stock_code` · `report_period` · `report_type` · `section` · `page_no` · `content` · `dense_vector` · `sparse_vector`

### MySQL 核心表

| 表 | 用途 |
|---|---|
| `report_documents` | 财报元数据与 `parse_status` |
| `report_tables` | MinerU 表格原文 + 结构化抽取快照（可重跑） |
| `financial_metrics` | 标准化财务指标（L1 原始科目 + L2 派生比率） |
| `metric_definitions` | 指标字典：编码、别名、`calc_formula`、默认单位 |
| `business_facts` | 业务切片（segment 开放维度、share 占比） |

### 指标分层

| 层级 | 例子 | 计算时机 |
|---|---|---|
| **L1** 原始科目 | 营业收入、净利润、总资产、经营现金流 | 入库时表格抽取 |
| **L2** 派生比率 | 毛利率、净利率、ROE、ROA、资产负债率 | 入库时按公式物化 |
| **L3** 趋势特征 | 同比、环比、CAGR、多期对比 | 查询时纯代码计算 |

字段设计要点（来自 3 份真实 A 股年报实测）：**合并/母公司双口径**（`statement_scope`）、**对比期多列落表**（`metric_period` + `metric_period_type`）、**追溯调整标记**（`is_restated`）。

完整 DDL 见 [sql/schema.sql](sql/schema.sql)。

---

## 项目结构

```
stock_rag/
├─ app/
│  ├─ conf/           # 配置（dataclass + .env）
│  ├─ clients/        # Qdrant / MySQL / Redis / 任务状态
│  ├─ lm/             # LLM、Embedding、Reranker 封装
│  ├─ import_graph/   # 导入图节点 + State + ProgressHub + API
│  ├─ query_graph/    # 查询图节点 + State + 会话历史 + API
│  ├─ metrics/        # 指标抽取、口径映射、L2/L3 计算
│  ├─ repositories/   # 数据访问层（依赖注入进节点）
│  ├─ entities/       # 领域实体 / State 类型
│  ├─ core/           # logger、prompt 加载、异常
│  ├─ utils/
│  └─ main.py         # FastAPI 入口（挂载 API + frontend 静态页）
├─ prompts/           # *.prompt 模板（外置、可单测）
├─ frontend/          # 静态页：index / chat / import
├─ sql/               # schema.sql
├─ scripts/           # init_db.py / init_qdrant.py
├─ report/            # 样例 PDF 财报
└─ tests/             # pytest
```

---

## 开发与测试

```bash
# 运行测试
pytest

# 只跑某一模块
pytest tests/test_cliff_topk.py -v
```

工程约定：

- 节点统一签名 `(state, runtime, writer)`，Repository 从 context 注入，不自建
- 节点内 `step_N` 函数拆分，便于单测
- Prompt 全部外置 `prompts/*.prompt`
- 禁止吞异常；失败写 `error` 并按需短路
- State 使用工厂默认值，防跨请求污染
- 数字纪律：生成节点禁止心算，引用数字必须来自证据

---

## 路线图

- [x] P0 工程骨架与表结构
- [x] P1 数据构建全链路
- [x] P2 导入图节点
- [x] P3 查询图节点
- [x] P4 查询 API（SSE）
- [x] Web UI（导入进度 + 对话，静态页）
- [ ] P5 指标字典全量 + L2/L3 公式（`06-指标字典与计算.md`）
- [ ] P6 评测与可观测（`07-评测与可观测.md`）
- [ ] 多轮对话体验优化、Agent 动态规划（远期，明确不做于一期）

### 一期明确不做（YAGNI）

Agent 动态规划循环 · 知识图谱 · 多模态图表理解 · 生产级权限体系（仅预留 token）

---

## License

MIT（如需其他协议请替换本节）。