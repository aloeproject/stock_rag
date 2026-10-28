-- stock_rag MySQL DDL + metric_definitions 种子
-- 口径与 docs/01 §5 一致；business_facts 见 docs/02g
-- 执行：python scripts/init_db.py（幂等，可重复执行）

-- 若使用独立库，先执行：
-- CREATE DATABASE IF NOT EXISTS stock_rag DEFAULT CHARSET utf8mb4;
-- USE stock_rag;

-- 财报文档表
CREATE TABLE IF NOT EXISTS report_documents (
  document_id   VARCHAR(64)  NOT NULL COMMENT '文档唯一ID，取 PDF 文件内容 sha1',
  stock_code    VARCHAR(16)  NOT NULL COMMENT '股票代码，6位',
  stock_name    VARCHAR(64)           COMMENT '股票简称',
  report_type   VARCHAR(16)  NOT NULL COMMENT '报告类型：annual/q1/q2/q3/interim/prospectus',
  report_period VARCHAR(16)  NOT NULL COMMENT '报告期 YYYY 或 YYYYQn',
  fiscal_year   SMALLINT     NOT NULL COMMENT '会计年度',
  source_file   VARCHAR(255)          COMMENT '原始上传文件名',
  parse_status  VARCHAR(24)  NOT NULL DEFAULT 'pending'
                  COMMENT '解析状态：pending/parsing/splitting/embedding/extracting/done/failed(part)/failed',
  stats         JSON                  COMMENT '阶段统计：chunks/tables_ok/metrics_rows/unmapped_names',
  created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '入库时间',
  updated_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (document_id),
  KEY idx_code_period (stock_code, report_period)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='财报文档表：元数据与解析状态';

-- 标准化财务指标表
CREATE TABLE IF NOT EXISTS financial_metrics (
  id              BIGINT       NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  document_id     VARCHAR(64)  NOT NULL COMMENT '来源文档ID',
  stock_code      VARCHAR(16)  NOT NULL COMMENT '股票代码',
  report_period   VARCHAR(16)  NOT NULL COMMENT '来源报告期',
  report_type     VARCHAR(16)  NOT NULL COMMENT '来源报告类型',
  metric_period   VARCHAR(16)  NOT NULL COMMENT '数值所属期间，如 2024 / 2023 / 2024末',
  metric_period_type VARCHAR(16) NOT NULL COMMENT 'current/comparative/restated',
  statement_type  VARCHAR(16)  NOT NULL COMMENT 'headline/income/balance/cashflow/ratio',
  statement_scope VARCHAR(16)  NOT NULL DEFAULT 'consolidated' COMMENT 'consolidated/parent',
  metric_code     VARCHAR(64)  NOT NULL COMMENT '标准化指标编码',
  metric_name     VARCHAR(128)          COMMENT '财报原文指标名',
  metric_value    DECIMAL(24,6)         COMMENT '标准化数值',
  unit            VARCHAR(16)           COMMENT '元/万元/亿元/%/元每股',
  value_raw       VARCHAR(64)           COMMENT '原文数值字符串',
  is_restated     TINYINT(1)   NOT NULL DEFAULT 0 COMMENT '是否追溯调整值',
  page_no         INT                   COMMENT 'PDF 页码',
  extract_source  VARCHAR(16)  NOT NULL DEFAULT 'table' COMMENT 'headline/table/derived',
  created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '入库时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_metric (document_id, metric_code, statement_scope, metric_period, metric_period_type),
  KEY idx_code_period (stock_code, report_period, metric_code),
  KEY idx_metric_period (stock_code, metric_code, metric_period, statement_scope)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='标准化财务指标表';

-- 指标口径字典
CREATE TABLE IF NOT EXISTS metric_definitions (
  metric_code    VARCHAR(64)  NOT NULL COMMENT '标准化指标编码',
  metric_name    VARCHAR(128) NOT NULL COMMENT '指标中文标准名',
  aliases        VARCHAR(255)          COMMENT '原文别名，竖线分隔',
  category       VARCHAR(32)           COMMENT 'headline/asset/liability/equity/income/cost/cashflow/profitability/solvency/efficiency/growth',
  statement_type VARCHAR(16)           COMMENT 'headline/income/balance/cashflow/ratio',
  metric_level   TINYINT      NOT NULL DEFAULT 1 COMMENT '1-L1 2-L2 3-L3',
  calc_formula   VARCHAR(255)          COMMENT 'L2/L3 公式；L1 空',
  unit_default   VARCHAR(16)           COMMENT '默认单位',
  description    VARCHAR(512)          COMMENT '口径说明',
  PRIMARY KEY (metric_code),
  KEY idx_stmt (statement_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='指标口径字典表';

-- 财报原始表格
CREATE TABLE IF NOT EXISTS report_tables (
  id                BIGINT       NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  document_id       VARCHAR(64)  NOT NULL COMMENT '来源文档ID',
  page_no           INT                   COMMENT 'PDF 页码',
  table_title       VARCHAR(255)          COMMENT '表格标题',
  statement_type    VARCHAR(16)           COMMENT 'headline/income/balance/cashflow/equity/segment/other',
  statement_scope   VARCHAR(16)           COMMENT 'consolidated/parent/empty',
  raw_markdown      MEDIUMTEXT            COMMENT 'MinerU Markdown 原文',
  extracted_metrics JSON                  COMMENT '结构化抽取快照',
  extract_status    VARCHAR(16)  NOT NULL DEFAULT 'pending' COMMENT 'pending/ok/failed/skipped',
  created_at        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_doc (document_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='财报原始表格表';

-- 业务切片事实表（开放维度 EAV，详见 docs/02g）
-- 为什么不并进 financial_metrics：开放维度不该掺进公司级 uk 与 L2 公式（docs/02g §1.1）
CREATE TABLE IF NOT EXISTS business_facts (
  id              BIGINT       NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  document_id     VARCHAR(64)  NOT NULL COMMENT '来源文档ID',
  stock_code      VARCHAR(16)  NOT NULL COMMENT '股票代码',
  report_period   VARCHAR(16)  NOT NULL COMMENT '来源报告期',
  report_type     VARCHAR(16)  NOT NULL COMMENT '来源报告类型',
  dim_type        VARCHAR(32)  NOT NULL COMMENT '切片类型：industry/product/region/channel/customer/other（枚举可扩）',
  dim_name        VARCHAR(128) NOT NULL COMMENT '切片值原文：煤炭采掘业/自产煤/直销/合计（不入字典）',
  metric_code     VARCHAR(64)           COMMENT '标准化指标编码；未映射为 NULL',
  metric_name     VARCHAR(128) NOT NULL COMMENT '原文列头：营业收入/毛利率(%)/营业收入比上年增减(%)',
  metric_period   VARCHAR(16)  NOT NULL COMMENT '数值所属期间，如 2024',
  metric_value    DECIMAL(24,6)         COMMENT '标准化数值',
  unit            VARCHAR(16)           COMMENT '元/万元/亿元/%',
  value_raw       VARCHAR(64)           COMMENT '原文数值字符串',
  page_no         INT                   COMMENT 'PDF 页码',
  extract_source  VARCHAR(16)  NOT NULL DEFAULT 'table' COMMENT 'table/derived',
  created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '入库时间',
  PRIMARY KEY (id),
  -- uk 用 metric_name 不用 metric_code：后者可空，MySQL NULL 不去重（docs/02g §3.2）
  UNIQUE KEY uk_fact (document_id, dim_type, dim_name, metric_name, metric_period),
  KEY idx_query (stock_code, report_period, dim_type, dim_name),
  KEY idx_metric (stock_code, metric_code, metric_period)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='业务切片事实表（开放维度）';

-- metric_definitions 种子数据（首批 39 条，docs/01 §5.2）
INSERT IGNORE INTO metric_definitions
(metric_code, metric_name, aliases, category, statement_type, metric_level, calc_formula, unit_default, description) VALUES
-- headline
('revenue', '营业收入', '营业总收入', 'headline', 'headline', 1, NULL, '元', '利润表营业收入/营业总收入'),
('net_profit_attributable', '归属于上市公司股东的净利润', '归属于母公司股东的净利润|归母净利润', 'headline', 'headline', 1, NULL, '元', '归母净利润'),
('net_profit_attributable_deducted', '归属于上市公司股东的扣除非经常性损益的净利润', '扣非归母净利润', 'headline', 'headline', 1, NULL, '元', '扣非归母'),
('net_operating_cashflow', '经营活动产生的现金流量净额', '经营活动现金流量净额', 'cashflow', 'cashflow', 1, NULL, '元', '经营净现金流'),
('eps_basic', '基本每股收益', NULL, 'headline', 'headline', 1, NULL, '元每股', '基本EPS'),
('eps_diluted', '稀释每股收益', NULL, 'headline', 'headline', 1, NULL, '元每股', '稀释EPS'),
('roe_weighted', '加权平均净资产收益率', '加权平均ROE', 'profitability', 'ratio', 1, NULL, '%', '披露的加权ROE'),
('total_assets', '总资产', '资产总计', 'asset', 'balance', 1, NULL, '元', '期末总资产'),
('net_assets_attributable', '归属于上市公司股东的净资产', '归属于母公司股东的净资产|归母净资产', 'equity', 'balance', 1, NULL, '元', '期末归母净资产'),
-- income L1
('cost_of_revenue', '营业成本', '营业总成本', 'cost', 'income', 1, NULL, '元', NULL),
('taxes_surcharges', '税金及附加', NULL, 'cost', 'income', 1, NULL, '元', NULL),
('selling_expenses', '销售费用', NULL, 'cost', 'income', 1, NULL, '元', NULL),
('admin_expenses', '管理费用', NULL, 'cost', 'income', 1, NULL, '元', NULL),
('rd_expenses', '研发费用', NULL, 'cost', 'income', 1, NULL, '元', NULL),
('finance_expenses', '财务费用', NULL, 'cost', 'income', 1, NULL, '元', NULL),
('operating_profit', '营业利润', NULL, 'income', 'income', 1, NULL, '元', NULL),
('total_profit', '利润总额', NULL, 'income', 'income', 1, NULL, '元', NULL),
('net_profit', '净利润', NULL, 'income', 'income', 1, NULL, '元', '含少数股东'),
('minority_profit', '少数股东损益', NULL, 'income', 'income', 1, NULL, '元', NULL),
-- balance L1（节选常用）
('cash_and_equivalents', '货币资金', NULL, 'asset', 'balance', 1, NULL, '元', NULL),
('accounts_receivable', '应收账款', NULL, 'asset', 'balance', 1, NULL, '元', NULL),
('inventory', '存货', NULL, 'asset', 'balance', 1, NULL, '元', NULL),
('fixed_assets', '固定资产', NULL, 'asset', 'balance', 1, NULL, '元', NULL),
('short_term_borrowings', '短期借款', NULL, 'liability', 'balance', 1, NULL, '元', NULL),
('accounts_payable', '应付账款', NULL, 'liability', 'balance', 1, NULL, '元', NULL),
('current_assets_total', '流动资产合计', NULL, 'asset', 'balance', 1, NULL, '元', NULL),
('current_liabilities_total', '流动负债合计', NULL, 'liability', 'balance', 1, NULL, '元', NULL),
('total_liabilities', '负债合计', NULL, 'liability', 'balance', 1, NULL, '元', NULL),
('share_capital', '股本', NULL, 'equity', 'balance', 1, NULL, '元', NULL),
('undistributed_profit', '未分配利润', NULL, 'equity', 'balance', 1, NULL, '元', NULL),
('equity_total', '所有者权益合计', NULL, 'equity', 'balance', 1, NULL, '元', NULL),
-- L2
('gross_margin', '毛利率', NULL, 'profitability', 'ratio', 2, '(revenue-cost_of_revenue)/revenue*100', '%', '综合毛利率'),
('net_margin', '净利率', NULL, 'profitability', 'ratio', 2, 'net_profit_attributable/revenue*100', '%', NULL),
('roe', '净资产收益率', NULL, 'profitability', 'ratio', 2, 'net_profit_attributable/avg_equity_attributable*100', '%', '期初期末平均'),
('roa', '总资产收益率', NULL, 'profitability', 'ratio', 2, 'net_profit/avg_total_assets*100', '%', NULL),
('debt_to_asset_ratio', '资产负债率', NULL, 'solvency', 'ratio', 2, 'total_liabilities/total_assets*100', '%', NULL),
('ocf_to_net_profit', '经营现金流/净利润', NULL, 'cashflow', 'ratio', 2, 'net_operating_cashflow/net_profit_attributable*100', '%', NULL),
('current_ratio', '流动比率', NULL, 'solvency', 'ratio', 2, 'current_assets_total/current_liabilities_total', '倍', NULL),
('revenue_yoy', '营业收入同比增长率', NULL, 'growth', 'ratio', 3, 'revenue_t/revenue_t1-100', '%', 'L3 查询期计算');
