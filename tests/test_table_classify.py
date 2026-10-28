"""classify_table 测试：title/sig 双通道、TITLE_EXCLUDE、弱标题约束（docs/02d §3/§4）。"""
from app.import_graph.table_classify import classify_table

# 核心报表样例：列头含期间栏，行首含科目，底部有合计锚点
BALANCE_HTML = (
    "<table><tr><td>项目</td><td>本年年末数</td><td>上年年末数</td></tr>"
    "<tr><td>货币资金</td><td>100</td><td>90</td></tr>"
    "<tr><td>流动资产合计</td><td>200</td><td>180</td></tr>"
    "<tr><td>资产总计</td><td>500</td><td>450</td></tr>"
    "<tr><td>负债合计</td><td>300</td><td>280</td></tr></table>"
)
INCOME_HTML = (
    "<table><tr><td>项目</td><td>本年金额</td><td>上年金额</td></tr>"
    "<tr><td>营业收入</td><td>100</td><td>90</td></tr>"
    "<tr><td>利润总额</td><td>50</td><td>40</td></tr>"
    "<tr><td>净利润</td><td>40</td><td>30</td></tr></table>"
)
CASHFLOW_HTML = (
    "<table><tr><td>项目</td><td>本年金额</td><td>上年金额</td></tr>"
    "<tr><td>经营活动产生的现金流量</td><td>10</td><td>9</td></tr>"
    "<tr><td>投资活动</td><td>1</td><td>2</td></tr>"
    "<tr><td>筹资活动</td><td>3</td><td>4</td></tr>"
    "<tr><td>现金及现金等价物</td><td>8</td><td>7</td></tr></table>"
)
EQUITY_HTML = (
    "<table><tr><td>项目</td><td>本期金额</td><td>上期金额</td></tr>"
    "<tr><td>股本</td><td>100</td><td>100</td></tr>"
    "<tr><td>资本公积</td><td>50</td><td>50</td></tr>"
    "<tr><td>未分配利润</td><td>20</td><td>10</td></tr></table>"
)
HEADLINE_HTML = (
    "<table><tr><td>项目</td><td>2024年</td><td>2023年</td></tr>"
    "<tr><td>营业收入</td><td>100</td><td>90</td></tr>"
    "<tr><td>归属于上市公司股东的净利润</td><td>10</td><td>9</td></tr>"
    "<tr><td>基本每股收益</td><td>0.1</td><td>0.09</td></tr>"
    "<tr><td>总资产</td><td>500</td><td>450</td></tr></table>"
)
# 附注「货币资金明细」：有 期末余额/货币资金，但没有 资产总计 等合计锚点
NOTES_HTML = (
    "<table><tr><td>项目</td><td>期末余额</td><td>期初余额</td></tr>"
    "<tr><td>库存现金</td><td>1</td><td>2</td></tr>"
    "<tr><td>其他货币资金</td><td>3</td><td>4</td></tr>"
    "<tr><td>合计</td><td>4</td><td>6</td></tr></table>"
)


def test_title_channel_balance_consolidated():
    kind, how, scope = classify_table("1、合并资产负债表", BALANCE_HTML)
    assert (kind, how, scope) == ("balance", "title", "consolidated")


def test_title_channel_balance_parent():
    kind, how, scope = classify_table("2、母公司资产负债表", BALANCE_HTML)
    assert (kind, how, scope) == ("balance", "title", "parent")


def test_title_channel_income_cashflow_equity():
    assert classify_table("3、合并利润表", INCOME_HTML)[0] == "income"
    assert classify_table("5、合并现金流量表", CASHFLOW_HTML)[0] == "cashflow"
    # （续）后缀被正则吞掉——深振业续表也要命中
    kind, how, scope = classify_table("合并所有者权益变动表（续）2024 年度", EQUITY_HTML)
    assert (kind, how) == ("equity", "title")
    assert how == "title"


def test_title_channel_headline():
    kind, how, _ = classify_table("六、主要会计数据和财务指标", HEADLINE_HTML)
    assert (kind, how) == ("headline", "title")
    kind, how, _ = classify_table("八、分季度主要财务指标", HEADLINE_HTML)
    assert (kind, how) == ("headline", "title")


def test_title_channel_segment():
    # 陕西煤业 id=34 实测 caption；切片列头与 income 重叠，只走标题通道（docs/02g §4.1）
    kind, how, _ = classify_table(
        "(1). 主营业务分行业、分产品、分地区、分销售模式情况", INCOME_HTML
    )
    assert (kind, how) == ("segment", "title")
    assert classify_table("主营业务构成情况", INCOME_HTML)[0] == "segment"
    assert classify_table("（一）分行业情况", INCOME_HTML)[0] == "segment"
    assert classify_table("分产品情况", INCOME_HTML)[0] == "segment"
    assert classify_table("分地区情况", INCOME_HTML)[0] == "segment"
    assert classify_table("分销售模式情况", INCOME_HTML)[0] == "segment"


def test_title_channel_segment_not_stolen_by_headline():
    # 「分季度主要财务」先命中 headline，segment 规则不得回吞
    kind, how, _ = classify_table("分季度主要财务指标", HEADLINE_HTML)
    assert (kind, how) == ("headline", "title")


def test_segment_title_does_not_enter_sig():
    # 切片表列头与 income 重叠，弱 caption 也不许进 sig 误判成利润表碎片（docs/02g §4.1）
    seg_html = (
        "<table><tr><td>项目</td><td>营业收入</td><td>毛利率(%)</td></tr>"
        "<tr><td>煤炭采掘业</td><td>100</td><td>30</td></tr>"
        "<tr><td>合计</td><td>200</td><td>25</td></tr></table>"
    )
    kind, how, _ = classify_table("", seg_html)
    assert (kind, how) == (None, "skip")


def test_title_channel_unit_line_in_caption_uses_first_line():
    # extract_caption 会把单位行拼在标题后，分类只看第 0 行
    cap = "深圳市振业（集团）股份有限公司合并利润表2024 年度\n（除特别注明外，金额单位均为人民币元）"
    kind, how, scope = classify_table(cap, INCOME_HTML)
    assert (kind, how, scope) == ("income", "title", "consolidated")


def test_title_exclude_beats_title_rules():
    # 「现金流量表补充资料」含报表字样，但属于补充资料，必须挡掉
    kind, how, _ = classify_table("现金流量表补充资料", CASHFLOW_HTML)
    assert kind is None and how == "skip"
    kind, how, _ = classify_table("十二、非经常性损益项目及金额", INCOME_HTML)
    assert kind is None
    kind, how, _ = classify_table("关键审计事项", BALANCE_HTML)
    assert kind is None
    kind, how, _ = classify_table("十一、资产构成情况", BALANCE_HTML)
    assert kind is None


def test_sig_channel_only_when_caption_weak():
    # 单位帽：深振业A 整本报表的 caption 就是这一行
    kind, how, scope = classify_table("（除特别注明外，金额单位均为人民币元）", BALANCE_HTML)
    assert (kind, how, scope) == ("balance", "sig", "")
    kind, how, _ = classify_table("", BALANCE_HTML)
    assert (kind, how) == ("balance", "sig")
    kind, how, _ = classify_table("单位：元 币种：人民币", BALANCE_HTML)
    assert (kind, how) == ("balance", "sig")


def test_sig_channel_three_conditions_required():
    # 缺 extra 锚点（只有合计，没有资产总计）→ 附注明细，不进 LLM
    kind, how, _ = classify_table("（除特别注明外，金额单位均为人民币元）", NOTES_HTML)
    assert kind is None and how == "skip"


def test_strong_caption_blocks_sig_channel():
    # 有明确标题却不像报表：不能被表内「货币资金/期末余额」误判
    kind, how, _ = classify_table("货币资金明细表", BALANCE_HTML)
    assert kind is None and how == "skip"


def test_sig_channel_hits_each_statement_type():
    assert classify_table("", INCOME_HTML)[0] == "income"
    assert classify_table("", CASHFLOW_HTML)[0] == "cashflow"
    assert classify_table("", EQUITY_HTML)[0] == "equity"
    assert classify_table("", HEADLINE_HTML)[0] == "headline"
