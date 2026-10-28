"""标题粗切测试——docs/02a §3.1：中文序号推层级，Markdown ## 层级失效。"""
from app.import_graph.split.title_split import split_by_title, infer_level


def test_infer_level_chinese_sections():
    assert infer_level("第十节 财务报告") == 1
    assert infer_level("第1节 管理层讨论与分析") == 1
    assert infer_level("五、合并财务报表") == 2
    assert infer_level("（一）主营业务分析") == 3
    assert infer_level("(一) 主营业务分析") == 3
    assert infer_level("1、 利润表变动分析") == 3


def test_infer_level_plain_title_hangs_on_recent():
    # 无序号 → 0（挂最近祖先）
    assert infer_level("报告期末公司前三年主要会计数据") == 0


def test_split_by_title_builds_section_path():
    md = (
        "# 第十节 财务报告\n"
        "前言\n"
        "## 五、合并财务报表\n"
        "报表说明\n"
        "## 合并利润表\n"
        "利润数据\n"
    )
    sections = split_by_title(md, file_title="新世界2024年报")
    paths = [s["section_path"] for s in sections]
    assert "第十节 财务报告" in paths[0]
    assert paths[1] == "第十节 财务报告/五、合并财务报表"
    # 无序号标题挂最近祖先（五、合并财务报表）
    assert paths[2] == "第十节 财务报告/五、合并财务报表/合并利润表"


def test_split_by_title_empty_doc_single_section():
    sections = split_by_title("没有标题的正文\n另一段\n", file_title="x")
    assert len(sections) == 1
    assert sections[0]["title"] == "文档正文"
    assert "没有标题的正文" in sections[0]["content"]


def test_split_by_title_skips_code_block_fake_title():
    md = (
        "# 真标题\n"
        "```\n"
        "# 这是代码注释不是标题\n"
        "```\n"
        "正文\n"
    )
    sections = split_by_title(md, file_title="x")
    titles = [s["title"] for s in sections]
    assert titles == ["# 真标题"]


def test_split_by_title_records_char_offsets():
    md = "# 标题一\n内容AAA\n\n# 标题二\n内容BBB\n"
    sections = split_by_title(md, file_title="x")
    s2 = sections[1]
    # 标题二 在原文中的位置
    assert s2["char_start"] == md.index("# 标题二")
    assert s2["char_end"] == len(md.rstrip())


def test_split_by_title_sets_parent_title():
    """粗切就要产出 parent_title（docs/02a §3.1）——合并阶段靠它判断同级。"""
    md = (
        "# 第十节 财务报告\n"
        "前言\n"
        "## 五、合并财务报表\n"
        "报表说明\n"
        "## 合并利润表\n"
        "利润数据\n"
    )
    sections = split_by_title(md, file_title="新世界2024年报")
    assert sections[0]["parent_title"] == "新世界2024年报"
    assert sections[1]["parent_title"] == "第十节 财务报告"
    assert sections[2]["parent_title"] == "五、合并财务报表"


def test_split_by_title_parent_is_ancestor_not_sibling():
    """父必须是更高级祖先，不能是同级前兄弟——否则 L3 各挂各的父，短块并不上。"""
    md = (
        "# 第十节 财务报告\n"
        "前言\n"
        "## 三、费用\n"
        "费用说明\n"
        "## 4、研发投入\n"
        "研发说明\n"
        "## 5、现金流\n"
        "现金说明\n"
    )
    sections = split_by_title(md, file_title="x")
    # 4、和 5、是 L3 兄弟，父都应是「三、费用」
    assert sections[2]["title"] == "## 4、研发投入"
    assert sections[3]["title"] == "## 5、现金流"
    assert sections[2]["parent_title"] == "三、费用"
    assert sections[3]["parent_title"] == "三、费用"
