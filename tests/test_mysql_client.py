"""mysql_config / mysql_client 测试：URL 拼装、单例、session_factory 可用。"""
from app.conf.mysql_config import MysqlConfig, mysql_config
from app.clients.mysql_client import MySQLClientManager


def test_mysql_config_from_env():
    assert isinstance(mysql_config, MysqlConfig)
    for f in ("host", "port", "user", "password", "database"):
        assert hasattr(mysql_config, f)


def test_build_url():
    cfg = MysqlConfig(
        host="127.0.0.1", port=3306, user="root", password="pw", database="stock_rag"
    )
    url = MySQLClientManager.build_url(cfg)
    assert url == "mysql+pymysql://root:pw@127.0.0.1:3306/stock_rag?charset=utf8mb4"


def test_manager_is_singleton():
    m1 = MySQLClientManager()
    m2 = MySQLClientManager()
    assert m1 is m2


def test_init_creates_engine_and_session_factory():
    m = MySQLClientManager()
    cfg = MysqlConfig(
        host="localhost", port=3306, user="u", password="p", database="d"
    )
    m.init(cfg)
    assert m.engine is not None
    assert m.session_factory is not None
    m.close()
