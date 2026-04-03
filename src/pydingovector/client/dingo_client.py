import logging
from typing import Optional, List
from urllib.parse import quote

import sqlalchemy.sql.functions as func_mod
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Index,
    text,
    inspect,
)
from sqlalchemy.dialects import registry

from pydingovector.schema.dingo_table import DingoTable

# 适配 Dingodb 自定义方言（若有），无则注册通用 MySQL 方言
registry.register(
    "mysql.dingodb", "pyobvector.schema.dialect", "OceanBaseDialect"  # 复用现有方言或替换为 Dingodb 方言
)

# 补充 Dingodb 向量相关函数（按需扩展）
from pyobvector.schema import (
    l2_distance,
    cosine_distance,
    inner_product,
    negative_inner_product,
    ST_GeomFromText,
    st_distance,
    st_dwithin,
    st_astext,
)
from pyobvector.client.partitions import ObPartition

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DingoClient:
    """The DingoDB Client (对齐 ObClient 接口风格)"""

    def __init__(
            self,
            uri: str = "127.0.0.1:8080",  # Dingodb 默认端口
            user: str = "root",
            password: str = "",
            db_name: str = "test",
            **kwargs,
    ):
        """
        初始化 Dingodb 客户端（同 ObClient 初始化逻辑）
        :param uri: Dingodb 连接地址，格式 host:port
        :param user: 数据库用户名
        :param password: 数据库密码
        :param db_name: 默认数据库名
        :param kwargs: 传递给 create_engine 的额外参数
        """
        # 注册 Dingodb 自定义函数（同 ObClient）
        setattr(func_mod, "l2_distance", l2_distance)
        setattr(func_mod, "cosine_distance", cosine_distance)
        setattr(func_mod, "inner_product", inner_product)
        setattr(func_mod, "negative_inner_product", negative_inner_product)
        setattr(func_mod, "ST_GeomFromText", ST_GeomFromText)
        setattr(func_mod, "st_distance", st_distance)
        setattr(func_mod, "st_dwithin", st_dwithin)
        setattr(func_mod, "st_astext", st_astext)

        # 编码用户名/密码，构建连接串
        user = quote(user, safe="")
        password = quote(password, safe="")
        connection_str = (
            f"mysql+dingodb://{user}:{password}@{uri}/{db_name}?charset=utf8mb4"
        )
        self.engine = create_engine(connection_str, **kwargs)
        self.metadata_obj = MetaData()
        self.metadata_obj.reflect(bind=self.engine)

        # 获取 Dingodb 版本信息（按需调整）
        # with self.engine.connect() as conn:
        #     with conn.begin():
        #         res = conn.execute(text("SELECT VERSION() FROM DUAL"))
        #         version = [r[0] for r in res][0]
        #         self.dingo_version = version  # 简化版本处理，可复用 ObVersion

    def refresh_metadata(self, tables: Optional[list[str]] = None):
        """刷新表元数据（同 ObClient）"""
        if tables is not None:
            for table_name in tables:
                if table_name in self.metadata_obj.tables:
                    self.metadata_obj.remove(Table(table_name, self.metadata_obj))
            self.metadata_obj.reflect(
                bind=self.engine, only=tables, extend_existing=True
            )
        else:
            self.metadata_obj.clear()
            self.metadata_obj.reflect(bind=self.engine, extend_existing=True)

    def check_table_exists(self, table_name: str) -> bool:
        """
        检查表是否存在（同 ObClient 逻辑）
        :param table_name: 表名
        :return: 存在返回 True，否则 False
        """
        inspector = inspect(self.engine)
        return inspector.has_table(table_name)

    def create_table(
            self,
            table_name: str,
            columns: List[Column],
            indexes: Optional[List[Index]] = None,
            partitions: Optional[ObPartition] = None,
            **kwargs,
    ):
        """
        创建表（适配 Dingodb 建表语法）
        :param table_name: 表名
        :param columns: 列定义列表（需包含 Dingodb 向量列，如 Column('vec', VectorType(128))）
        :param indexes: 普通索引列表（非向量索引）
        :param partitions: 分区策略（Dingodb 支持则保留，否则忽略）
        :param kwargs: 额外参数
        """
        kwargs.setdefault("extend_existing", True)
        with self.engine.connect() as conn:
            with conn.begin():
                # 构建表对象（复用 ObTable 或自定义 DingodbTable）
                if indexes is not None:
                    table = DingoTable(
                        table_name,
                        self.metadata_obj,
                        *columns,
                        *indexes,
                        **kwargs,
                    )
                else:
                    table = DingoTable(
                        table_name,
                        self.metadata_obj,
                        *columns,
                        **kwargs,
                    )
                # 执行建表
                table.create(self.engine, checkfirst=True)

                # 分区处理（Dingodb 支持则保留，否则注释）
                if partitions is not None:
                    conn.execute(
                        text(f"ALTER TABLE `{table_name}` {partitions.do_compile()}")
                    )
                logger.info(f"Dingodb table {table_name} created successfully")
