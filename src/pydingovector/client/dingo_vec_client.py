from typing import Optional

from sqlalchemy import Table, Index

from pydingovector.client.dingo_client import DingoClient
from pydingovector.schema.vector_index import VectorIndex


class DingoVecClient(DingoClient):
    """The OceanBase Vector Client"""

    def __init__(
        self,
        uri: str = "127.0.0.1:2881",
        user: str = "root@test",
        password: str = "",
        db_name: str = "test",
        **kwargs,
    ):
        super().__init__(uri, user, password, db_name, **kwargs)

    def create_index(
            self,
            table_name: str,
            is_vec_index: bool,
            index_name: str,
            column_names: list[str],
            vidx_params: Optional[str] = None,
            **kw,
    ):
        """Create common index or vector index.

        Args:
            table_name (string): table name
            is_vec_index (bool): common index or vector index
            index_name (string): index name
            column_names (List[string]): create index on which columns
            vidx_params (Optional[str]): vector index params, for example 'distance=l2, type=hnsw, lib=vsag'
            **kw: additional keyword arguments
        """
        table = Table(table_name, self.metadata_obj, autoload_with=self.engine)
        columns = [table.c[column_name] for column_name in column_names]
        with self.engine.connect() as conn:
            with conn.begin():
                if is_vec_index:
                    vidx = VectorIndex(index_name, *columns, params=vidx_params, **kw)
                    vidx.create(self.engine, checkfirst=True)
                else:
                    idx = Index(index_name, *columns, **kw)
                    idx.create(self.engine, checkfirst=True)