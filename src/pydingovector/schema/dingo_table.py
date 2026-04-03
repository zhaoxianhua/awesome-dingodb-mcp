"""ObTable: extension to Table for creating table with vector index."""
from sqlalchemy import Table
from pydingovector.client.vector_index import DingoSchemaGenerator


class DingoTable(Table):
    """A class extends SQLAlchemy Table to do table creation with vector index."""
    def create(self, bind, checkfirst: bool = False) -> None:
        bind._run_ddl_visitor(DingoSchemaGenerator, self, checkfirst=checkfirst)
