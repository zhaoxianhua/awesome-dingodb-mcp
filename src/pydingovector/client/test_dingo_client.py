from sqlalchemy import Column, Integer, String, JSON

from pydingovector.client.dingo_vec_client import DingoVecClient
from pydingovector.schema.vector import VECTOR


def test_check_table_exists():
    # client = DingoVecClient("127.0.0.1:3306", "root", "12345678","dingo")
    client = DingoVecClient("172.30.14.123:3307", "root", "123123","dingo")
    exist = client.check_table_exists("dingospeed")
    print(exist)

def test_create_table():
    client = DingoVecClient("172.30.14.123:3307", "root", "123123","dingo")
    # client = DingoVecClient("127.0.0.1:3306", "root", "12345678","dingo")
    cols = [
                    Column("mem_id", Integer, primary_key=True, autoincrement=True),
                    Column("content", String(8000)),
                    # Column("embedding", VECTOR(self.embedding_dimension)),
                    Column("meta", JSON),
                ]
    exist = client.create_table("dingospeed1", cols)
    print(exist)

def test_create_index():
    client = DingoVecClient("172.30.14.123:3307", "root", "123123", "dingo")
    # client = DingoVecClient("127.0.0.1:3306", "root", "12345678", "dingo")
    client.create_index(table_name="tag",is_vec_index=False,
                    index_name="vidx",
                    column_names=["label"],
                    vidx_params="type=hnsw, metricType=COSINE, dimension=768, efConstruction=200, nlinks=16")  #   vidx_params="distance=l2, type=hnsw, lib=vsag")