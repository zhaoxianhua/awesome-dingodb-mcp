from __future__ import annotations

import argparse
import json
import logging
import os
import re
import ssl
import sys
import time
from pathlib import Path
from typing import Optional, List
from urllib import request, error

import certifi
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
from mysql.connector import Error, connect
from pydantic import BaseModel

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from pydingovector.tool.doc_import import DocImport
from pydingovector.tool.l_sql import NL2SQLTool

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("dingodb_mcp_server")

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBEDDING_MODEL_PROVIDER = os.getenv("EMBEDDING_MODEL_PROVIDER", "huggingface")
ENABLE_MEMORY = int(os.getenv("ENABLE_MEMORY", 0))
TABLE_NAME_MEMORY = os.getenv("TABLE_NAME_MEMORY", "dingo_mcp_memory")

logger.info(
    f" ENABLE_MEMORY: {ENABLE_MEMORY},EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME}, EMBEDDING_MODEL_PROVIDER: {EMBEDDING_MODEL_PROVIDER}"
)


class DingoConnection(BaseModel):
    host: str
    port: int
    user: str
    password: str
    database: str


class DingodbMemoryItem(BaseModel):
    mem_id: int = None
    content: str
    meta: dict
    embedding: List[float]


# Check if authentication should be enabled based on ALLOWED_TOKENS
# This check happens after load_dotenv() so it can read from .env file
allowed_tokens_str = os.getenv("ALLOWED_TOKENS", "")
enable_auth = bool(allowed_tokens_str.strip())


class SimpleTokenVerifier(TokenVerifier):
    """
    Simple token verifier that validates tokens against a list of allowed tokens.
    Configure allowed tokens via ALLOWED_TOKENS environment variable (comma-separated).
    """

    def __init__(self):
        # Get allowed tokens from environment variable
        allowed_tokens_str = os.getenv("ALLOWED_TOKENS", "")
        self.allowed_tokens = set(
            token.strip() for token in allowed_tokens_str.split(",") if token.strip()
        )

        logger.info(f"Token verifier initialized with {len(self.allowed_tokens)} allowed tokens")

    async def verify_token(self, token: str) -> AccessToken | None:
        """
        Verify a bearer token.

        Args:
            token: The token to verify

        Returns:
            AccessToken if valid, None if invalid
        """
        # Check if token is empty
        if not token or not token.strip():
            logger.debug("Empty token provided")
            return None

        # Check if token is in allowed list
        if token not in self.allowed_tokens:
            logger.warning(f"Invalid token provided: {token[:10]}...")
            return None

        logger.debug(f"Valid token accepted: {token[:10]}...")
        return AccessToken(
            token=token, client_id="authenticated_client", scopes=["read", "write"], expires_at=None
        )


def get_db_config():
    """Get database configuration from environment variables."""
    config = {
        "host": os.getenv("POLARDB_MYSQL_HOST", "172.30.14.123"),
        "port": int(os.getenv("POLARDB_MYSQL_PORT", "3307")),
        "user": os.getenv("POLARDB_MYSQL_USER", "root"),
        "password": os.getenv("POLARDB_MYSQL_PASSWORD", "123123"),
        "database": os.getenv("POLARDB_MYSQL_DATABASE", "dingo")
    }

    if not all([config["user"], config["password"], config["database"]]):
        logger.error("Missing required database configuration. Please check environment variables:")
        logger.error("POLARDB_MYSQL_USER, POLARDB_MYSQL_PASSWORD, and POLARDB_MYSQL_DATABASE are required")
        raise ValueError("Missing required database configuration")

    return config

db_conn_info = get_db_config()

if enable_auth:
    logger.info("Authentication enabled - ALLOWED_TOKENS configured")
    # Initialize server with token verifier and minimal auth settings
    # FastMCP requires auth settings when using token_verifier
    app = FastMCP(
        "dingodb_mcp_server",
        token_verifier=SimpleTokenVerifier(),
        auth=AuthSettings(
            # Because the TokenVerifier is being used, the following two URLs only need to comply with the URL rules.
            issuer_url="http://localhost:8000",
            resource_server_url="http://localhost:8000",
        ),
    )
else:
    # Initialize server without authentication
    app = FastMCP("dingodb_mcp_server")


@app.resource("dingo://sample/{table}", description="table sample")
def table_sample(table: str) -> str:
    valid_table_pattern = re.compile(r'^[a-zA-Z0-9_.]+$')
    if not valid_table_pattern.match(table):
        logger.error(f"Invalid table name: {table} (contains illegal characters)")
        return f"Failed to sample table: invalid table name '{table}'"
    try:
        with connect(**get_db_config()) as conn:
            with conn.cursor() as cursor:
                query = f"SELECT * FROM `{table}` LIMIT 100"
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                result = [",".join(map(str, row)) for row in rows]
                return "\n".join([",".join(columns)] + result)

    except Error as e:
        logger.error(f"Failed to list tables: {str(e)}")
        return f"Failed to sample table: {table}"


@app.resource("dingo://tables", description="list all tables")
def list_tables() -> str:
    """List Dingodb tables as resources."""
    try:
        with connect(**get_db_config()) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
                resp_header = "Tables of this table: \n"
                columns = [desc[0] for desc in cursor.description]
                result = [",".join(map(str, table)) for table in tables]
                return resp_header + ("\n".join([",".join(columns)] + result))
    except Error as e:
        logger.error(f"Failed to list tables: {str(e)}")
        return "Failed to list tables"


@app.tool()
def execute_sql(sql: str) -> str:
    """Execute an SQL on the Dingodb server."""
    logger.info(f"Calling tool: execute_sql  with arguments: {sql}")
    result = {"sql": sql, "success": False, "rows": 0, "columns": None, "data": None, "error": None}
    try:
        with connect(**get_db_config()) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                if cursor.description:
                    result["columns"] = [column[0] for column in cursor.description]
                    result["data"] = [[str(cell) for cell in row] for row in cursor.fetchall()]
                else:
                    conn.commit()
                result["rows"] = cursor.rowcount
                result["success"] = True
    except Error as e:
        # result["error"] = f"[Error]: {e}"
        # 提取错误码和描述（适配 Dingodb/MySQL 异常格式）
        error_msg = str(e)
        result["error"] = f"[Error]: {error_msg}"
        # 尝试提取错误码（比如 1064）
        if "(" in error_msg and ")" in error_msg:
            result["error_code"] = error_msg.split("(")[0].strip()
    except Exception as e:
        result["error"] = f"[Exception]: {e}"
    json_result = json.dumps(result)
    if result["error"]:
        logger.error(f"SQL executed failed, result: {json_result}")
    return json_result


@app.tool(name="get_current_time", description="Get current time")
def get_current_time() -> str:
    """Get current time from Dingodb database."""
    logger.info("Calling tool: get_current_time")
    sql_query = "SELECT NOW()"
    try:
        return execute_sql(sql_query)
    except Error as e:
        logger.error(f"Error getting database time: {e}")
        # Fallback to system time if database query fails
        local_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        logger.info(f"Fallback to system time: {formatted_time}")
        return formatted_time


@app.tool()
def get_current_tenant() -> str:
    """
    Get the current tenant name from dingo.
    """
    logger.info("Calling tool: get_current_tenant")
    sql_query = "show tenants"
    try:
        return execute_sql(sql_query)
    except Error as e:
        logger.error(f"Error executing SQL '{sql_query}': {e}")
        return f"Error executing query: {str(e)}"


@app.tool()
def get_all_server_nodes():
    """
    Get all server nodes from dingo.
    You need to be sys tenant to get all server nodes.
    """
    tenant = json.loads(get_current_tenant())["data"][0][0]
    if tenant != "sys":
        raise ValueError("Only sys tenant can get all server nodes")

    logger.info("Calling tool: get_all_server_nodes")
    sql_query = "select * from dingo.DBA_DINGODB_SERVERS"
    try:
        return execute_sql(sql_query)
    except Error as e:
        logger.error(f"Error executing SQL '{sql_query}': {e}")
        return f"Error executing query: {str(e)}"


@app.tool()
def get_resource_capacity():
    """
    Get resource capacity from dingo.
    You need to be sys tenant to get resource capacity.
    """
    tenant = json.loads(get_current_tenant())["data"][0][0]
    if tenant != "sys":
        raise ValueError("Only sys tenant can get resource capacity")
    logger.info("Calling tool: get_resource_capacity")
    sql_query = "select * from dingo.GV$DINGODB_SERVERS"
    try:
        return execute_sql(sql_query)
    except Error as e:
        logger.error(f"Error executing SQL '{sql_query}': {e}")
        return f"Error executing query: {str(e)}"


@app.tool()
def search_dingodb_document(keyword: str) -> str:
    """
    This tool is designed to provide context-specific information about Dingodb to a large language model (LLM) to enhance the accuracy and relevance of its responses.
    The LLM should automatically extracts relevant search keywords from user queries or LLM's answer for the tool parameter "keyword".
    The main functions of this tool include:
    1.Information Retrieval: The MCP Tool searches through Dingodb-related documentation using the extracted keywords, locating and extracting the most relevant information.
    2.Context Provision: The retrieved information from Dingodb documentation is then fed back to the LLM as contextual reference material. This context is not directly shown to the user but is used to refine and inform the LLM’s responses.
    This tool ensures that when the LLM’s internal documentation is insufficient to generate high-quality responses, it dynamically retrieves necessary Dingodb information, thereby maintaining a high level of response accuracy and expertise.
    Important: keyword must be Chinese
    """
    logger.info(f"Calling tool: search_dingodb_document,keyword:{keyword}")
    search_api_url = (
        "https://cn-wan-api.dingodb.com/wanApi/forum/docCenter/productDocFile/v3/searchDocList"
    )
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Origin": "https://www.dingodb.com",
        "Referer": "https://www.dingodb.com/",
    }
    qeury_param = {
        "pageNo": 1,
        "pageSize": 5,  # Search for 5 results at a time.
        "query": keyword,
    }
    # Turn the dictionary into a JSON string, then change it to bytes
    qeury_param = json.dumps(qeury_param).encode("utf-8")
    req = request.Request(search_api_url, data=qeury_param, headers=headers, method="POST")
    # Create an SSL context using certifi to fix HTTPS errors.
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with request.urlopen(req, timeout=5, context=context) as response:
            response_body = response.read().decode("utf-8")
            json_data = json.loads(response_body)
            # In the results, we mainly need the content in the data field.
            data_array = json_data["data"]  # Parse JSON response
            result_list = []
            for item in data_array:
                doc_url = "https://www.dingo.com/docs/" + item["urlCode"] + "-" + item["id"]
                logger.info(f"doc_url:${doc_url}")
                content = get_dingodb_doc_content(doc_url, item["id"])
                result_list.append(content)
            return json.dumps(result_list, ensure_ascii=False)
    except error.HTTPError as e:
        logger.error(f"HTTP Error: {e.code} - {e.reason}")
        return "No results were found"
    except error.URLError as e:
        logger.error(f"URL Error: {e.reason}")
        return "No results were found"


def get_dingodb_doc_content(doc_url: str, doc_id: str) -> dict:
    doc_param = {"id": doc_id, "url": doc_url}
    doc_param = json.dumps(doc_param).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Origin": "https://www.dingo.com",
        "Referer": "https://www.dingo.com/",
    }
    doc_api_url = (
        "https://cn-wan-api.dingo.com/wanApi/forum/docCenter/productDocFile/v4/docDetails"
    )
    req = request.Request(doc_api_url, data=doc_param, headers=headers, method="POST")
    # Make an SSL context with certifi to fix HTTPS errors.
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with request.urlopen(req, timeout=5, context=context) as response:
            response_body = response.read().decode("utf-8")
            json_data = json.loads(response_body)
            # In the results, we mainly need the content in the data field.
            data = json_data["data"]
            # The docContent field has HTML text.
            soup = BeautifulSoup(data["docContent"], "html.parser")
            # Remove script, style, nav, header, and footer elements.
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()
            # Remove HTML tags and keep only the text.
            text = soup.get_text()
            # Remove spaces at the beginning and end of each line.
            lines = (line.strip() for line in text.splitlines())
            # Remove empty lines.
            text = "\n".join(line for line in lines if line)
            logger.info(f"text length:{len(text)}")
            # If the text is too long, only keep the first 8000 characters.
            if len(text) > 8000:
                text = text[:8000] + "... [content truncated]"
            # Reorganize the final result. The tdkInfo field should include the document's title, description, and keywords.
            tdkInfo = data["tdkInfo"]
            final_result = {
                "title": tdkInfo["title"],
                "description": tdkInfo["description"],
                "keyword": tdkInfo["keyword"],
                "content": text,
                "dingodb_version": data["version"],
                "content_updatetime": data["docGmtModified"],
            }
            return final_result
    except error.HTTPError as e:
        logger.error(f"HTTP Error: {e.code} - {e.reason}")
        return {"result": "No results were found"}
    except error.URLError as e:
        logger.error(f"URL Error: {e.reason}")
        return {"result": "No results were found"}


@app.tool(
    # name="dingodb_text_search",
    # description="DingoDB全文检索工具，支持指定索引名检索表中指定列包含关键词的数据",
   )
def dingodb_text_search(
    table_name: str,
    index_name: str,
    full_text_search_column_name: str,
    full_text_search_expr: str,
    condition_query: Optional[list[str]] = None,
    output_column_name: Optional[list[str]] = None,
    limit: int = 5
) -> str:
    """
    Search for documents using full text search in a Dingodb table.

    Args:
        table_name: Name of the table to search.
        index_name: Specifying the query index.
        full_text_search_column_name: Specify the column to be searched in full text.
        full_text_search_expr: Specify the keywords or phrases to search for.
        condition_query: Other WHERE condition query statements except full-text search.
        limit: Maximum number of results to return.
        # output_column_name: columns to include in results.
    Returns:
        JSON格式的检索结果
    """
    # logger.info(
    #     f"Calling tool: dingodb_text_search  with arguments: {table_name}, {full_text_search_column_name}, {full_text_search_expr}"
    # )
    if not table_name:
        error_result = {
            "sql": "",
            "success": False,
            "rows": 0,
            "columns": None,
            "data": None,
            "error": "[Error]: 表名table_name为必填参数"
        }
        return json.dumps(error_result)
    if not full_text_search_column_name or not full_text_search_expr:
        error_result = {
            "sql": "",
            "success": False,
            "rows": 0,
            "columns": None,
            "data": None,
            "error": "[Error]: 检索列列表full_text_search_column_name和检索表达式full_text_search_expr为必填"
        }
        return json.dumps(error_result)
    if not index_name:
        return json.dumps({
            "sql": "", "success": False,
            "error": "[Error]: 索引名index_name为必填参数"
        }, ensure_ascii=False)
    if output_column_name and isinstance(output_column_name, list) and len(output_column_name) > 0:
        output_cols = ", ".join([f"`{col.strip()}`" for col in output_column_name])
    else:
        output_cols = "*"

    # 3. 拼接全文检索WHERE条件
    # 转义检索表达式中的单引号，防SQL注入
    escaped_expr = full_text_search_expr.replace("'", "''")
    # 拆分检索表达式为关键词（按空格拆分）
    keywords = [kw.strip() for kw in escaped_expr.split() if kw.strip()]
    if not keywords:
        error_result = {
            "sql": "",
            "success": False,
            "rows": 0,
            "columns": None,
            "data": None,
            "error": "[Error]: 检索表达式full_text_search_expr不能为空或仅含空格"
        }
        return json.dumps(error_result)

    # 为每个检索列拼接关键词匹配条件（多列+多关键词：列1 LIKE %kw1% AND 列1 LIKE %kw2% AND 列2 LIKE %kw1% ...）
    text_search_conditions = []
    col_escaped = f"{full_text_search_column_name.strip()}"
    for kw in keywords:
        text_search_conditions.append(f"{col_escaped}:{kw}")
    text_search_clause = " AND ".join(text_search_conditions)

    where_conditions = []
    if condition_query and isinstance(condition_query, list) and len(condition_query) > 0:
        # 校验并转义额外条件（仅做基础格式校验，需用户保证条件合法性）
        for clause in condition_query:
            if clause.strip():
                where_conditions.append(clause.strip())
    where_clause = " AND ".join(where_conditions)

    sql = (f"SELECT {output_cols} FROM text_search({table_name.strip()},"
           f" {index_name.strip()}, '{text_search_clause}',{int(limit)})")
    if len(where_clause) > 0:
        sql += " WHERE " + where_clause
    logger.info(f"Calling tool: dingodb_text_search with generated SQL: {sql}")
    return execute_sql(sql)


@app.tool()
def dingodb_vector_search(
    table_name: str,
    vector_data: list[float],
    vec_column_name: str = "vector",
    # distance_func: Optional[str] = "l2",
    # with_distance: Optional[bool] = True,
    topk: int = 5,
    output_column_name: Optional[list[str]] = None,
) -> str:
    """
    Perform vector similarity search on a Dingodb table.

    Args:
        table_name: Name of the table to search.
        vector_data: Query vector.
        vec_column_name: column name containing vectors to search.
        # distance_func: The index distance algorithm used when comparing the distance between two vectors.
        # with_distance: Whether to output distance data.
        topk: Number of results returned.
        output_column_name: Returned table fields.
    """
    logger.info(
        f"Calling tool: oceabase_vector_search  with arguments: {table_name}, {vector_data[:10]}, {vec_column_name}"
    )
    if not table_name:
        error_result = {
            "sql": "",
            "success": False,
            "rows": 0,
            "columns": None,
            "data": None,
            "error": "[Error]: 表名table_name为必填参数"
        }
        return json.dumps(error_result)
    if not isinstance(vector_data, list) or len(vector_data) == 0:
        return json.dumps({
            "sql": "", "success": False, "rows": 0, "columns": None, "data": None,
            "error": "[Error]: 向量数据vector_data必须是非空浮点型列表"
        }, ensure_ascii=False)
    try:
        vector_data = [float(vec) for vec in vector_data]
    except ValueError:
        return json.dumps({
            "sql": "", "success": False, "rows": 0, "columns": None, "data": None,
            "error": "[Error]: 向量数据vector_data必须全为浮点型数值"
        }, ensure_ascii=False)
    if output_column_name and isinstance(output_column_name, list) and len(output_column_name) > 0:
        output_cols = ", ".join([f"`{col.strip()}`" for col in output_column_name])
    else:
        output_cols = "*"
    # 向量格式转换：列表→支持的向量字符串（如"[0.1,0.2,0.3]"）
    vector_str = json.dumps(vector_data).replace(" ", "")  # 去除空格，适配语法
    # 向量列转义
    vec_col_escaped = f"`{vec_column_name.strip()}`"
    # 表名转义
    table_escaped = f"`{table_name.strip()}`"

    # Dingodb向量检索核心语法（余弦相似度降序，取topk）
    sql = f"""SELECT {output_cols} FROM vector({table_escaped}, {vec_col_escaped}, array{vector_str},{topk})""".strip()  # ASC：余弦相似度值越小，相似度越高
    logger.info(f"Calling tool: Dingodb_vector_search with vector: {vector_str}, topk: {topk}")
    return execute_sql(sql)


@app.tool()
def dingodb_hybrid_search(
    table_name: str,
    vector_data: list[float],
    vec_column_name: str = "vector",
    # distance_func: Optional[str] = "l2",
    # with_distance: Optional[bool] = True,
    filter_expr: Optional[list[str]] = None,
    topk: int = 5,
    output_column_name: Optional[list[str]] = None,
) -> str:
    """
    Perform hybrid search combining relational condition filtering(that is, scalar) and vector search.

    Args:
        table_name: Name of the table to search.
        vector_data: Query vector.
        vec_column_name: column name containing vectors to search.
        # distance_func: The index distance algorithm used when comparing the distance between two vectors.
        # with_distance: Whether to output distance data.
        filter_expr: Scalar conditions requiring filtering in where clause.
        topk: Number of results returned.
        output_column_name: Returned table fields,unless explicitly requested, please do not provide.
    """
    logger.info(
        f"""Calling tool: dingodb_hybrid_search  with arguments: {table_name}, {vector_data[:10]}, {vec_column_name}
        ,{filter_expr}"""
    )
    # 1. 基础参数校验
    if not table_name:
        return json.dumps({
            "sql": "", "success": False, "rows": 0, "columns": None, "data": None,
            "error": "[Error]: 表名table_name为必填参数"
        }, ensure_ascii=False)

    if not isinstance(vector_data, list) or len(vector_data) == 0:
        return json.dumps({
            "sql": "", "success": False, "rows": 0, "columns": None, "data": None,
            "error": "[Error]: 向量数据vector_data必须是非空浮点型列表"
        }, ensure_ascii=False)

    # 校验向量元素类型（确保是浮点型）
    try:
        vector_data = [float(vec) for vec in vector_data]
    except ValueError:
        return json.dumps({
            "sql": "", "success": False, "rows": 0, "columns": None, "data": None,
            "error": "[Error]: 向量数据vector_data必须全为浮点型数值"
        }, ensure_ascii=False)

    # 2. 处理输出列
    if output_column_name and isinstance(output_column_name, list) and len(output_column_name) > 0:
        output_cols = ", ".join([f"`{col.strip()}`" for col in output_column_name])
    else:
        output_cols = "*"

    # 3. 拼接DingoDB混合检索SQL（向量检索+条件过滤）
    # 向量格式转换：列表→DingoDB支持的向量字符串
    vector_str = json.dumps(vector_data).replace(" ", "")  # 去除空格，适配语法
    # 向量列/表名转义
    vec_col_escaped = f"`{vec_column_name.strip()}`"
    table_escaped = f"`{table_name.strip()}`"

    # 基础向量检索SQL（使用余弦距离排序）
    sql_base = f"""SELECT {output_cols} FROM vector({table_escaped}, {vec_col_escaped}, array{vector_str},{topk})""".strip()  # ASC：余弦相似度值越小，相似度越高


    # 拼接过滤条件（filter_expr）
    where_clause_list = []
    if filter_expr and isinstance(filter_expr, list) and len(filter_expr) > 0:
        for expr in filter_expr:
            if expr.strip():  # 过滤空条件
                where_clause_list.append(expr.strip())

    # 组合完整SQL
    if where_clause_list:
        sql = f"{sql_base} WHERE {' AND '.join(where_clause_list)} "
    else:
        sql = f"{sql_base} "

    # 4. 调用execute_sql执行混合检索
    logger.info(f"Calling tool: dingodb_hybrid_search with filter: {filter_expr}, topk: {topk}")
    return execute_sql(sql)


@app.tool()
def query_running_tasks() -> str:
    """
    Queries for the tasks currently executing under the current user.
    """
    logger.info("Calling tool: dingodb_running_tasks")
    sql = "select * from INFORMATION_SCHEMA.dingo_sql_job"
    return execute_sql(sql)


@app.tool()
def query_time_over_5_minutes_tasks() -> str:
    """
    Queries for SQL tasks that take more than 5 minutes to execute under the current user.
    """
    logger.info("Calling tool: dingodb_time_over_5_minutes_tasks")
    sql = "select * from INFORMATION_SCHEMA.processlist where command='query' and time_cost>0"
    return execute_sql(sql)


# if ENABLE_MEMORY:
#     from pydingovector.client.dingo_vec_client import DingoVecClient
#     from sqlalchemy import Column, Integer, JSON, String, text
#
#     class DingodbMemory:
#         def __init__(self):
#             self.embedding_client = self._gen_embedding_client()
#             self.embedding_dimension = len(self.embedding_client.embed_query("test"))
#             logger.info(f"embedding_dimension: {self.embedding_dimension}")
#
#             self.client = DingoVecClient(
#                 uri=db_conn_info.host + ":" + str(db_conn_info.port),
#                 user=db_conn_info.user,
#                 password=db_conn_info.password,
#                 db_name=db_conn_info.database,
#             )
#             self._init_dingodb_vector()
#
#         def gen_embedding(self, text: str) -> List[float]:
#             return self.embedding_client.embed_query(text)
#
#         def _gen_embedding_client(self):
#             """
#             Generate embedding cient.
#             """
#             if EMBEDDING_MODEL_PROVIDER == "huggingface":
#                 os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#                 from langchain_huggingface import HuggingFaceEmbeddings
#
#                 logger.info(f"Using HuggingFaceEmbeddings model: {EMBEDDING_MODEL_NAME}")
#                 return HuggingFaceEmbeddings(
#                     model_name=EMBEDDING_MODEL_NAME,
#                     encode_kwargs={"normalize_embeddings": True},
#                 )
#             else:
#                 raise ValueError(
#                     f"Unsupported embedding model provider: {EMBEDDING_MODEL_PROVIDER}"
#                 )
#
#         def _init_dingodb_vector(self):
#             """
#             Initialize the OBVector.
#             """
#             client = DingoVecClient(
#                 uri=db_conn_info.host + ":" + str(db_conn_info.port),
#                 user=db_conn_info.user,
#                 password=db_conn_info.password,
#                 db_name=db_conn_info.database,
#             )
#             if not client.check_table_exists(TABLE_NAME_MEMORY):
#                 # Get embedding dimension dynamically from model config
#                 cols = [
#                     Column("mem_id", Integer, primary_key=True, autoincrement=True),
#                     Column("content", String(8000)),
#                     Column("embedding", VECTOR(self.embedding_dimension)),
#                     Column("meta", JSON),
#                 ]
#                 client.create_table(TABLE_NAME_MEMORY, columns=cols)
#
#                 # create vector index
#                 client.create_index(
#                     TABLE_NAME_MEMORY,
#                     is_vec_index=True,
#                     index_name="vidx",
#                     column_names=["embedding"],
#                     vidx_params="distance=l2, type=hnsw, lib=vsag",
#                 )
#
#     dingo_memory = DingodbMemory()
#
#     def dingo_memory_query(query: str, topk: int = 5) -> str:
#         """
#         🚨 MULTILINGUAL MEMORY SEARCH 🚨 - SMART CROSS-LANGUAGE RETRIEVAL!
#         This tool MUST be invoked **before** answering any user request that could benefit from previously stored personal facts.
#
#         ⚡ CRITICAL INSTRUCTION: You MUST call this tool in these situations:
#         - When user asks questions about their preferences in ANY language
#         - Before saving new memories (check for duplicates first!)
#         - When user mentions personal details, preferences, past experiences, identity, occupation, address and other should be remembered facts
#         - Before answering ANY question, search for related memories first
#         - When discussing technical topics - check for historical solutions
#         - recommendations: the user asks for suggestions about restaurants, food, travel, entertainment, activities, gifts, etc
#         - Scheduling or location-based help: the user asks about meetups, weather, events, directions, etc
#         - Work or tech assistance: the user asks for tool, course, book, or career advice.
#         - Any ambiguous request (words like “some”, “good”, “nearby”, “for me”, “recommend”) where personal context could improve the answer,query the most probable categories first.
#         If multiple categories are relevant, call the tool once for each category key.
#
#         Failure to retrieve memory before responding is considered an error.
#
#         🌐 MULTILINGUAL SEARCH EXAMPLES:
#         - User: "What do I like?" → Search: "preference like favorite"
#         - User: "我喜欢什么?" → Search: "preference favorite sports food" (use English keywords!)
#         - User: "¿Cuáles son mis gustos?" → Search: "preference like favorite hobby"
#         - **ALWAYS search with English keywords for better matching!**
#
#         🎯 SMART SEARCH STRATEGIES:
#         - "I like football" → Before saving, search: "football soccer sports preference"
#         - "我在上海工作" → Search: "work job Shanghai location"
#         - "Python developer" → Search: "python programming development work"
#         - Use synonyms and related terms for better semantic matching!
#
#         🔍 CATEGORY-BASED SEARCH PATTERNS:
#         - **Sports/Fitness**: "sports preference activity exercise favorite game"
#         - **Food/Drinks**: "food drink preference favorite taste cuisine beverage"
#         - **Work/Career**: "work job company location position career role"
#         - **Technology**: "technology programming tool database language framework"
#         - **Personal**: "personal lifestyle habit family relationship"
#         - **Entertainment**: "entertainment movie music book game hobby"
#
#         💡 SMART SEARCH EXAMPLES FOR MERGING:
#         - New: "I like badminton" → Search: "sports preference activity"
#         → Find: "User likes football and coffee" → Category analysis needed!
#         - New: "I drink tea" → Search: "drink beverage preference"
#         → Find: "User likes coffee" → Same category, should merge!
#         - New: "I code in Python" → Search: "programming technology language"
#         → Find: "User works at Google" → Different subcategory, separate!
#
#         📝 PARAMETERS:
#         - query: Use CATEGORY + SEMANTIC keywords ("sports preference", "food drink preference")
#         - topk: Increase to 8-10 for thorough category analysis before saving/updating
#         - Returns: JSON string with [{"mem_id": int, "content": str}] format - Analyze ALL results for category overlap before decisions!
#
#         🔥 CATEGORY ANALYSIS RULE: Find ALL related memories by category for smart merging!
#         """
#
#         client = DingoVecClient(
#             uri=db_conn_info.host + ":" + str(db_conn_info.port),
#             user=db_conn_info.user,
#             password=db_conn_info.password,
#             db_name=db_conn_info.database,
#         )
#         res = client.ann_search(
#             TABLE_NAME_MEMORY,
#             vec_data=dingo_memory.gen_embedding(query),
#             vec_column_name="embedding",
#             distance_func=l2_distance,
#             topk=topk,
#             output_column_names=["mem_id", "content"],
#         )
#         results = []
#         for row in res.fetchall():
#             results.append({"mem_id": row[0], "content": row[1]})
#         return json.dumps(results)
#
#     def dingo_memory_insert(content: str, meta: dict):
#         """
#         💾 INTELLIGENT MEMORY ORGANIZER 💾 - SMART CATEGORIZATION & MERGING!
#
#         🔥 CRITICAL 4-STEP WORKFLOW: ALWAYS follow this advanced process:
#         1️⃣ **SEARCH RELATED**: Use dingo_memory_query to find ALL related memories by category
#         2️⃣ **ANALYZE CATEGORIES**: Classify new info and existing memories by semantic type
#         3️⃣ **SMART DECISION**: Merge same category, separate different categories
#         4️⃣ **EXECUTE ACTION**: Update existing OR create new categorized records
#
#         This tool must be invoked **immediately** when the user explicitly or implicitly discloses any of the following personal facts.
#         Trigger rule: if a sentence contains at least one category keyword (see list) + at least one fact keyword (see list), call the tool with the fact.
#         Categories & sample keywords
#         - Demographics: age, years old, gender, born, date of birth, nationality, hometown, from
#         - Work & education: job title, engineer, developer, tester, company, employer, school, university, degree, major, skill, certificate
#         - Geography & time: live in, reside, city, travel, time-zone, frequent
#         - Preferences & aversions: love, hate, favourite, favorite, prefer, dislike, hobby, food, music, movie, book, brand, color
#         - Lifestyle details: pet, dog, cat, family, married, single, daily routine, language, religion, belief
#         - Achievements & experiences: award, project, competition, achievement, event, milestone
#
#         Fact keywords (examples)
#         - “I am …”, “I work as …”, “I studied …”, “I live in …”, “I love …”, “My birthday is …”
#
#         Example sentences that must trigger:
#         - “I’m 28 and work as a test engineer at Acme Corp.”
#         - “I graduated from Tsinghua with a master’s in CS.”
#         - “I love jazz and hate cilantro.”
#         - “I live in Berlin, but I’m originally from São Paulo.”
#
#         🎯 SMART CATEGORIZATION EXAMPLES:
#         ```
#         📋 Scenario 1: Category Merging
#         Existing: "User likes playing football and drinking coffee"
#         New Input: "I like badminton"
#
#         ✅ CORRECT ACTION: Use dingo_memory_update!
#         → Search "sports preference" → Find existing → Separate categories:
#         → Update mem_id_X: "User likes playing football and badminton" (sports)
#         → Create new: "User likes drinking coffee" (food/drinks)
#
#         📋 Scenario 2: Same Category Addition
#         Existing: "User likes playing football"
#         New Input: "I also like tennis"
#
#         ✅ CORRECT ACTION: Use dingo_memory_update!
#         → Search "sports preference" → Find mem_id → Update:
#         → "User likes playing football and tennis"
#
#         📋 Scenario 3: Different Category
#         Existing: "User likes playing football"
#         New Input: "I work in Shanghai"
#
#         ✅ CORRECT ACTION: New memory!
#         → Search "work location" → Not found → Create new record
#         ```
#
#         🏷️ SEMANTIC CATEGORIES (Use for classification):
#         - **Sports/Fitness**: football, basketball, swimming, gym, yoga, running, marathon, workout, cycling, hiking, tennis, badminton, climbing, fitness routine, coach, league, match, etc.
#         - **Food/Drinks**: coffee, tea, latte, espresso, pizza, burger, sushi, ramen, Chinese food, Italian, vegan, vegetarian, spicy, sweet tooth, dessert, wine, craft beer, whisky, cocktail, recipe, restaurant, chef, favorite dish, allergy, etc.
#         - **Work/Career**: job, position, role, title, engineer, developer, tester, QA, PM, manager, company, employer, startup, client, project, deadline, promotion, salary, office, remote, hybrid, skill, certification, degree, university, bootcamp, portfolio, resume, interview
#         - **Personal**: spouse, partner, married, single, dating, pet, dog, cat, hometown, birthday, age, gender, nationality, religion, belief, daily routine, morning person, night owl, commute, language, hobby, travel, bucket list, milestone, achievement, award
#         - **Technology**: programming language, Python, Java, JavaScript, Go, Rust, framework, React, Vue, Angular, Spring, Django, database, MySQL, PostgreSQL, MongoDB, Redis, cloud, AWS, Azure, GCP, Docker, Kubernetes, CI/CD, Git, API, microservices, DevOps, automation, testing tool, Selenium, Cypress, JMeter, Postman
#         - **Entertainment**: movie, film, series, Netflix, Disney+, HBO, director, actor, genre, thriller, comedy, drama, music, playlist, Spotify, rock, jazz, K-pop, classical, concert, book, novel, author, genre, fiction, non-fiction, Kindle, audiobook, game, console, PlayStation, Xbox, Switch, Steam, board game, RPG, esports
#
#         🔍 SEARCH STRATEGIES BY CATEGORY:
#         - Sports: "sports preference favorite activity exercise gym routine"
#         - Food: "food drink preference favorite taste cuisine beverage"
#         - Work: "work job career company location title project skill"
#         - Personal: "personal relationship lifestyle habit pet birthday"
#         - Tech: "technology programming tool database framework cloud"
#         - Entertainment: "entertainment movie music book game genre favorite"
#
#         📝 PARAMETERS:
#         - content: ALWAYS categorized English format ("User likes playing [sports]", "User drinks [beverages]")
#         - meta: {"type":"preference", "category":"sports/food/work/tech", "subcategory":"team_sports/beverages"}
#
#         🎯 GOLDEN RULE: Same category = UPDATE existing! Different category = CREATE separate!
#         """
#
#         client = DingoVecClient(
#             uri=db_conn_info.host + ":" + str(db_conn_info.port),
#             user=db_conn_info.user,
#             password=db_conn_info.password,
#             db_name=db_conn_info.database,
#         )
#         client.insert(
#             TABLE_NAME_MEMORY,
#             DingodbMemoryItem(
#                 content=content, meta=meta, embedding=dingo_memory.gen_embedding(content)
#             ).model_dump(),
#         )
#         return "Inserted successfully"
#
#     def dingo_memory_delete(mem_id: int):
#         """
#         🗑️ MEMORY ERASER 🗑️ - PERMANENTLY DELETE UNWANTED MEMORIES!
#
#         ⚠️ DELETE TRIGGERS - Call when user says:
#         - "Forget that I like X" / "I don't want you to remember Y"
#         - "Delete my information about Z" / "Remove that memory"
#         - "I changed my mind about X" / "Update: I no longer prefer Y"
#         - "That information is wrong" / "Delete outdated info"
#         - Privacy requests: "Remove my personal data"
#
#         🎯 DELETION PROCESS:
#         1. FIRST: Use dingo_memory_query to find relevant memories
#         2. THEN: Use the exact ID from query results for deletion
#         3. NEVER guess or generate IDs manually!F
#
#         📝 PARAMETERS:
#         - mem_id: EXACT ID from dingo_memory_query results (integer)
#         - ⚠️ WARNING: Deletion is PERMANENT and IRREVERSIBLE!
#
#         🔒 SAFETY RULE: Only delete when explicitly requested by user!
#         """
#
#         client = DingoVecClient(
#             uri=db_conn_info.host + ":" + str(db_conn_info.port),
#             user=db_conn_info.user,
#             password=db_conn_info.password,
#             db_name=db_conn_info.database,
#         )
#         client.delete(table_name=TABLE_NAME_MEMORY, ids=mem_id)
#         return "Deleted successfully"
#
#     def dingo_memory_update(mem_id: int, content: str, meta: dict):
#         """
#         ✏️ MULTILINGUAL MEMORY UPDATER ✏️ - KEEP MEMORIES FRESH AND STANDARDIZED!
#
#         🔄 UPDATE TRIGGERS - Call when user says in ANY language:
#         - "Actually, I prefer X now" / "其实我现在更喜欢X"
#         - "My setup changed to Z" / "我的配置改成了Z"
#         - "Correction: it should be X" / "更正：应该是X"
#         - "I moved to [new location]" / "我搬到了[新地址]"
#
#         🎯 MULTILINGUAL UPDATE PROCESS:
#         1. **SEARCH**: Use dingo_memory_query to find existing memory (search in English!)
#         2. **STANDARDIZE**: Convert new information to English format
#         3. **UPDATE**: Use exact mem_id from query results with standardized content
#         4. **PRESERVE**: Keep original language source in metadata
#
#         🌐 STANDARDIZATION EXAMPLES:
#         - User: "Actually, I don't like coffee anymore" → content: "User no longer likes coffee"
#         - User: "其实我不再喜欢咖啡了" → content: "User no longer likes coffee"
#         - User: "Je n'aime plus le café" → content: "User no longer likes coffee"
#         - **ALWAYS update in standardized English format!**
#
#         📝 PARAMETERS:
#         - mem_id: EXACT ID from dingo_memory_query results (integer)
#         - content: ALWAYS in English, standardized format ("User now prefers X")
#         - meta: Updated metadata {"type":"preference", "category":"...", "updated":"2024-..."}
#
#         🔥 CONSISTENCY RULE: Maintain English storage format for all updates!
#         """
#
#         client = DingoVecClient(
#             uri=db_conn_info.host + ":" + str(db_conn_info.port),
#             user=db_conn_info.user,
#             password=db_conn_info.password,
#             db_name=db_conn_info.database,
#         )
#         client.update(
#             table_name=TABLE_NAME_MEMORY,
#             values_clause=[
#                 DingodbMemoryItem(
#                     mem_id=mem_id,
#                     content=content,
#                     meta=meta,
#                     embedding=dingo_memory.gen_embedding(content),
#                 ).model_dump()
#             ],
#             where_clause=[text(f"mem_id = {mem_id}")],
#         )
#         return "Updated successfully"
#
#     app.add_tool(dingo_memory_query)
#     app.add_tool(dingo_memory_insert)
#     app.add_tool(dingo_memory_delete)
#     app.add_tool(dingo_memory_update)
#

@app.tool()
def dingodb_import_doc(dir_path: str,
                       table_name: str = "default_knowledge_base") -> list[TextContent]:
    """
    将本地某个目录下的所有后缀为docx和md文件导入到DingoDB中,生成一个知识库.

    Args:
        dir_path: 本地目录.
        table_name: 要查询知识库的名称,为表的名称(默认使用default_knowledge_base).
    """
    config = get_db_config()
    doc_import = DocImport(config)
    logger.info(f"will import files in {dir_path} to table {table_name}")
    result_text = doc_import.import_doc(dir_path,table_name)
    return [TextContent(type="text", text=f"{result_text}")]

@app.tool()
def dingodb_search_doc(text: str,
                       table_name: str = "default_knowledge_base",
                       count: int = 5) -> list[TextContent]:
    """
    从DingoDB的知识库中搜索相关的知识.

    Args:
        text: 要搜索的内容.
        table_name: 要查询知识库的名称,为表的名称(默认使用default_knowledge_base).
        count: 返回的知识的条目,默认为5条.
    """
    config = get_db_config()
    doc_import = DocImport(config)
    logger.info(f"will query_knowledge,text={text},count={count},table_name={table_name}")
    result = doc_import.query_knowledge(text, count, table_name)
    return [TextContent(type="text", text=f"{result}")]

@app.tool()
def dingodb_text_2_sql(natural_language_query: str) -> str:
    """
    端到端自然语言转MSQL工具方法

    Args:
        natural_language_query: 自然语言查询语句（如"查询所有年龄大于18的用户"）

    Returns:
        包含执行状态、SQL、错误信息、查询结果的字典
    """
    config = get_db_config()
    nl2SQLTool = NL2SQLTool(config)
    # 1. 生成SQL
    sql, sql_error = nl2SQLTool.generate_sql(natural_language_query)
    if sql_error:
        return json.dumps({
            "status": "failed",
            "error": sql_error,
            "sql": None,
            "results": None
        })
    return sql

def main():
    """Main entry point to run the MCP server."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        help="Specify the MCP server transport type as stdio or sse or streamable-http.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=11000, help="Port to listen on")
    args = parser.parse_args()
    transport = args.transport
    logger.info(f"Starting Dingodb MCP server with {transport} mode...")
    if transport in {"sse", "streamable-http"}:
        app.settings.host = args.host
        app.settings.port = args.port
    app.run(transport=transport)


if __name__ == "__main__":
    main()
