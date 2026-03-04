from pydingovector.tool.doc_import import DocImport
from .server import execute_sql, list_tables, table_sample, get_current_time, dingodb_text_search, \
    dingodb_vector_search, dingodb_hybrid_search, query_running_tasks, query_time_over_5_minutes_tasks, get_db_config, \
    dingodb_text_2_sql


def test_execute_sql():
    res = execute_sql("SELECT * FROM `dingo`.`dingospeed`")
    print(res)

def test_list_tables():
    res = list_tables()
    print(res)

def test_table_sample():
    res = table_sample("cache_job")
    print(res)

def test_get_current_time():
    res = get_current_time()
    print(res)


def test_dingodb_text_search():
    res = dingodb_text_search("t2", "text_index",["description"], "Ergonomic")
    # res = dingodb_text_search("t2", ["description"], "keyboard", "text_index",["text_id >=4"])
    print(res)

def test_oceabase_vector_search():
    res = dingodb_vector_search("t2",[0.8894774317741394, 0.7277960181236267, 0.692345142364502, 0.47235092520713806, 0.8568729162216187, 0.6647433042526245, 0.3333759307861328, 0.5181455016136169],"feature")
    print(res)

def test_dingodb_hybrid_search():
    res = dingodb_hybrid_search("t2",[0.8894774317741394, 0.7277960181236267, 0.692345142364502, 0.47235092520713806, 0.8568729162216187, 0.6647433042526245, 0.3333759307861328, 0.5181455016136169],"feature", ["rating = 5"])
    print(res)

def test_query_running_tasks():
    res = query_running_tasks()
    print(res)

def test_query_time_over_5_minutes_tasks():
    res = query_time_over_5_minutes_tasks()
    print(res)

def test_doc_import():
    di = DocImport(get_db_config())
    di.import_doc('/Users/zhaoli/Documents/docs/fentai/01.产品/其他资料',"ss_doc")

def test_query_knowledge():
    di = DocImport(get_db_config())
    res = di.query_knowledge("GoLang编码规范", table="ss_doc")
    print(res)

def test_dingodb_text_2_sql():
    res = dingodb_text_2_sql("查询dingospeed表的前10条数据")
    print(res)