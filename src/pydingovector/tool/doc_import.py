import logging
import os

import mammoth
from langchain_text_splitters import MarkdownTextSplitter
from mysql.connector import connect, Error
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("dingodb-mysql-mcp-server")
DEFAULT_TABLE_NAME = "default_knowledge_base"




class DocImport:
    def __init__(self, db_config, chunk_size_ = 500, chunk_overlap_=50):
        self.text_splitter = MarkdownTextSplitter(chunk_size=chunk_size_, chunk_overlap=chunk_overlap_)
        self.config = db_config
        self.local_model_path = "/Users/zhaoli/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/"
        self.model = SentenceTransformer(self.local_model_path)
    def split_documents(self,file_path):
        texts = self.text_splitter.split_text(self.doc_to_markdown(file_path))
        return texts
    def _convert_image(self,image):
        return []
    def doc_to_markdown(self,file_path) -> str:
        if file_path.endswith(".docx"):
            with open(file_path, "rb") as doc_file:
                result = mammoth.convert_to_markdown(doc_file, convert_image=self._convert_image)
                markdown_text = result.value
                return markdown_text
        else:
            with open(file_path, 'r', encoding='utf-8') as doc_file:
                contents = doc_file.read()
                return contents

    # def text_to_vect(self, text):
    #     query_sql = f"/*polar4ai*/SELECT * FROM predict(model _polar4ai_text2vec, SELECT '{text}') with();"
    #     rows, ok = self.exec_sql(query_sql)
    #     if ok:
    #         if len(rows) != 1:
    #             logger.error(f"executing SQL '{query_sql}' with more than one rows({len(rows)})")
    #             return ""
    #         else:
    #             vec = rows[0][0]
    #             return vec
    #     else:
    #         return ""

    def text_to_vect(self, text):
        # 1. 空值校验（和原有逻辑一致）
        if not text:
            logger.error("输入文本为空，无法生成向量")
            return ""

        # 2. 模型未加载成功时返回空
        if self.model is None:
            logger.error("本地模型未初始化，无法转向量")
            return ""

        # 3. 本地模型生成向量（替换原有 SQL 逻辑）
        try:
            # 生成向量（返回numpy数组，维度768）
            vec_array = self.model.encode(text, convert_to_numpy=True)
            # 转为字符串（和原有 SQL 返回的向量格式兼容，用逗号分隔）
            vec = ",".join([str(round(x, 6)) for x in vec_array])
            return vec
        except Exception as e:
            logger.error(f"文本转向量失败：{e}")
            return ""

    def exec_sql(self, sql):
        rows=[]
        try:
            with connect(**self.config) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql)
                    if cursor.description is not None:
                        return cursor.fetchall(),True
                    else:
                        conn.commit()
                        return rows,True
        except Error as e:
            logger.error(f"Error executing SQL '{sql}': {e}")
            return rows,False

    def get_all_docxs(self, dir):
        result = []
        files_and_dirs = os.listdir(dir)
        for file_or_dir in files_and_dirs:
            file_path = os.path.join(dir, file_or_dir)
            if os.path.isfile(file_path) and (file_or_dir.endswith('docx') or file_or_dir.endswith('md')):
                result.append((file_path, file_or_dir))
        return result

    def text_deal(self, text):
        text = text.replace('\'','"')
        text = text.replace('\\"', '"')
        text = text.replace('\\','\\\\')
        text = text.replace('\\\'', '"') 
        text = text.replace("/*","/.")
        text = text.replace("*/","./")
        return text
                  
    def import_doc(self, dir, table='') -> str:
        if table == '':
            table_name = DEFAULT_TABLE_NAME
        else:
            table_name = table
        logger.info(f"table_name:{table_name}")
        create_table_sql = f"""
            create table if not exists {table_name}(
            id bigint auto_increment,
            chunk_content varchar, 
            file_name varchar(256),
            vecs float array not null,
            index vecs_index vector(id, vecs) parameters(type=hnsw, metricType=L2, dimension=384, efConstruction=40, nlinks=32),
            primary key(id)
            )comment 'columnar=1';
        """
        rows, ok = self.exec_sql(create_table_sql)
        if not ok:
            logger.error(f"Error creating table '{table_name}'")
            return f"Error creating table '{table_name}'"
        docs = self.get_all_docxs(dir)
        entry_count = 0
        file_count = 0
        for doc in docs:
            file_count += 1
            file_path, file_name = doc
            texts = self.split_documents(file_path)
            for text in texts:
                text = self.text_deal(text)
                vec = self.text_to_vect(text)
                if vec != "":
                    insert_sql = f"""
                        insert into {table_name} (chunk_content, file_name, vecs) values ('{text}', '{file_name}', array[{vec}])
                    """
                    rows, ok = self.exec_sql(insert_sql)
                    if not ok:
                        logger.error(f"Error inserting into table '{table_name}'")
                    else:
                        entry_count += 1
        logger.info(f"success import {entry_count} entries with {file_count} files to table({table_name})")
        return f"success import {entry_count} entries with {file_count} files"
    
    def query_knowledge(self, text: str, count=5, table=''):
        text = self.text_deal(text)
        vec = self.text_to_vect(text)
        if table=='':
            table_name = DEFAULT_TABLE_NAME
        else:
            table_name = table
        query_sql = f"""
                       select file_name,chunk_content,vecs_index$distance from vector({table_name}, vecs, array[{vec}], {count});
                    """
        rows, ok = self.exec_sql(query_sql)
        result = []
        if ok:
            for row in rows:
                result.append({"file_name": row[0], "content": row[1]})
        return result