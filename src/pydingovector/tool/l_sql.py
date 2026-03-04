import re
from typing import Optional, Tuple

import pymysql  # MySQL连接库
from pydantic import BaseModel
from pymysql.err import OperationalError, ProgrammingError
from safetensors import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class MCPConfig(BaseModel):
    """MCP工具配置类（适配MySQL + 开源模型）"""
    # 开源模型配置
    model_name: str = "/Users/zhaoli/mistralai/Mistral-7B-Instruct-v0.2"  # 可替换为其他模型
    load_in_4bit: bool = True  # 4位量化，降低显存占用
    temperature: float = 0.1
    max_tokens: int = 500

    # MySQL数据库配置
    dbConfig: dict


# ===================== 核心工具类 =====================
class NL2SQLTool:
    def __init__(self, config):
        self.mcpConfig = MCPConfig(
            dbConfig=config,
        )
        # 初始化LLM客户端
        self.tokenizer, self.generator = self._init_open_source_model()
        # 初始化MySQL连接
        self.db_conn = self._init_mysql_connection()

    def _init_open_source_model(self):
        """初始化开源大模型（Mistral-7B-Instruct）"""
        try:
            # 加载Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.mcpConfig.model_name,
                trust_remote_code=True
            )

            # 修复1：根据硬件自动选择兼容的数据类型
            import torch
            # 检查是否支持bfloat16（GPU架构/环境）
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                dtype = torch.bfloat16
            else:
                # 不支持bfloat16时用float16（GPU）/float32（CPU）
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            # 修复2：检查accelerate库是否安装，适配device_map参数
            model_kwargs = {
                "dtype": dtype,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "use_safetensors": False if dtype == torch.float16 else True,
            }

            # 仅当安装了accelerate且有GPU时，才使用device_map
            try:
                import accelerate  # 检查accelerate是否安装
                if torch.cuda.is_available():
                    model_kwargs["device_map"] = "auto"
                else:
                    # CPU环境强制使用cpu设备
                    model_kwargs["device_map"] = "cpu"
            except ImportError:
                # 无accelerate时，手动指定设备（放弃auto分配）
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model_kwargs["device_map"] = device
                print("警告：未安装accelerate，使用手动设备分配（建议安装accelerate优化大模型加载）")

            # 修复3：禁用4位量化（核心！避免依赖bitsandbytes）
            # 注释/删除load_in_4bit参数，或强制设为False
            model = AutoModelForCausalLM.from_pretrained(
                self.mcpConfig.model_name,
                load_in_4bit=False,  # 强制禁用量化，无需bitsandbytes
                **model_kwargs  # 解包参数，兼容有无accelerate的情况
            )

            # 创建文本生成pipeline
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.mcpConfig.max_tokens,
                temperature=self.mcpConfig.temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            return tokenizer, generator
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def _init_mysql_connection(self) -> pymysql.connections.Connection:
        """初始化MySQL数据库连接"""
        try:
            conn = pymysql.connect(
                host=self.mcpConfig.dbConfig.host,
                port=self.mcpConfig.dbConfig.port,
                user=self.mcpConfig.dbConfig.user,
                password=self.mcpConfig.dbConfig.password,
                database=self.mcpConfig.dbConfig.database,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor  # 让查询结果以字典返回
            )
            return conn
        except OperationalError as e:
            raise RuntimeError(f"MySQL连接失败: {str(e)}")

    def _get_db_schema(self) -> str:
        """获取MySQL数据库表结构（用于LLM提示词）"""
        cursor = self.db_conn.cursor()

        # 获取数据库中所有表名
        cursor.execute("SHOW TABLES;")
        tables = [list(table.values())[0] for table in cursor.fetchall()]

        schema_info = []
        for table in tables:
            # 获取表字段详细信息
            cursor.execute(f"DESCRIBE {table};")
            columns = cursor.fetchall()

            column_info = []
            for col in columns:
                # 提取字段名、类型、是否可为空、默认值等关键信息
                field = col['Field']
                dtype = col['Type']
                null = "NOT NULL" if col['Null'] == "NO" else "NULL"
                default = f"DEFAULT {col['Default']}" if col['Default'] else ""
                column_info.append(f"- {field}: {dtype} {null} {default}".strip())

            schema_info.append(f"表名: {table}\n字段:\n" + "\n".join(column_info))

        return "\n\n".join(schema_info)

    def _build_prompt(self, natural_language_query: str) -> str:
        """构建适配开源模型的提示词（遵循Mistral指令格式）"""
        db_schema = self._get_db_schema()

        # Mistral-7B-Instruct的标准指令格式：<s>[INST] 指令 [/INST]
        prompt = f"""
    <s>[INST]
    你是一个专业的MySQL SQL生成助手，需要将用户的自然语言查询转换为有效的MySQL SQL语句。
    数据库类型：MySQL 8.0+
    数据库名称：{self.mcpConfig.dbConfig.database}
    数据库结构：
    {db_schema}

    生成规则：
    1. 只返回纯MySQL SQL语句，不要包含任何解释、说明或额外文本
    2. 严格遵循MySQL语法规范（如使用反引号`包裹表名/字段名，避免关键字冲突）
    3. 只生成SELECT查询语句，禁止生成DELETE/UPDATE/INSERT/DROP等高危操作
    4. 基于提供的表结构生成，不要虚构表或字段
    5. 处理中文字段/表名时使用正确的字符集（utf8mb4）
    6. 对于数值比较、字符串匹配等场景使用MySQL兼容的语法

    用户查询：{natural_language_query}
    [/INST]
    """
        return prompt.strip()

    def _validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """校验MySQL SQL语句合法性"""
        sql = sql.strip()

        # 1. 检查是否为SELECT语句
        if not sql.upper().startswith("SELECT"):
            return False, "仅支持生成SELECT查询语句"

        # 2. 过滤高危操作
        dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE", "RENAME"]
        for keyword in dangerous_keywords:
            if re.search(rf"\b{keyword}\b", sql.upper()):
                return False, f"禁止生成包含{keyword}的SQL语句"

        # 3. MySQL语法校验（通过EXPLAIN验证）
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(f"EXPLAIN {sql}")
            return True, None
        except ProgrammingError as e:
            return False, f"MySQL语法错误: {str(e)}"
        except Exception as e:
            return False, f"SQL校验失败: {str(e)}"

    def generate_sql(self, natural_language_query: str) -> Tuple[Optional[str], Optional[str]]:
        """生成MySQL SQL语句"""
        try:
            prompt = self._build_prompt(natural_language_query)

            # 调用开源模型生成SQL
            response = self.generator(prompt)
            generated_sql = response[0]["generated_text"].split("[/INST]")[-1].strip()

            # 清理可能的多余文本（模型可能输出额外说明）
            # 只保留第一个分号前的SQL语句
            if ";" in generated_sql:
                generated_sql = generated_sql.split(";")[0] + ";"

            # 校验SQL
            is_valid, error_msg = self._validate_sql(generated_sql)
            if not is_valid:
                return None, error_msg

            return generated_sql, None
        except Exception as e:
            return None, f"SQL生成失败: {str(e)}"

