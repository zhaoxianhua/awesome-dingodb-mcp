[English](README.md) | 简体中文<br>
# DingoDB MCP Server

DingoDB MCP Server 通过 MCP (模型上下文协议) 可以和 DingoDB 进行交互。使用支持 MCP 的客户端，连接上 DingoDB 数据库，可以列出所有的表、读取数据以及执行 SQL，然后可以使用大模型对数据库中的数据进一步分析。

[<img src="https://cursor.com/deeplink/mcp-install-dark.svg" alt="Install in Cursor">](https://cursor.com/en/install-mcp?name=DingoDB-MCP&config=eyJjb21tYW5kIjogInV2eCIsICJhcmdzIjogWyItLWZyb20iLCAib2NlYW5iYXNlLW1jcCIsICJvY2VhbmJhc2VfbWNwX3NlcnZlciJdLCAiZW52IjogeyJPQl9IT1NUIjogIiIsICJPQl9QT1JUIjogIiIsICJPQl9VU0VSIjogIiIsICJPQl9QQVNTV09SRCI6ICIiLCAiT0JfREFUQUJBU0UiOiAiIn19)

## 📋 目录

- [特性](#-特性)
- [可用工具](#%EF%B8%8F-可用工具)
- [前提条件](#-前提条件)
- [安装](#-安装)
  - [从源码安装](#从源码安装)
  - [从 PyPI 仓库安装](#从-pypi-仓库安装)
- [配置](#%EF%B8%8F-配置)
- [快速开始](#-快速开始)
  - [Stdio 模式](#stdio-模式)
  - [SSE 模式](#sse-模式)
  - [Streamable HTTP 模式](#streamable-http-模式)
- [高级功能](#-高级功能)
  - [鉴权](#-鉴权)
  - [AI 记忆系统](#-ai-记忆系统)
- [示例](#-示例)
- [安全](#-安全)
- [许可证](#-许可证)
- [贡献](#-贡献)

## ✨ 特性

- **数据库操作**: 列出表、读取数据、执行 SQL 查询
- **AI 记忆系统**: 基于 DingoDB 的持久化向量记忆
- **高级搜索**: 全文搜索、向量搜索和混合搜索
- **安全**: 鉴权支持和安全的数据库访问
- **多传输模式**: 支持 stdio、SSE 和 Streamable HTTP 模式

## 🛠️ 可用工具

### 核心数据库工具
- [✔️] **执行 SQL 语句** - 运行自定义 SQL 命令
- [✔️] **查询当前租户** - 获取当前租户信息
- [✔️] **查询所有 server 节点** - 列出所有服务器节点（仅支持 root 租户）
- [✔️] **查询资源信息** - 查看资源容量（仅支持 root 租户）

### 搜索与记忆工具
- [✔️] **搜索 DingoDB 文档** - 搜索官方文档（实验特性）
- [✔️] **AI 记忆系统** - 基于向量的持久化记忆（实验特性）
- [✔️] **全文搜索** - 在 DingoDB 表中搜索文档
- [✔️] **向量相似性搜索** - 执行基于向量的相似性搜索
- [✔️] **混合搜索** - 结合关系过滤和向量搜索

> **注意**: 实验性工具可能会随着发展而改变 API。

## 📋 前提条件

你需要有一个 DingoDB 数据库。你可以：
- **本地安装**: 参考[安装文档](https://dingodb.readthedocs.io/en/latest/deployment/standalone/windows.html)

## 🚀 安装

### 从源码安装

#### 1. 克隆仓库
```bash
git clone https://github.com/dingodb/awesome-dingodb-mcp.git
cd awesome-dingodb-mcp/src/dingodb_mcp_server
```

#### 2. 安装 Python 包管理器并创建虚拟环境
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate  # 在Windows系统上执行 `.venv\Scripts\activate`
```

#### 3. 配置环境（可选）
如果你想使用 `.env` 文件进行配置：
```bash
cp .env.template .env
# 编辑 .env 文件，填入你的 DingoDB 连接信息
```

#### 4. 处理网络问题（可选）
如果遇到网络问题，可以使用阿里云镜像：
```bash
export UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple/"
```

#### 5. 安装依赖
```bash
uv pip install .
```

### 从 PyPI 仓库安装

快速通过 pip 安装：
```bash
uv pip install dingodb-mcp
```
## ⚙️ 配置

有两种方式可以配置 DingoDB 连接信息：

### 方法 1: 环境变量
设置以下环境变量：

```bash
DINGODB_HOST=localhost     # 数据库的地址
DINGODB_PORT=3307         # 可选的数据库的端口（如果没有配置，默认是3307)
DINGODB_USER=root
DINGODB_PASSWORD=xxxxxx
DINGODB_DATABASE=dingo
```

### 方法 2: .env 文件
在 `.env` 文件中进行配置（从 `.env.template` 复制并修改）。
## 🚀 快速开始

DingoDB MCP Server 支持三种传输模式：

### Stdio 模式

在你的 MCP 客户端配置文件中添加以下内容：

```json
{
  "mcpServers": {
    "dingodb": {
      "command": "uv",
      "args": [
        "--directory", 
        "path/to/awesome-dingodb-mcp/src/dingodb_mcp_server",
        "run",
        "dingodb_mcp_server"
      ],
      "env": {
        "DINGODB_HOST": "localhost",
        "DINGODB_PORT": "3307",
        "DINGODB_USER": "root",
        "DINGODB_PASSWORD": "xxxxxx",
        "DINGODB_DATABASE": "dingo"
      }
    }
  }
}
```

### SSE 模式

在src/dingodb_mcp_server目录下，启动 SSE 模式服务器：

```bash
uv run dingodb_mcp_server --transport sse --port 8000
```

**参数说明:**
- `--transport`: MCP 服务器传输类型（默认: stdio）
- `--host`: 绑定的主机（默认: 127.0.0.1，使用 0.0.0.0 允许远程访问）
- `--port`: 监听端口（默认: 8000）

**替代启动方式（不使用 uv）:**
```bash
cd dingodb_mcp/ && python3 -m server --transport sse --port 8000
```

**配置 URL:** `http://ip:port/sse`
#### 客户端配置示例

**VSCode 插件 Cline:**
```json
"sse-dingodb": {
  "autoApprove": [],
  "disabled": false,
  "timeout": 60,
  "type": "sse",
  "url": "http://ip:port/sse"
}
```

**Cursor:**
```json
"sse-dingodb": {
  "autoApprove": [],
  "disabled": false,
  "timeout": 60,
  "type": "sse",
  "url": "http://ip:port/sse"
}
```
**Cherry Studio:**
- MCP → 通用 → 类型: 从下拉菜单中选择 "服务器发送事件 (sse)"

### Streamable HTTP 模式

启动 Streamable HTTP 模式服务器：

```bash
uv run dingodb_mcp_server --transport streamable-http --port 8000
```

**替代启动方式（不使用 uv）:**
```bash
cd dingodb_mcp/ && python3 -m server --transport streamable-http --port 8000
```

**配置 URL:** `http://ip:port/mcp`

#### 客户端配置示例

**VSCode 插件 Cline:**
```json
"streamable-dingodb": {
  "autoApprove": [],
  "disabled": false,
  "timeout": 60,
  "type": "streamableHttp",
  "url": "http://ip:port/mcp"
}
```

**Cursor:**
```json
"streamable-dingodb": {
  "autoApprove": [],
  "disabled": false,
  "timeout": 60,
  "type": "streamableHttp",
  "url": "http://ip:port/mcp"
}
```

**Cherry Studio:**
- MCP → 通用 → 类型: 从下拉菜单中选择 "可流式传输的 HTTP (streamableHttp)"
## 🔧 高级功能

### 🔐 鉴权

在环境变量或 `.env` 文件中配置 `ALLOWED_TOKENS` 变量。在 MCP 客户端请求头中添加 `"Authorization": "Bearer <token>"`。只有携带有效 token 的请求才能访问 MCP 服务器服务。

**示例:**
```bash
ALLOWED_TOKENS=tokenOne,tokenTwo
```

### 客户端配置

**Cherry Studio:**
- 在 MCP → General → Headers 输入框中添加 `Authorization=Bearer <token>`

**Cursor:**
```json
{
  "mcpServers": {
    "dingodb-sse": {
      "autoApprove": [],
      "disabled": false,
      "timeout": 60,
      "type": "sse",
      "url": "http://ip:port/sse",
      "headers": {
        "Authorization": "Bearer <token>"
      }
    }
  }
}
```

**Cline:**
- Cline 目前不支持在请求头中设置 Authorization
- 可以参考这个 [issue](https://github.com/cline/cline/issues/4391) 了解更新
### 🧠 AI 记忆系统

**实验特性**: 基于 DingoDB 先进向量能力的持久化记忆系统，让您的 AI 助手拥有超强记忆力。

记忆系统使您的 AI 能够在对话间保持连续的上下文，无需重复告知个人偏好和信息。四个智能工具协同工作，创造无缝记忆体验：

- **`dingodb_memory_query`** - 语义搜索和检索相关记忆
- **`dingodb_memory_insert`** - 自动捕获和存储重要对话内容  
- **`dingodb_memory_delete`** - 删除过时或不需要的记忆
- **`dingodb_memory_update`** - 根据新信息演进和更新记忆

## 📚 示例

以下是一些展示 DingoDB MCP Server 功能的示例：

### 示例 1: 列出所有表
问题：
```plaintext
列出dingodb数据库下所有的表
```
回答：
```plaintext
表清单: 
序号	表名	描述
1	cache_job	缓存任务表
2	dingospeed	Dingo性能测试表
```
### Example 2: 查看当前租户
问题：
```
查看当前租户
```
回答：
```
当前租户信息如下：

属性	值
租户ID	0
租户名称	root
创建时间	2020-01-01 00:00:00
```
### 例子 3: 查看所有的 server 节点
你需要是root租户才能查询所有 server 节点  
问题：
```
查询所有服务节点
```
回答：
```
服务节点信息
服务节点ID	节点名称	IP地址	状态	角色
1	node01	192.168.1.10	ONLINE	COORDINATOR
2	node02	192.168.1.11	ONLINE	STORE
3	node03	192.168.1.12	ONLINE	EXECUTOR
```
### 例子 4: 查看资源容量
你需要是root租户，才可以查询资源容量  
问题：
```
查看资源容量
```
回答：
```
资源容量信息
类型	总容量	已使用容量	剩余容量
CPU	48 核	24 核	24 核
RAM	128 GB	64 GB	64 GB
存储空间	1000 GB	750 GB	250 GB
磁盘空间	5000 GB	4500 GB	500 GB
```
### 例子 5: 使用标量查询
问题：
```
搜索表test下索引text_index中description字段包含keyboard字符，且text_id>=3的数据，最多3条。
```
回答：
```
根据搜索结果，在表 test 下使用 text_index 并指定 text_id >= 3 和 description LIKE keyboard 的条件，找到以下记录：

搜索结果
ID	Description	Category	Rating	Text ID	相关性得分
4	Plastic Keyboard	Electronics	4	4	0.3884579
```
### 例子 6: 向量查询
问题：
```
搜索表test下字段feature向量为[0.19151945, 0.62210876, 0.43772775, 0.7853586, 0.77997583, 0.2725926, 0.27646425, 0.8018722]的数据.
```
回答：
```
根据查询结果，表test中匹配特征向量 [0.19151945, 0.62210876, 0.43772775, 0.7853586, 0.77997583, 0.2725926, 0.27646425, 0.8018722] 的唯一数据记录如下：

查询结果
字段	值
ID	1
特征向量	[0.19151945, 0.62210876, 0.43772775,
0.7853586, 0.77997583, 0.2725926,
0.27646425, 0.8018722]
特征ID	1
描述	Plastic Keyboard（塑料键盘）
类别	Electronics（电子设备）
评分	4
文本ID	1
距离值	0.0（完全匹配）
```
### 例子 7: 混合查询
进行关系条件过滤（即标量）和向量的混合查询，同时使用向量和标量作为查询条件  
问题：
```
搜索表test下字段feature向量为[0.19151945, 0.62210876, 0.43772775, 0.7853586, 0.77997583, 0.2725926, 0.27646425, 0.8018722]，且索引text_index中description字段包含keyboard字符，text_id>=3的数据
```
回答：
```
查询结果（共 2 条）：
ID	text_id	Description	Category	向量距离	Rating
5	5	KYO Keyboard	Electronics	0.0（精确匹配）	5
4	4	Plastic Keyboard	Electronics	1.3437778	4
```

### 例子 8：跨会话智能记忆

体验跨会话智能记忆的强大能力：

```
📅 星期一对话
用户: "我喜欢足球和篮球，但是不喜欢游泳。另外我在上海工作，使用Python开发。"
AI: "好的，我已经记住了您的偏好和工作信息！" 
    💾 [自动调用 dingodb_memory_insert 保存偏好信息]

📅 星期三对话  
用户: "推荐一些我可能感兴趣的运动"
AI: 🔍 [自动调用 dingodb_memory_query 搜索"运动 偏好"]
    "根据您之前提到的偏好，我推荐足球和篮球相关的活动！您之前说过不太喜欢游泳，
     所以我为您推荐一些陆地运动..."

📅 一周后对话
用户: "我的工作地点在哪里？用什么编程语言？"  
AI: 🔍 [自动调用 dingodb_memory_query 搜索"工作 编程"]
    "您在上海工作，主要使用Python进行开发。"
```

## 🔒 安全

此 MCP 服务器需要数据库访问才能正常工作。请遵循以下安全最佳实践：

### 基本安全措施

1. **创建专用的 DingoDB 用户**，拥有最小权限
2. **不要使用 root 用户**或管理账户
3. **限制数据库访问**，仅允许必要的操作
4. **启用日志记录**，以便进行审计
5. **定期进行数据库访问的安全审查**

### 安全检查清单

- ❌ 不要将环境变量或凭证提交到版本控制
- ✅ 使用具有最小必需权限的数据库用户
- ✅ 考虑在生产环境中实施查询白名单
- ✅ 监控并记录所有数据库操作
- ✅ 使用鉴权令牌进行 API 访问

### 详细配置

查看 [DingoDB 安全配置指南](./SECURITY.md) 获取详细说明：
- 创建受限的 DingoDB 用户
- 设置适当的权限
- 监控数据库访问
- 安全最佳实践

> ⚠️ **重要**: 配置数据库访问时始终遵循最小权限原则。

## 📄 许可证

Apache License - 查看 [LICENSE](LICENSE) 文件获取详细信息。

## 🤝 贡献

我们欢迎贡献！请按照以下步骤：

1. **Fork 仓库**
2. **创建你的功能分支**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **提交你的修改**
   ```bash
   git commit -m 'Add some amazing feature'
   ```
4. **推送到分支**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **创建 Pull Request**
