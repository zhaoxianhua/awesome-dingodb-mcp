English | [简体中文](README_CN.md)
# DingoDB MCP Server

DingoDB MCP Server can interact with DingoDB through MCP (Model Context Protocol). Using an MCP-compatible client to connect to a DingoDB database, you can list all tables, read data, and execute SQL, then use large language models to further analyze the data in the database.

[<img src="https://cursor.com/deeplink/mcp-install-dark.svg" alt="Install in Cursor">](https://cursor.com/en/install-mcp?name=DingoDB-MCP&config=eyJjb21tYW5kIjogInV2eCIsICJhcmdzIjogWyItLWZyb20iLCAib2NlYW5iYXNlLW1jcCIsICJvY2VhbmJhc2VfbWNwX3NlcnZlciJdLCAiZW52IjogeyJPQl9IT1NUIjogIiIsICJPQl9QT1JUIjogIiIsICJPQl9VU0VSIjogIiIsICJPQl9QQVNTV09SRCI6ICIiLCAiT0JfREFUQUJBU0UiOiAiIn19)

## 📋 Table of Contents

- [Features](#-features)
- [Available Tools](#-available-tools)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
  - [Install from Source](#install-from-source)
  - [Install from PyPI](#install-from-pypi)
- [Configuration](#-configuration)
- [Quick Start](#-quick-start)
  - [Stdio Mode](#stdio-mode)
  - [SSE Mode](#sse-mode)
  - [Streamable HTTP Mode](#streamable-http-mode)
- [Advanced Features](#-advanced-features)
  - [Authentication](#-authentication)
  - [AI Memory System](#-ai-memory-system)
- [Examples](#-examples)
- [Security](#-security)
- [License](#-license)
- [Contributing](#-contributing)

## ✨ Features

- **Database Operations**: List tables, read data, execute SQL queries
- **AI Memory System**: Persistent vector memory based on DingoDB
- **Advanced Search**: Full-text search, vector search, and hybrid search
- **Security**: Authentication support and secure database access
- **Multiple Transport Modes**: Support for stdio, SSE, and Streamable HTTP modes

## 🛠️ Available Tools

### Core Database Tools
- [✔️] **Execute SQL Statement** - Run custom SQL commands
- [✔️] **Query Current Tenant** - Get current tenant information
- [✔️] **Query All Server Nodes** - List all server nodes (root tenant only)
- [✔️] **Query Resource Information** - View resource capacity (root tenant only)

### Search & Memory Tools
- [✔️] **Search DingoDB Documents** - Search official documentation (experimental)
- [✔️] **AI Memory System** - Vector-based persistent memory (experimental)
- [✔️] **Full-Text Search** - Search documents in DingoDB tables
- [✔️] **Vector Similarity Search** - Perform vector-based similarity search
- [✔️] **Hybrid Search** - Combine relational filtering and vector search

> **Note**: Experimental tools may have their APIs change as they evolve.

## 📋 Prerequisites

You need to have a DingoDB database. You can:
- **Local Installation**: Refer to the [Installation Guide](https://dingodb.readthedocs.io/en/latest/deployment/standalone/windows.html)

## 🚀 Installation

### Install from Source

#### 1. Clone the Repository
```bash
git clone https://github.com/dingodb/awesome-dingodb-mcp.git
cd awesome-dingodb-mcp/src/dingodb_mcp_server
```

#### 2. Install Python Package Manager and Create Virtual Environment
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate  # On Windows, execute `.venv\Scripts\activate`
```

#### 3. Configure Environment (Optional)
If you want to use a `.env` file for configuration:
```bash
cp .env.template .env
# Edit the .env file with your DingoDB connection information
```

#### 4. Handle Network Issues (Optional)
If you encounter network issues, you can use the Aliyun mirror:
```bash
export UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple/"
```

#### 5. Install Dependencies
```bash
uv pip install .
```

### Install from PyPI

Quick installation via pip:
```bash
uv pip install dingodb-mcp
```
## ⚙️ Configuration

There are two ways to configure DingoDB connection information:

### Method 1: Environment Variables
Set the following environment variables:

```bash
DINGODB_HOST=localhost     # Database host address
DINGODB_PORT=2881         # Optional database port (default is 2881 if not configured)
DINGODB_USER=your_username
DINGODB_PASSWORD=your_password
DINGODB_DATABASE=your_database
```

### Method 2: .env File
Configure in the `.env` file (copy from `.env.template` and modify).
## 🚀 Quick Start

DingoDB MCP Server supports three transport modes:

### Stdio Mode

Add the following to your MCP client configuration file:

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
        "DINGODB_PORT": "2881",
        "DINGODB_USER": "your_username",
        "DINGODB_PASSWORD": "your_password",
        "DINGODB_DATABASE": "your_database"
      }
    }
  }
}
```

### SSE Mode

Start the SSE mode server:

```bash
uv run dingodb_mcp_server --transport sse --port 8000
```

**Parameter Description:**
- `--transport`: MCP server transport type (default: stdio)
- `--host`: Bind address (default: 127.0.0.1, use 0.0.0.0 for remote access)
- `--port`: Listening port (default: 8000)

**Alternative Startup (without uv):**
```bash
cd dingodb_mcp/ && python3 -m server --transport sse --port 8000
```

**Configure URL:** `http://ip:port/sse`
#### Client Configuration Example

**VSCode Plugin Cline:**
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
- MCP → General → Type: Select "Server-Sent Events (sse)" from the dropdown

### Streamable HTTP Mode

Start the Streamable HTTP mode server:

```bash
uv run dingodb_mcp_server --transport streamable-http --port 8000
```

**Alternative Startup (without uv):**
```bash
cd dingodb_mcp/ && python3 -m server --transport streamable-http --port 8000
```

**Configure URL:** `http://ip:port/mcp`

#### Client Configuration Example

**VSCode Plugin Cline:**
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
- MCP → General → Type: Select "Streamable HTTP (streamableHttp)" from the dropdown
## 🔧 Advanced Features

### 🔐 Authentication

Configure the `ALLOWED_TOKENS` variable in environment variables or `.env` file. Add `"Authorization": "Bearer <token>"` to MCP client request headers. Only requests with valid tokens can access the MCP server services.

**Example:**
```bash
ALLOWED_TOKENS=tokenOne,tokenTwo
```

### Client Configuration

**Cherry Studio:**
- In MCP → General → Headers input box, add `Authorization=Bearer <token>`

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
- Cline currently does not support setting Authorization in request headers
- Refer to this [issue](https://github.com/cline/cline/issues/4391) for updates
### 🧠 AI Memory System

**Experimental Feature**: A persistent memory system based on DingoDB's advanced vector capabilities, giving your AI assistant super memory.

The memory system enables your AI to maintain continuous context across conversations without repeating personal preferences and information. Four intelligent tools work together to create a seamless memory experience:

- **`dingodb_memory_query`** - Semantic search and retrieve relevant memories
- **`dingodb_memory_insert`** - Automatically capture and store important conversation content
- **`dingodb_memory_delete`** - Delete outdated or unwanted memories
- **`dingodb_memory_update`** - Evolve and update memories based on new information

## 📚 Examples

Here are some examples demonstrating DingoDB MCP Server functionality:

### Example 1: List All Tables
Question:
```plaintext
How many tables are there in the test database, and what are they?
```
Answer:
```plaintext
Tables in test:
t1
t2
```
### Example 2: View Current Tenant
Question:
```
What is my current tenant name?
```
Answer:
```
Your current tenant name is 'sys'.
```
### Example 3: View All Server Nodes
You need to be the sys tenant to query all server nodes
Question:
```
Query all server nodes
```
Answer:
```
Successfully queried all server nodes. Here are the details:

| SVR_IP   | SVR_PORT | ID | ZONE  | SQL_PORT | WITH_ROOTSERVER | STATUS | START_SERVICE_TIME        | STOP_TIME | BLOCK_MIGRATE_IN_TIME | CREATE_TIME            | MODIFY_TIME            | BUILD_VERSION                                                                 | LAST_OFFLINE_TIME |
|----------|----------|----|-------|----------|-----------------|--------|---------------------------|-----------|-----------------------|-------------------------|-------------------------|------------------------------------------------------------------------------|-------------------|
| 127.0.0.1 | 2882     | 1  | zone1 | 2881     | YES             | ACTIVE | 2024-06-27 15:40:19.003123 | None      | None                  | 2024-06-19 16:39:11.749421 | 2024-06-27 15:40:21.001573 | 4.2.3.1_101000012024061216-c436ab3557c826ab31931d11ec836dc4df09d9de(Jun 12 2024 16:25:30) | None              |

This table provides comprehensive information about each server node, including IP addresses, ports, zones, statuses, and timestamps.
```
### Example 4: View Resource Capacity
You need to be the root tenant to query resource capacity
Question:
```
Query the current resource capacity information
```
Answer:
```
The current resource capacity information has been successfully queried. Here are the details:

| SVR_IP   | SVR_PORT | ZONE  | SQL_PORT | CPU_CAPACITY | CPU_CAPACITY_MAX | CPU_ASSIGNED | CPU_ASSIGNED_MAX | MEM_CAPACITY | MEM_ASSIGNED | LOG_DISK_CAPACITY | LOG_DISK_ASSIGNED | LOG_DISK_IN_USE | DATA_DISK_CAPACITY | DATA_DISK_IN_USE | DATA_DISK_HEALTH_STATUS | MEMORY_LIMIT | DATA_DISK_ALLOCATED | DATA_DISK_ABNORMAL_TIME | SSL_CERT_EXPIRED_TIME |
|----------|----------|-------|----------|--------------|------------------|--------------|------------------|--------------|--------------|-------------------|-------------------|-----------------|--------------------|-----------------|-------------------------|--------------|---------------------|-------------------------|-----------------------|
| 127.0.0.1 | 2882     | zone1 | 2881     | 4            | 4.0              | 2.0          | 2.0              | 5368709120    | 3221225472    | 5368709120         | 4294967296         | 3556769792       | 10737418240        | 664797184        | NORMAL                  | 6442450944    | 10737418240         | None                    | None                  |

This table provides comprehensive information about the resource capacities and assignments for the server.
```
### Example 5: Full-Text Search
Question:
```
Find content in t2 table that mention "apple"
```
Answer:
```
I have successfully queried the documents in the content field of table t2 that mention "apple". Here are the results:

1. Document ID: 2, Author: john, Age: 20, Content: I have a pear and an apple
2. Document ID: 1, Author: david, Age: 19, Content: I have a pineapple and an apple
3. Document ID: 3, Author: tom, Age: 21, Content: I also have a pear and an apple
```
### Example 6: Vector Search
Question:
```
Please search for records in table t2 that are close to the vector [0.1, 0.1, 0.1].
```
Answer:
```
The vector search on table t2 has returned results successfully. I can now analyze and present these results to the user.

The search found three records that are close to the vector [0.1, 0.1, 0.1]:

1. Record with id 2, name 'john', age 20, description 'I have a pear and an apple', vector '[0.1,0.1,0.2]', and distance 0.1
2. Record with id 3, name 'tom', age 21, description 'I also have a pear and an apple', vector '[0.1,0.1,0.2]', and distance 0.1
3. Record with id 1, name 'david', age 19, description 'I have a pineapple and an apple', vector '[0.1,0.2,0.3]', and distance 0.2236

These records are ordered by their distance from the search vector, with closer matches appearing first.
```
### Example 7: Hybrid Search
Perform hybrid query with relational filtering (scalar) and vector, using both vector and scalar as query conditions
Question:
```
Please find the record in table t2 that has a vector close to [0.1, 0.1, 0.2] and an age of 20.
```
Answer:
```
The record in table `t2` that has a vector close to `[0.1, 0.1, 0.2]` and an age of 20 is:
(2, 'john', 20, 'I have a pear and an apple', '[0.1,0.1,0.2]', 0.0)
```

### Example 8: Cross-Session Intelligent Memory

Experience the powerful capability of cross-session intelligent memory:

```
📅 Monday Conversation
User: "I like football and basketball, but I don't like swimming. Also, I work in Shanghai and use Python for development."
AI: "Got it, I've remembered your preferences and work information!"
    💾 [Automatically calls dingodb_memory_insert to save preference information]

📅 Wednesday Conversation
User: "Recommend some sports I might be interested in"
AI: 🔍 [Automatically calls dingodb_memory_query to search for "sports preference"]
    "Based on your preferences mentioned earlier, I recommend football and basketball related activities! You mentioned
     you don't like swimming much, so I'm recommending some land sports for you..."

📅 One Week Later
User: "Where is my workplace? What programming language do you use?"
AI: 🔍 [Automatically calls dingodb_memory_query to search for "work programming"]
    "You work in Shanghai, mainly using Python for development."
```

## 🔒 Security

This MCP server requires database access to function properly. Please follow these security best practices:

### Basic Security Measures

1. **Create a dedicated DingoDB user** with minimum privileges
2. **Do not use root user** or admin accounts
3. **Limit database access** to only necessary operations
4. **Enable logging** for auditing purposes
5. **Regularly conduct security reviews** of database access

### Security Checklist

- ❌ Do not commit environment variables or credentials to version control
- ✅ Use database users with minimum required privileges
- ✅ Consider implementing query whitelists in production
- ✅ Monitor and log all database operations
- ✅ Use authentication tokens for API access

### Detailed Configuration

See the [DingoDB Security Configuration Guide](./SECURITY.md) for detailed instructions:
- Creating restricted DingoDB users
- Setting appropriate permissions
- Monitoring database access
- Security best practices

> ⚠️ **Important**: Always follow the principle of least privilege when configuring database access.

## 📄 License

Apache License - See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the Repository**
2. **Create your feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Create a Pull Request**