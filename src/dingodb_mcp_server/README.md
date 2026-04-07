English | [简体中文](README_CN.md)
# DingoDB MCP Server

DingoDB MCP Server can interact with DingoDB through MCP (Model Context Protocol). Using an MCP-compatible client to connect to a DingoDB database, you can list all tables, read data, and execute SQL. Then, you can use large language models to further analyze the data in the database.

[<img src="https://cursor.com/deeplink/mcp-install-dark.svg" alt="Install in Cursor">](https://cursor.com/en/install-mcp?name=DingoDB-MCP&config=eyJjb21tYW5kIjogInV2eCIsICJhcmdzIjogWyItLWZyb20iLCAib2NlYW5iYXNlLW1jcCIsICJvY2VhbmJhc2VfbWNwX3NlcnZlciJdLCAiZW52IjogeyJPQl9IT1NUIjogIiIsICJPQl9QT1JUIjogIiIsICJPQl9VU0VSIjogIiIsICJPQl9QQVNTV09SRCI6ICIiLCAiT0JfREFUQUJBU0UiOiAiIn19)

## Table of Contents

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

## Features

- **Database Operations**: List tables, read data, execute SQL queries
- **AI Memory System**: Persistent vector memory based on DingoDB
- **Advanced Search**: Full-text search, vector search, and hybrid search
- **Security**: Authentication support and secure database access
- **Multiple Transport Modes**: Supports stdio, SSE, and Streamable HTTP modes

## Available Tools

### Core Database Tools
- [✔️] **Execute SQL Statement** - Run custom SQL commands
- [✔️] **Query Current Tenant** - Get current tenant information
- [✔️] **Query All Server Nodes** - List all server nodes (root tenant only)
- [✔️] **Query Resource Information** - View resource capacity (root tenant only)

### Search and Memory Tools
- [✔️] **Search DingoDB Documentation** - Search official documentation (experimental feature)
- [✔️] **AI Memory System** - Vector-based persistent memory (experimental feature)
- [✔️] **Full-text Search** - Search documents in DingoDB tables
- [✔️] **Vector Similarity Search** - Perform vector-based similarity search
- [✔️] **Hybrid Search** - Combine relational filtering with vector search

> **Note**: Experimental tools may have changing APIs as they evolve.

## Prerequisites

You need to have a DingoDB database. You can:
- **Local Installation**: Refer to the [Installation Guide](https://dingodb.readthedocs.io/en/latest/deployment/standalone/windows.html)

## Installation

### Install from Source

#### 1. Clone the repository
```bash
git clone https://github.com/dingodb/awesome-dingodb-mcp.git
cd awesome-dingodb-mcp/src/dingodb_mcp_server
```

#### 2. Install Python package manager and create virtual environment
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate  # On Windows, run `.venv\Scripts\activate`
```

#### 3. Configure environment (optional)
If you want to use a `.env` file for configuration:
```bash
cp .env.template .env
# Edit .env file with your DingoDB connection information
```

#### 4. Handle network issues (optional)
If you encounter network issues, you can use the Aliyun mirror:
```bash
export UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple/"
```

#### 5. Install dependencies
```bash
uv pip install .
```

### Install from PyPI

Quick installation via pip:
```bash
uv pip install dingodb-mcp
```

## Configuration

There are two ways to configure DingoDB connection information:

### Method 1: Environment Variables
Set the following environment variables:

```bash
DINGODB_HOST=localhost     # Database host address
DINGODB_PORT=3307         # Optional database port (default is 3307 if not configured)
DINGODB_USER=root
DINGODB_PASSWORD=xxxxxx
DINGODB_DATABASE=dingo
```

### Method 2: .env file
Configure in the `.env` file (copy from `.env.template` and modify).

## Quick Start

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
        "DINGODB_PORT": "3307",
        "DINGODB_USER": "root",
        "DINGODB_PASSWORD": "xxxxxx",
        "DINGODB_DATABASE": "dingo"
      }
    }
  }
}
```

### SSE Mode

Start the SSE mode server in the src/dingodb_mcp_server directory:

```bash
uv run dingodb_mcp_server --transport sse --port 8000
```

**Parameter Description:**
- `--transport`: MCP server transport type (default: stdio)
- `--host`: Bind to host (default: 127.0.0.1, use 0.0.0.0 for remote access)
- `--port`: Listen port (default: 8000)

**Alternative startup method (without uv):**
```bash
cd dingodb_mcp/ && python3 -m server --transport sse --port 8000
```

**Configure URL:** `http://ip:port/sse`

#### Client Configuration Examples

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

**Alternative startup method (without uv):**
```bash
cd dingodb_mcp/ && python3 -m server --transport streamable-http --port 8000
```

**Configure URL:** `http://ip:port/mcp`

#### Client Configuration Examples

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

## Advanced Features

### Authentication

Configure the `ALLOWED_TOKENS` variable in environment variables or `.env` file. Add `"Authorization": "Bearer <token>"` to the MCP client request header. Only requests with a valid token can access the MCP server services.

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

### AI Memory System

**Experimental Feature**: A persistent memory system based on DingoDB's advanced vector capabilities, giving your AI assistant a powerful memory.

The memory system enables your AI to maintain continuous context across conversations without repeating personal preferences and information. Four intelligent tools work together to create a seamless memory experience:

- **`dingodb_memory_query`** - Semantically search and retrieve relevant memories
- **`dingodb_memory_insert`** - Automatically capture and store important conversation content
- **`dingodb_memory_delete`** - Delete outdated or unwanted memories
- **`dingodb_memory_update`** - Evolve and update memories based on new information

## Examples

Here are some examples demonstrating DingoDB MCP Server functionality:

### Example 1: List All Tables
Question:
```plaintext
List all tables under dingodb database
```
Answer:
```plaintext
Table List:
No.	Table Name	Description
1	cache_job	Cache job table
2	dingospeed	Dingo performance test table
```

### Example 2: View Current Tenant
Question:
```
View current tenant
```
Answer:
```
Current tenant information:

Property	Value
Tenant ID	0
Tenant Name	root
Created At	2020-01-01 00:00:00
```

### Example 3: View All Server Nodes
You need to be the root tenant to query all server nodes

Question:
```
Query all service nodes
```
Answer:
```
Service Node Information
Service Node ID	Node Name	IP Address	Status	Role
1	node01	192.168.1.10	ONLINE	COORDINATOR
2	node02	192.168.1.11	ONLINE	STORE
3	node03	192.168.1.12	ONLINE	EXECUTOR
```

### Example 4: View Resource Capacity
You need to be the root tenant to query resource capacity

Question:
```
View resource capacity
```
Answer:
```
Resource Capacity Information
Type	Total Capacity	Used Capacity	Remaining Capacity
CPU	48 cores	24 cores	24 cores
RAM	128 GB	64 GB	64 GB
Storage	1000 GB	750 GB	250 GB
Disk Space	5000 GB	4500 GB	500 GB
```

### Example 5: Use Scalar Query
Question:
```
Search in table test under index text_index for records where the description field contains "keyboard" and text_id >= 3, limit to 3 results.
```
Answer:
```
Based on the search results, using text_index in table test with conditions text_id >= 3 and description LIKE keyboard, the following records were found:

Search Results
ID	Description	Category	Rating	Text ID	Relevance Score
4	Plastic Keyboard	Electronics	4	4	0.3884579
```

### Example 6: Vector Query
Question:
```
Search in table test for data where the feature field vector is [0.19151945, 0.62210876, 0.43772775, 0.7853586, 0.77997583, 0.2725926, 0.27646425, 0.8018722].
```
Answer:
```
Based on the query results, the unique data record in table test matching the feature vector [0.19151945, 0.62210876, 0.43772775, 0.7853586, 0.77997583, 0.2725926, 0.27646425, 0.8018722] is as follows:

Query Result
Field	Value
ID	1
Feature Vector	[0.19151945, 0.62210876, 0.43772775,
0.7853586, 0.77997583, 0.2725926,
0.27646425, 0.8018722]
Feature ID	1
Description	Plastic Keyboard
Category	Electronics
Rating	4
Text ID	1
Distance Value	0.0 (exact match)
```

### Example 7: Hybrid Query
Perform hybrid query with relational condition filtering (scalar) and vector, using both vector and scalar as query conditions

Question:
```
Search in table test for data where the feature field vector is [0.19151945, 0.62210876, 0.43772775, 0.7853586, 0.77997583, 0.2725926, 0.27646425, 0.8018722], and under index text_index where description field contains "keyboard", text_id >= 3
```
Answer:
```
Query results (total 2 records):
ID	text_id	Description	Category	Vector Distance	Rating
5	5	KYO Keyboard	Electronics	0.0 (exact match)	5
4	4	Plastic Keyboard	Electronics	1.3437778	4
```

### Example 8: Cross-session Intelligent Memory

Experience the powerful capability of cross-session intelligent memory:

```
📅 Monday Conversation
User: "I like football and basketball, but I don't like swimming. Also, I work in Shanghai and use Python for development."
AI: "Got it, I've remembered your preferences and work information!"
    💾 [Automatically calls dingodb_memory_insert to save preference information]

📅 Wednesday Conversation
User: "Recommend some sports I might be interested in"
AI: 🔍 [Automatically calls dingodb_memory_query to search "sports preferences"]
    "Based on your preferences mentioned earlier, I recommend football and basketball related activities! You mentioned
    that you don't like swimming much, so I recommend some land sports for you..."

📅 One Week Later Conversation
User: "Where is my workplace? What programming language do I use?"
AI: 🔍 [Automatically calls dingodb_memory_query to search "work programming"]
    "You work in Shanghai, mainly using Python for development."
```

## Security

This MCP server requires database access to function properly. Please follow these security best practices:

### Basic Security Measures

1. **Create a dedicated DingoDB user** with minimal privileges
2. **Do not use root user** or admin accounts
3. **Limit database access** to only necessary operations
4. **Enable logging** for auditing purposes
5. **Perform regular security reviews** of database access

### Security Checklist

- ❌ Do not commit environment variables or credentials to version control
- ✅ Use database users with the minimum required privileges
- ✅ Consider implementing query whitelisting in production environments
- ✅ Monitor and log all database operations
- ✅ Use authentication tokens for API access

### Detailed Configuration

See the [DingoDB Security Configuration Guide](./SECURITY.md) for detailed instructions:
- Creating restricted DingoDB users
- Setting appropriate permissions
- Monitoring database access
- Security best practices

> **Important**: Always follow the principle of least privilege when configuring database access.

## License

Apache License - See the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
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