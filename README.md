# RAG Agent

面向 **抖音店铺买家** 的智能问答与客服 Agent：融合 **crm_agent**（统一 API）与 **RAG_mode**（Streamlit 模拟台 + 知识库检索）。
目标回复（约 15 秒内）。

## 快速开始

```bash
cd crm_agent/crm_agent
venv\Scripts\activate
python main.py
```

- **IM 模拟台（推荐）**：http://localhost:7120/console/
- 交互式文档：http://localhost:7120/docs
- 接口文档：[API.md](./API.md)

## 手动重启 Streamlit（可选）

```bash
cd "c:\Users\Administrator\Desktop\自动化汇总\RAG自动问答\RAG_mode\mode"
python -m streamlit run app.py --server.port 7121 --server.headless true
```

## 统一 API（推荐）

所有能力通过 **`POST /api/v1/chat`** 统一 Agent 调用：

```bash
curl -X POST http://localhost:7120/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "这款多久发货？有运费险吗？",
    "user_id": "buyer_001",
    "session_key": "dy_session_001",
    "buyer_name": "买家小李"
  }'
```

流程：**意图识别 → 分流 →（咨询类）RAG 检索 → 读历史对话 → 拼接上下文 → 大模型生成**（`reply_mode`: `rag` / `no_hit` 等；`mode` / `user_tag` 参数已废弃语义）

兼容旧字段：`talent_id` ↔ `buyer_name`，`contact_username` ↔ `session_key`。

### 端点一览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/chat` | **主入口** — 统一聊天 |
| GET | `/api/v1/health` | 健康检查 |
| GET | `/api/v1/meta` | 服务元信息 |
| GET | `/console/` | IM 模拟台（抖音客服风前端） |
| GET | `/docs` | Swagger 交互文档 |
| GET | `/redoc` | ReDoc 文档 |

### 响应格式

```json
{
  "data": {
    "mode": "agent",
    "route": "unified_agent",
    "answer": "...",
    "reply_mode": "rag",
    "rag_hit": true,
    "sources": [],
    "history_count": 0,
    "tag_upgrade": null
  },
  "state": {"code": 0, "message": "ok"}
}
```

完整字段说明见 [API.md](./API.md)（v2.1.0）。

## 项目结构

```
RAG_Agent/
├── API.md                   # 接口文档
├── services/                # [共享] RAG 检索、意图分流等核心业务（28文件）
├── rag/                     # [共享] 知识库入库管线（分块/加载/向量化）
├── data/                    # [共享] 知识库文档
├── scripts/                 # [共享] 入库/测试/模拟脚本
├── embedding.py             # [共享] BGE-M3 Embedding
├── vectorstore.py           # [共享] Milvus 混合向量检索
├── reranker.py              # [共享] BGE-Reranker 重排序
├── shared_config/           # [共享] Redis/MySQL 会话配置
├── web/                     # IM 模拟台前端（挂载于 /console/）
├── crm_agent/crm_agent/
│   ├── main.py              # API 入口 :7120
│   ├── settings.py          # 系统配置
│   ├── app/                 # FastAPI 应用层（API路由/Agent编排）
│   ├── config.yaml          # LLM 配置
│   └── agents.yaml          # Agent 定义
├── RAG_mode/mode/
│   ├── app.py               # Streamlit 模拟台（端口 7121）
│   └── settings.py          # 系统配置
└── start_api.bat
```

启动 API 后打开 **http://localhost:7120/console/** 即可联调。独立打开静态页时可用 `?api=http://localhost:7120` 指定后端

## Streamlit 模拟台（RAG_mode，可选）

```bash
cd RAG_mode/mode
streamlit run app.py
```

- 访问地址：http://localhost:7121

## 环境依赖

- `D:\Milvus` — 向量库 Milvus Lite（`rag_collection`，稠密+稀疏混合检索）
- `D:\BGE-M3` — BGE-M3 本地 Embedding 模型
- `D:\BGE-Reranker-v2-m3` — BGE-Reranker（GPU 重排）
- Redis + MySQL — 会话状态（**RAG_mode 与 crm_agent 共用**）
  - 复制仓库根目录 `.env.session.example` → `.env.session`，只改这一份即可
  - 代码入口：`shared_config/session_store.py`
- API Key：设置环境变量 `DEEPSEEK_KEY`（见各子项目 `.env.example`）

## 知识库入库

```bash
# 从仓库根目录运行
python scripts/ingest_bd_docx.py --file "path/to/your_kb.docx"
# 或重新构建索引
python scripts/ingest_documents.py --rebuild
```
