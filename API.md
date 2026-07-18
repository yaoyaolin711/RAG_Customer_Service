# RAG Agent API 接口文档

> 版本：v2.1.0  
> 基础地址：`http://localhost:7120`  
> 在线文档：[/docs](http://localhost:7120/docs) · [/redoc](http://localhost:7120/redoc)  
> 定位：抖音店铺买家智能问答；买家一视同仁；目标尽快回复（约 15 秒内）

---

## 统一响应格式

所有接口（成功、业务错误、参数校验失败、404、未捕获异常）均返回：

```json
{
  "data": { "count": 0 },
  "state": {
    "code": 0,
    "message": "ok"
  }
}
```

| state.code | 含义 |
|------------|------|
| `0` | 成功 |
| `400` | 请求参数错误（含 Pydantic 校验失败、消息为空） |
| `404` | 路径不存在 |
| `500` | 服务内部错误 |

> HTTP 状态码统一为 `200`，通过 `state.code` 判断成败。  
> 失败时 `data` 至少包含 `"count": 0`。

---

## 1. 统一聊天接口（主入口）

**`POST /api/v1/chat`**

由 **UnifiedReplyAgent** 处理。`mode` / `user_tag` 参数保留兼容，**当前所有买家消息均走同一流水线，不按标签分层**。

固定流程：

```
意图识别 → 分流 →（咨询类）RAG 检索 → 读历史对话 → 拼接上下文 → 大模型生成
```

### 请求参数

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `message` | string | 是 | 买家消息 |
| `user_id` | string | 否 | 会话 ID，默认 `buyer_demo_001` |
| `buyer_name` | string | 否 | 买家展示名，写入用户画像；为空则用 `user_id` |
| `session_key` | string | 否 | 会话键，用于读历史对话；为空则用 `user_id` |
| `talent_id` | string | 否 | **兼容旧字段**，等同 `buyer_name` |
| `contact_username` | string | 否 | **兼容旧字段**，等同 `session_key` |
| `user_tag` | string | 否 | **已废弃**，保留兼容，不参与路由 |
| `mode` | string | 否 | **已废弃语义**；传 `rag`/`talent`/`auto` 效果相同 |

### 响应字段（`data`）

| 字段 | 类型 | 说明 |
|------|------|------|
| `count` | int | `sources` 条数 |
| `mode` | string | 固定 `"agent"` |
| `route` | string | 路由类型（如 `rag_agent` / `unified_agent`） |
| `answer` | string | 生成的回复文本 |
| `reply_mode` | string | `rag` / `no_hit` / `casual` / `transaction` / `handoff` |
| `rag_hit` | bool | 是否有有效知识库命中 |
| `user_id` | string | 请求中的会话 ID |
| `buyer_name` | string \| null | 买家展示名 |
| `talent_id` | string \| null | 兼容旧字段，等同 `buyer_name` |
| `session_key` | string \| null | 实际使用的会话键 |
| `history_count` | int | 读到的历史对话条数 |
| `sources` | array | RAG 召回片段 |
| `tools_used` | array | 工具调用记录（含 `search_knowledge_base`） |
| `tag_upgrade` | null | **已废弃**，固定为 `null`（不再升标签） |
| `received` | object | `{ direction: "receive", message }` |
| `reply` | object | `{ direction: "send", message }` |
| `success` | bool | Agent 是否成功 |

**`reply_mode` 说明：**
- `rag` — 知识库有有效命中（score ≥ 阈值，默认 0.45）
- `no_hit` — 未命中，按提示词「帮你确认」兜底，不编造价格/物流政策

### 请求示例

```bash
curl -X POST http://localhost:7120/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "这款多久发货？有运费险吗？",
    "user_id": "buyer_demo_001",
    "session_key": "dy_session_001",
    "buyer_name": "买家小李"
  }'
```

### 响应示例 — 知识库命中

```json
{
  "data": {
    "count": 2,
    "mode": "agent",
    "route": "rag_agent",
    "answer": "亲这款一般下单后48小时内发，具体以物流更新为准，运费险以商详页为准哈",
    "reply_mode": "rag",
    "rag_hit": true,
    "user_id": "buyer_demo_001",
    "buyer_name": "买家小李",
    "talent_id": "买家小李",
    "session_key": "dy_session_001",
    "history_count": 12,
    "sources": [
      {
        "content": "发货时效说明...",
        "source": "店铺知识库",
        "score": 0.70,
        "section": "物流与售后",
        "chunk_type": "faq_qa",
        "question": "多久发货"
      }
    ],
    "tools_used": [
      {
        "tool": "search_knowledge_base",
        "input": { "query": "这款多久发货？有运费险吗？" },
        "success": true,
        "result": { "hit": true, "count": 2, "threshold": 0.45 }
      }
    ],
    "tag_upgrade": null,
    "received": {
      "direction": "receive",
      "message": "这款多久发货？有运费险吗？"
    },
    "reply": {
      "direction": "send",
      "message": "亲这款一般下单后48小时内发，具体以物流更新为准，运费险以商详页为准哈"
    },
    "success": true
  },
  "state": { "code": 0, "message": "ok" }
}
```

### 响应示例 — 未命中知识库

```json
{
  "data": {
    "count": 0,
    "mode": "agent",
    "route": "rag_agent",
    "answer": "这个我帮你确认下，确认了马上回你哈",
    "reply_mode": "no_hit",
    "rag_hit": false,
    "user_id": "buyer_demo_001",
    "buyer_name": null,
    "talent_id": null,
    "history_count": 0,
    "sources": [],
    "tools_used": [
      {
        "tool": "search_knowledge_base",
        "input": { "query": "你们公司上市了吗" },
        "success": true,
        "result": { "hit": false, "count": 0, "threshold": 0.45 }
      }
    ],
    "tag_upgrade": null,
    "received": { "direction": "receive", "message": "你们公司上市了吗" },
    "reply": { "direction": "send", "message": "这个我帮你确认下，确认了马上回你哈" },
    "success": true
  },
  "state": { "code": 0, "message": "ok" }
}
```

### 处理流程

```
买家消息
  → BERT 意图识别（咨询 / 交易 / 投诉 / 其他）
  → 咨询类：向量检索（Milvus + BGE-M3）+ 历史对话 + LLM
  → 交易类：Mock 查单/物流（预留真实 API）
  → 投诉类：建工单 + 转人工话术
  → 其他类：闲聊快回
  → 将本轮对话写回历史库（会话键为 session_key / contact_username）
```

> `talent_id`、`tag_upgrade`、`buyer_name` 为 `null` 时，响应中可能省略对应字段（`response_model_exclude_none`）。

**历史对话库：**
- 默认路径：`data/wechat_messages/chat_export/exported_chats.db`（仓库根目录）
- 可通过环境变量 `EXPORT_DB_PATH` 覆盖
- 会话键为 `session_key`（兼容旧字段 `contact_username`）

---

## 2. 健康检查

**`GET /api/v1/health`**

```bash
curl http://localhost:7120/api/v1/health
```

```json
{
  "data": {
    "count": 0,
    "status": "ok",
    "milvus": "connected",
    "redis": "connected",
    "llm": "configured",
    "agent": "unified_reply"
  },
  "state": { "code": 0, "message": "ok" }
}
```

| 字段 | 说明 |
|------|------|
| `status` | `ok` / `degraded` |
| `milvus` | `connected` 或 `error: ...` |
| `agent` | 当前 Agent 名称 |

---

## 3. 服务元信息

**`GET /api/v1/meta`**

```bash
curl http://localhost:7120/api/v1/meta
```

```json
{
  "data": {
    "count": 4,
    "version": "3.1.0",
    "agent": "unified_reply",
    "llm_model": "deepseek-chat",
    "collection": "rag_collection",
    "relevance_threshold": 0.45,
    "kb_doc": "店铺知识库",
    "modes": ["agent", "rag", "talent", "auto"],
    "routes": ["rag_agent", "transaction", "complaint_handoff", "manual_handoff", "casual_chat", "fallback"],
    "intents": ["咨询类", "交易类", "投诉类", "其他类"],
    "reply_modes": ["rag", "no_hit", "casual", "transaction", "handoff"],
    "tools": ["intent_classifier", "search_knowledge_base", "answer_confidence", "query_strategy"],
    "flow": "intent_classify → route → query_strategy → rag/handler/llm → confidence/handoff → session",
    "sla_note": "目标尽快回复买家（约 15 秒内）"
  },
  "state": { "code": 0, "message": "ok" }
}
```

> `modes` 中 `rag`/`talent`/`auto` 为历史兼容枚举；主链路为意图分流 + UnifiedReplyAgent。

---

## 4. 服务首页

### `GET /`

```json
{
  "data": {
    "count": 3,
    "service": "RAG Agent",
    "version": "2.1.0",
    "docs": "/docs",
    "redoc": "/redoc",
    "openapi": "/openapi.json",
    "api": "/api/v1",
    "endpoints": {
      "chat": "POST /api/v1/chat",
      "health": "GET /api/v1/health",
      "meta": "GET /api/v1/meta"
    }
  },
  "state": { "code": 0, "message": "ok" }
}
```

---

## 5. 兼容接口

### `POST /api/wechat/chat`

旧路径兼容，与 `POST /api/v1/chat` 相同，均调用 `handle_chat()` → `UnifiedReplyAgent`。请求体字段：

| 字段 | 说明 |
|------|------|
| `message` | 必填 |
| `user_id` | 默认 `buyer_demo_001` |
| `user_tag` | 已废弃，默认 `B` |
| `session_key` / `contact_username` | 为空则用 `user_id` |

响应格式与 `/api/v1/chat` 完全一致。

### `GET /api/wechat/health` / `GET /api/wechat/meta`

分别等同 `/api/v1/health`、`/api/v1/meta`。

### `POST /api/talents/{talent_id}/simulate`

旧路径兼容：按 ID 模拟买家消息，内部走 `UnifiedReplyAgent`，但**不经过** `handle_chat()`，响应字段与 v1/chat 略有不同：

- `tag_upgrade` 恒为无业务逻辑
- **不会**将对话写入 SQLite 历史库
- 响应中无完整 `answer`/`route`/`rag_hit`/`tools_used` 等字段

**请求体：**

```json
{
  "message": "你好，想问下尺码怎么选",
  "session_key": ""
}
```

**响应示例：**

```json
{
  "data": {
    "count": 1,
    "mode": "agent",
    "received": {
      "direction": "receive",
      "message": "你好，想问下尺码怎么选",
      "intent": "simulate"
    },
    "reply": {
      "direction": "send",
      "message": "亲你说下平时穿多大，我帮你对一下尺码表哈",
      "intent": "simulate_reply"
    },
    "sources": [],
    "reply_mode": "no_hit",
    "history_count": 0
  },
  "state": { "code": 0, "message": "ok" }
}
```

---

## 错误响应

所有错误均使用统一格式，HTTP 状态码为 `200`：

```json
{
  "data": { "count": 0 },
  "state": {
    "code": 400,
    "message": "message: Field required"
  }
}
```

### 参数错误（state.code = 400）

```json
{
  "data": { "count": 0 },
  "state": {
    "code": 400,
    "message": "消息不能为空"
  }
}
```

### 服务错误（state.code = 500）

```json
{
  "data": { "count": 0 },
  "state": {
    "code": 500,
    "message": "Agent 处理失败"
  }
}
```

---

## 启动服务

```bash
cd crm_agent/crm_agent
venv\Scripts\activate
python main.py
```

或双击根目录 `start_api.bat`，然后访问 http://localhost:7120/docs 查看交互式文档。
