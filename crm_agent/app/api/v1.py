"""
RAG Agent 统一 API v1 — 抖音店铺买家智能问答

所有消息由 UnifiedReplyAgent 处理：
  意图识别 → 分流 →（咨询类）RAG 检索 + 历史对话 → 大模型生成回复
  目标：尽快回复买家（约 15 秒内）
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.response import api_ok
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HistoryReadResponse,
    HistoryWriteRequest,
    HistoryWriteResponse,
    MetaResponse,
)
from app.services.chat_history import delete_export_messages, get_export_contacts, get_export_messages, save_chat_turn
from app.services.chat_service import get_health, get_meta, handle_chat

router = APIRouter()


@router.post(
    "/chat",
    summary="统一聊天接口（店铺买家）",
    description="""
**处理流程：**

1. **BERT 意图识别**（咨询类 / 交易类 / 投诉类 / 其他类）
2. 按意图分流：
   - 咨询类 → RAG 检索 + 历史对话 + LLM
   - 交易类 → Mock 订单查询（预留真实 API）
   - 投诉类 → 建工单 + 转人工话术
   - 其他类 / 低置信 → LLM 闲聊
3. 将本轮对话写入历史库
4. 买家一视同仁，不按 A/B/C 标签分流；目标尽快回复（约 15 秒内）

**参数说明：**
- `session_key`：会话键，用于读历史（为空则用 `user_id`）；兼容旧字段 `contact_username`
- `buyer_name`：买家展示名，写入用户画像；兼容旧字段 `talent_id`
- `user_tag`：已废弃，保留兼容，不参与路由
- `mode`：保留兼容，传 rag/talent/auto 效果相同
""",
    response_model=ChatResponse,
    response_model_exclude_none=True,
)
def unified_chat(body: ChatRequest):
    try:
        data = handle_chat(
            message=body.message,
            mode=body.mode,
            user_id=body.user_id,
            user_tag=body.user_tag,
            buyer_name=body.buyer_name,
            talent_id=body.talent_id,
            session_key=body.session_key,
            contact_username=body.contact_username,
            tool_loop=body.tool_loop,
        )
        return api_ok(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/history/{contact_username}",
    summary="读取历史对话",
    description="读取指定联系人的近期文本消息历史",
)
def read_history(contact_username: str, limit: int = 50):
    messages = get_export_messages(contact_username=contact_username, limit=limit)
    return api_ok({
        "count": len(messages),
        "contact_username": contact_username,
        "messages": messages,
    })


@router.post(
    "/history/write",
    summary="写入一轮历史对话",
    description="将一轮对话（对方消息 + 我方回复）写入历史库",
)
def write_history(body: HistoryWriteRequest):
    ok = save_chat_turn(
        contact_username=body.contact_username,
        self_username=body.self_username,
        incoming_message=body.incoming_message,
        outgoing_message=body.outgoing_message,
    )
    return api_ok({"count": 1, "success": ok})


@router.get("/history/contacts", summary="列出所有历史会话联系人")
def list_history_contacts():
    contacts = get_export_contacts()
    return api_ok({"count": len(contacts), "contacts": contacts})


@router.delete("/history/{contact_username}", summary="删除指定联系人的历史记录")
def remove_history(contact_username: str):
    ok = delete_export_messages(contact_username=contact_username)
    return api_ok({"deleted": ok})


@router.get("/health", summary="健康检查", response_model=HealthResponse)
def unified_health():
    return api_ok(get_health())


@router.get("/meta", summary="服务元信息", response_model=MetaResponse)
def unified_meta():
    return api_ok(get_meta())
