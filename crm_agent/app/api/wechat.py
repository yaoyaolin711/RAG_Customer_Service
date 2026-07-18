"""旧路径兼容层（原微信 RAG），转发至统一 Agent。"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.response import api_ok
from app.api.schemas import ChatMode
from app.services.chat_service import get_health, get_meta, handle_chat

logger = logging.getLogger(__name__)
router = APIRouter()


class WeChatChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="买家消息")
    user_id: str = Field(default="buyer_demo_001", description="买家/会话 ID")
    user_tag: str = Field(default="B", description="已废弃，保留兼容")
    session_key: str = Field(default="", description="会话键")
    contact_username: str = Field(default="", description="兼容旧字段，等同 session_key")
    tool_loop: bool = Field(False, description="启用灵活工具调用模式")


@router.get("/health")
def wechat_health():
    return api_ok(get_health())


@router.post("/chat")
def wechat_chat(body: WeChatChatRequest):
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")
    try:
        session = body.session_key or body.contact_username or body.user_id
        data = handle_chat(
            message=body.message,
            mode=ChatMode.AUTO,
            user_id=body.user_id,
            user_tag=body.user_tag,
            session_key=session,
            contact_username=session,
            tool_loop=body.tool_loop,
        )
        return api_ok(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/meta")
def wechat_meta():
    return api_ok(get_meta())
