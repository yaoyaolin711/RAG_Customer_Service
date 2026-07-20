"""
RAG Agent 统一 API v1 — 抖音店铺买家智能问答

所有消息由 UnifiedReplyAgent 处理：
  意图识别 → 分流 →（咨询类）RAG 检索 + 历史对话 → 大模型生成回复
  目标：尽快回复买家（约 15 秒内）
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

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
from app.agents.unified_reply import UnifiedReplyAgent
from app.services.chat_history import delete_export_messages, get_export_contacts, get_export_messages, save_chat_turn
from app.services.chat_service import get_health, get_meta, handle_chat, _build_ctx_from_body

_unified_agent: UnifiedReplyAgent | None = None

def _get_agent() -> UnifiedReplyAgent:
    global _unified_agent
    if _unified_agent is None:
        _unified_agent = UnifiedReplyAgent()
    return _unified_agent

_PLACEHOLDER_POOL = [
    "嗯嗯，我在呢~，我帮你确认下哈",
    "好的，我看看哈~",
    "稍等，我查一下后台~",
    "收到，我核实一下哈~",
    "我看一下数据哈~",
    "行，我确认一下~",
    "嗯，我帮你看看~",
]

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


@router.post(
    "/chat/stream",
    summary="流式聊天接口（SSE）",
    description="""
SSE 事件流：
- placeholder: 请求 10 秒后发占位回复
- tool_call: LLM 决定调用的工具（name, args）
- tool_result: 工具执行结果（name, summary）
- result: 最终回复（text, sources）
""",
)
async def unified_chat_stream(body: ChatRequest):
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _send(event_type: str, data: dict[str, Any]):
        loop.call_soon_threadsafe(queue.put_nowait, (event_type, data))

    def _sync_run():
        _sent_placeholder = False
        try:
            ctx = _build_ctx_from_body(
                message=body.message, user_id=body.user_id,
                buyer_name=body.buyer_name, talent_id=body.talent_id,
                session_key=body.session_key, contact_username=body.contact_username,
                tool_loop=body.tool_loop,
            )
            task = f"用户新消息：{body.message}"

            def callback(typ: str, data: dict[str, Any]):
                _send(typ, data)

            agent = _get_agent()
            result = agent.invoke_stream(
                input_data={"task": task, "context": ctx},
                event_callback=callback,
            )
            _sent_placeholder = True

            if result.get("success"):
                output = result.get("output", {})
                _send("result", {
                    "text": output.get("result", ""),
                    "sources": output.get("sources", []),
                    "needs_handoff": output.get("needs_handoff", False),
                    "route": output.get("route", ""),
                })
            else:
                _send("error", {"message": result.get("error", "未知错误")})
        except Exception as e:
            _send("error", {"message": str(e)})
        finally:
            if not _sent_placeholder:
                _send("placeholder", {"text": random.choice(_PLACEHOLDER_POOL)})
            _send("__done__", {})

    def _timer_thread():
        time.sleep(10)
        loop.call_soon_threadsafe(queue.put_nowait, ("__timer__", {}))

    async def event_generator():
        import threading
        threading.Thread(target=_timer_thread, daemon=True).start()
        result_sent = False

        with ThreadPoolExecutor() as pool:
            pool.submit(_sync_run)
            while True:
                typ, data = await queue.get()
                if typ == "__done__":
                    break
                if typ == "__timer__":
                    if not result_sent:
                        yield f"data: {json.dumps({'type': 'placeholder', 'text': random.choice(_PLACEHOLDER_POOL)}, ensure_ascii=False)}\n\n"
                    continue
                if typ == "result":
                    result_sent = True
                yield f"data: {json.dumps({'type': typ, **data}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
