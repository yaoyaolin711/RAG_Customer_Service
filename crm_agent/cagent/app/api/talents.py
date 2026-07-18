import os
import sqlite3
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agents.talent_reply import TalentReplyAgent
from app.agents.tools.registry import ToolRegistry
from app.agents.tools.talent_tools import (
    TALENT_INTENT_RECOGNITION_TOOL,
    TALENT_MESSAGE_GENERATOR_TOOL,
    TALENT_FOLLOWUP_GENERATOR_TOOL,
    TALENT_PROFILE_ANALYZER_TOOL,
    TALENT_AUTO_REPLY_TOOL,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_talent_agent = None


def _get_talent_agent():
    global _talent_agent
    if _talent_agent is None:
        for _t in [TALENT_INTENT_RECOGNITION_TOOL, TALENT_MESSAGE_GENERATOR_TOOL,
                   TALENT_FOLLOWUP_GENERATOR_TOOL, TALENT_PROFILE_ANALYZER_TOOL,
                   TALENT_AUTO_REPLY_TOOL]:
            _schema = ToolRegistry.get_schema(_t.name)
            ToolRegistry.register(_t, _schema)
        _talent_agent = TalentReplyAgent()
    return _talent_agent


_DEFAULT_DB = Path(__file__).resolve().parents[3] / "data" / "wechat_messages" / "chat_export" / "exported_chats.db"
_EXPORT_DB_PATH = Path(os.getenv("EXPORT_DB_PATH", str(_DEFAULT_DB)))
_TEXT_TYPES = frozenset({"text", "link_or_file", "文本", "链接/文件"})


def _get_export_messages(contact_username: str, limit: int = 50) -> list[dict]:
    if not contact_username or not _EXPORT_DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(_EXPORT_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT m.datetime, m.sender_username, m.type, mc.content
            FROM messages m
            LEFT JOIN message_contents mc ON mc.message_id = m.id
            WHERE m.username = ?
            ORDER BY m.id DESC LIMIT ?
        """, (contact_username, limit)).fetchall()
        result = []
        for r in reversed(rows):
            content = (r["content"] or "").strip()
            if r["type"] in _TEXT_TYPES and content:
                is_self = r["sender_username"] != contact_username
                result.append({
                    "role": "assistant" if is_self else "user",
                    "content": content[:1000],
                })
        return result
    finally:
        conn.close()


class SimulateRequest(BaseModel):
    message: str
    contact_username: str = ""


class SimulateResponse(BaseModel):
    received: dict
    reply: dict


@router.post("/{talent_id}/simulate")
def simulate_talent_message(talent_id: str, body: SimulateRequest):
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    recent = _get_export_messages(body.contact_username, 50)
    context = {
        "recent_history": recent,
        "talent_profile": f"昵称：{talent_id}",
        "session_id": f"talent_{talent_id}",
    }

    reply_text = ""
    try:
        agent = _get_talent_agent()
        result = agent.invoke({
            "task": f"达人的新消息：{body.message}",
            "context": context,
        })
        output = result.get("output", {})
        if isinstance(output, dict):
            reply_text = output.get("result", "") or str(output)
        elif isinstance(output, str):
            reply_text = output
        else:
            reply_text = str(result)
    except Exception as e:
        logger.exception("AI回复失败: talent_id=%s", talent_id)
        reply_text = f"AI回复失败({type(e).__name__})"

    return {
        "received": {"direction": "receive", "message": body.message, "intent": "simulate"},
        "reply": {"direction": "send", "message": reply_text, "intent": "simulate_reply"},
    }
