"""投诉类工单 + 转人工骨架。"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from settings import COMPLAINT_DB_PATH
from services.models import IntentResult, ReplyMode, RouteType, WeChatMessageResponse

_SCHEMA = """
CREATE TABLE IF NOT EXISTS complaint_tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    message TEXT NOT NULL,
    intent_confidence REAL,
    summary TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL
);
"""


def ensure_complaint_db(db_path: str | Path | None = None) -> Path:
    path = Path(db_path or COMPLAINT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()
    return path


def create_complaint_ticket(
    user_id: str,
    message: str,
    intent_confidence: float,
    summary: str = "",
    db_path: str | Path | None = None,
) -> int:
    path = ensure_complaint_db(db_path)
    conn = sqlite3.connect(str(path))
    try:
        cur = conn.execute(
            """
            INSERT INTO complaint_tickets
            (user_id, message, intent_confidence, summary, status, created_at)
            VALUES (?, ?, ?, ?, 'open', ?)
            """,
            (
                user_id,
                message,
                intent_confidence,
                summary or message[:200],
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def handoff_to_douyin(ticket_id: int, user_id: str, message: str) -> None:
    """占位：后期对接抖音客服系统。"""
    # TODO: 对接抖音客服 webhook / API
    _ = (ticket_id, user_id, message)


class ComplaintHandler:
    def handle(
        self,
        user_id: str,
        message: str,
        intent: IntentResult,
    ) -> WeChatMessageResponse:
        ticket_id = create_complaint_ticket(
            user_id=user_id,
            message=message,
            intent_confidence=intent.confidence,
        )
        handoff_to_douyin(ticket_id, user_id, message)
        answer = (
            "非常抱歉给您带来不好的体验，我已经帮您登记了，"
            "马上转接同事专门处理，请稍等一下哈。"
        )
        return WeChatMessageResponse(
            user_id=user_id,
            route=RouteType.COMPLAINT_HANDOFF,
            reply_mode=ReplyMode.HANDOFF,
            answer=answer,
            ticket_id=ticket_id,
            intent=intent.category.value,
            intent_confidence=intent.confidence,
            action=intent.action,
            intent_probabilities=intent.probabilities,
        )
