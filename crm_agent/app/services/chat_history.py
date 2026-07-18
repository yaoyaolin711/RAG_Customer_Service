"""从本地历史对话库读取 / 写入会话消息（会话键兼容原 contact_username）。"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path

_DEFAULT_DB = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "chat_history"
    / "exported_chats.db"
)
_env_db = (os.getenv("EXPORT_DB_PATH") or "").strip()
EXPORT_DB_PATH = Path(_env_db) if _env_db else _DEFAULT_DB
_TEXT_TYPES = frozenset({"text", "link_or_file", "文本", "链接/文件"})
_TEXT_TYPE = "text"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    datetime TEXT NOT NULL,
    sender_username TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'text',
    username TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS message_contents (
    message_id INTEGER NOT NULL,
    content TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(id)
);

CREATE INDEX IF NOT EXISTS idx_messages_username_id ON messages(username, id);
"""


def ensure_db(db_path: Path | None = None) -> Path:
    """创建导出库目录与表结构（幂等）。"""
    path = db_path or EXPORT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
    return path


def _insert_message(
    conn: sqlite3.Connection,
    *,
    contact_username: str,
    sender_username: str,
    content: str,
    msg_type: str = _TEXT_TYPE,
    dt: str | None = None,
) -> int:
    timestamp = dt or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.execute(
        """
        INSERT INTO messages (datetime, sender_username, type, username)
        VALUES (?, ?, ?, ?)
        """,
        (timestamp, sender_username, msg_type, contact_username),
    )
    message_id = int(cur.lastrowid)
    conn.execute(
        "INSERT INTO message_contents (message_id, content) VALUES (?, ?)",
        (message_id, content),
    )
    return message_id


def save_chat_turn(
    *,
    contact_username: str,
    self_username: str,
    incoming_message: str,
    outgoing_message: str,
) -> bool:
    """
    将一轮对话写入导出库：对方消息 + 我方回复。
    contact_username 为会话键（messages.username）；self_username 为我方微信 ID。
    """
    contact = contact_username.strip()
    self_id = self_username.strip()
    incoming = incoming_message.strip()
    outgoing = outgoing_message.strip()
    if not contact or not self_id or contact == self_id:
        return False
    if not incoming and not outgoing:
        return False

    ensure_db()
    conn = sqlite3.connect(str(EXPORT_DB_PATH))
    try:
        base_dt = datetime.now()
        if incoming:
            _insert_message(
                conn,
                contact_username=contact,
                sender_username=contact,
                content=incoming[:1000],
                dt=base_dt.strftime("%Y-%m-%d %H:%M:%S"),
            )
        if outgoing:
            reply_dt = base_dt.replace(microsecond=0)
            if incoming:
                reply_dt = reply_dt.replace(second=min(reply_dt.second + 1, 59))
            _insert_message(
                conn,
                contact_username=contact,
                sender_username=self_id,
                content=outgoing[:1000],
                dt=reply_dt.strftime("%Y-%m-%d %H:%M:%S"),
            )
        conn.commit()
        return True
    finally:
        conn.close()


def get_export_messages(contact_username: str, limit: int = 50) -> list[dict]:
    """读取指定联系人的近期文本消息，转为 LLM 消息格式。"""
    if not contact_username or not EXPORT_DB_PATH.is_file():
        return []

    conn = sqlite3.connect(str(EXPORT_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT m.datetime, m.sender_username, m.type, mc.content
            FROM messages m
            LEFT JOIN message_contents mc ON mc.message_id = m.id
            WHERE m.username = ?
            ORDER BY m.id DESC LIMIT ?
            """,
            (contact_username, limit),
        ).fetchall()
        result: list[dict] = []
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


def get_export_contacts() -> list[dict]:
    """列出所有有历史记录的联系人（含最后消息时间）。"""
    if not EXPORT_DB_PATH.is_file():
        return []
    conn = sqlite3.connect(str(EXPORT_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT m.username, m.sender_username, mc.content, m.datetime
            FROM messages m
            LEFT JOIN message_contents mc ON mc.message_id = m.id
            WHERE m.id IN (
                SELECT MAX(id) FROM messages GROUP BY username
            )
            ORDER BY m.id DESC
            """
        ).fetchall()
        result = []
        for r in rows:
            content = (r["content"] or "").strip()[:60]
            result.append({
                "contact_username": r["username"],
                "last_sender": r["sender_username"],
                "last_content": content,
                "last_datetime": r["datetime"],
            })
        return result
    finally:
        conn.close()


def delete_export_messages(contact_username: str) -> bool:
    """删除指定联系人的所有历史消息。"""
    if not contact_username or not EXPORT_DB_PATH.is_file():
        return False
    conn = sqlite3.connect(str(EXPORT_DB_PATH))
    try:
        conn.execute(
            "DELETE FROM message_contents WHERE message_id IN (SELECT id FROM messages WHERE username = ?)",
            (contact_username,),
        )
        conn.execute("DELETE FROM messages WHERE username = ?", (contact_username,))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()
