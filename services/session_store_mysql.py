"""MySQL 长期会话归档（会话级）。"""

from __future__ import annotations

import json
from dataclasses import asdict

import pymysql

from settings import MYSQL_DATABASE, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_PORT, MYSQL_USER
from services.models import SessionSnapshot


def _connect(*, database: str | None = None):
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=database,
        charset="utf8mb4",
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=3,
        read_timeout=5,
        write_timeout=5,
    )


_DDL_DB = f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"

_DDL_TABLE = """
CREATE TABLE IF NOT EXISTS chat_sessions (
  session_id VARCHAR(64) PRIMARY KEY,
  user_id VARCHAR(128) NOT NULL,
  channel VARCHAR(32) NOT NULL,
  started_at VARCHAR(32) NOT NULL,
  updated_at VARCHAR(32) NOT NULL,
  ended_at VARCHAR(32) NULL,
  status VARCHAR(32) NOT NULL,
  end_reason VARCHAR(255) NOT NULL DEFAULT '',
  turn_count INT NOT NULL DEFAULT 0,
  last_intent VARCHAR(64) NOT NULL DEFAULT '',
  last_route VARCHAR(64) NOT NULL DEFAULT '',
  last_reply_mode VARCHAR(64) NOT NULL DEFAULT '',
  last_answer_confidence DOUBLE NOT NULL DEFAULT 0,
  last_needs_handoff TINYINT(1) NOT NULL DEFAULT 0,
  final_summary TEXT,
  turns_json LONGTEXT
);
"""

_DDL_TURNS_COLUMN = """
ALTER TABLE chat_sessions
  ADD COLUMN IF NOT EXISTS turns_json LONGTEXT
"""

_DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_chat_sessions_status ON chat_sessions(status)",
    "CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at ON chat_sessions(updated_at)",
]


class MySQLSessionArchive:
    def __init__(self):
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with _connect(database=None) as conn:
            with conn.cursor() as cur:
                cur.execute(_DDL_DB)
        with _connect(database=MYSQL_DATABASE) as conn:
            with conn.cursor() as cur:
                cur.execute(_DDL_TABLE)
                for ddl in _DDL_INDEXES:
                    try:
                        cur.execute(ddl)
                    except Exception:
                        # MySQL 8.0+ supports IF NOT EXISTS for indexes in 8.0.13+,
                        # but some distributions may still fail; ignore.
                        pass
                try:
                    cur.execute(_DDL_TURNS_COLUMN)
                except Exception:
                    try:
                        cur.execute("ALTER TABLE chat_sessions ADD COLUMN turns_json LONGTEXT")
                    except Exception:
                        pass

    def upsert_session(self, snapshot: SessionSnapshot) -> None:
        data = asdict(snapshot)
        data["status"] = snapshot.status.value
        data["turns_json"] = json.dumps(snapshot.turns or [], ensure_ascii=False)
        del data["turns"]
        sql = """
INSERT INTO chat_sessions (
  session_id, user_id, channel, started_at, updated_at, ended_at, status, end_reason,
  turn_count, last_intent, last_route, last_reply_mode, last_answer_confidence, last_needs_handoff, final_summary, turns_json
) VALUES (
  %(session_id)s, %(user_id)s, %(channel)s, %(started_at)s, %(updated_at)s, %(ended_at)s, %(status)s, %(end_reason)s,
  %(turn_count)s, %(last_intent)s, %(last_route)s, %(last_reply_mode)s, %(last_answer_confidence)s, %(last_needs_handoff)s, %(final_summary)s, %(turns_json)s
)
ON DUPLICATE KEY UPDATE
  user_id=VALUES(user_id),
  channel=VALUES(channel),
  started_at=VALUES(started_at),
  updated_at=VALUES(updated_at),
  ended_at=VALUES(ended_at),
  status=VALUES(status),
  end_reason=VALUES(end_reason),
  turn_count=VALUES(turn_count),
  last_intent=VALUES(last_intent),
  last_route=VALUES(last_route),
  last_reply_mode=VALUES(last_reply_mode),
  last_answer_confidence=VALUES(last_answer_confidence),
  last_needs_handoff=VALUES(last_needs_handoff),
  final_summary=VALUES(final_summary),
  turns_json=VALUES(turns_json)
"""
        with _connect(database=MYSQL_DATABASE) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, data)

    def finalize_session(self, snapshot: SessionSnapshot) -> None:
        self.upsert_session(snapshot)


_archive: MySQLSessionArchive | None = None


def get_mysql_session_archive() -> MySQLSessionArchive:
    global _archive
    if _archive is None:
        _archive = MySQLSessionArchive()
    return _archive

