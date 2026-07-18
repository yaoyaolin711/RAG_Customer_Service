"""Redis 短期会话状态存储（TTL 缓存）。"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict

import redis

from settings import REDIS_DB, REDIS_HOST, REDIS_PASSWORD, REDIS_PORT, SESSION_TTL_SECONDS
from services.models import SessionSnapshot, SessionStatus


def _redis_key(session_id: str) -> str:
    return f"rag:session:{session_id}"


class RedisSessionStore:
    def __init__(self):
        self._client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            db=REDIS_DB,
            decode_responses=True,
            protocol=2,
            socket_connect_timeout=2,
            socket_timeout=2,
        )

    def ping(self) -> bool:
        return self._client.ping() is True

    def get(self, session_id: str) -> SessionSnapshot | None:
        raw = self._client.get(_redis_key(session_id))
        if not raw:
            return None
        data = json.loads(raw)
        snap = SessionSnapshot(
            session_id=data["session_id"],
            user_id=data["user_id"],
            channel=data.get("channel", "streamlit"),
        )
        for k, v in data.items():
            if hasattr(snap, k):
                setattr(snap, k, v)
        if isinstance(snap.status, str):
            snap.status = SessionStatus(snap.status)
        if not isinstance(snap.turns, list):
            snap.turns = []
        return snap

    def create(self, user_id: str, *, channel: str = "streamlit") -> SessionSnapshot:
        session_id = uuid.uuid4().hex
        snap = SessionSnapshot(session_id=session_id, user_id=user_id, channel=channel)
        snap.touch()
        self.update(snap)
        return snap

    def get_or_create(self, session_id: str | None, user_id: str, *, channel: str = "streamlit") -> SessionSnapshot:
        if session_id:
            existing = self.get(session_id)
            if existing and existing.user_id == user_id:
                return existing
        return self.create(user_id, channel=channel)

    def update(self, snapshot: SessionSnapshot) -> None:
        snapshot.touch()
        payload = asdict(snapshot)
        payload["status"] = snapshot.status.value
        self._client.setex(_redis_key(snapshot.session_id), SESSION_TTL_SECONDS, json.dumps(payload, ensure_ascii=False))

    def mark_closed(self, snapshot: SessionSnapshot, *, status: SessionStatus = SessionStatus.CLOSED, reason: str = "") -> None:
        snapshot.status = status
        snapshot.end_reason = reason or snapshot.end_reason
        snapshot.ended_at = snapshot.ended_at or snapshot.updated_at
        self.update(snapshot)


_store: RedisSessionStore | None = None


def get_redis_session_store() -> RedisSessionStore:
    global _store
    if _store is None:
        _store = RedisSessionStore()
    return _store
