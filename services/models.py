"""RAG 客服服务层数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from datetime import datetime


class IntentCategory(str, Enum):
    CONSULT = "咨询类"
    TRANSACTION = "交易类"
    COMPLAINT = "投诉类"
    OTHER = "其他类"


class RouteType(str, Enum):
    RAG_AGENT = "rag_agent"
    TRANSACTION = "transaction"
    COMPLAINT_HANDOFF = "complaint_handoff"
    MANUAL_HANDOFF = "manual_handoff"
    CASUAL_CHAT = "casual_chat"
    FALLBACK = "fallback"
    UNSUPPORTED = "unsupported"


class ReplyMode(str, Enum):
    RAG = "rag"
    CACHE = "cache"
    CASUAL = "casual"
    TRANSACTION = "transaction"
    HANDOFF = "handoff"


@dataclass
class IntentResult:
    category: IntentCategory
    confidence: float
    action: str
    probabilities: dict[str, float] = field(default_factory=dict)
    raw_text: str = ""
    latency_ms: float = 0.0
    is_fallback: bool = False


@dataclass
class WeChatMessageRequest:
    user_id: str
    message: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    content: str
    source: str
    chunk_id: str
    page: int
    score: float
    section: str = ""
    chunk_type: str = ""
    question: str = ""


@dataclass
class WeChatMessageResponse:
    user_id: str
    route: RouteType
    reply_mode: ReplyMode
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    intent: str = ""
    intent_confidence: float = 0.0
    action: str = ""
    ticket_id: int | None = None
    intent_probabilities: dict[str, float] = field(default_factory=dict)
    answer_confidence: float = 0.0
    answer_supported: bool = False
    needs_handoff: bool = False
    confidence_reason: str = ""
    query_strategy: str = ""
    query_strategy_name: str = ""
    query_strategy_reason: str = ""


class SessionStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    HANDOFF_PENDING = "handoff_pending"
    NEED_CLARIFICATION = "need_clarification"
    CLOSED = "closed"


@dataclass
class SessionTurn:
    turn_index: int
    user_message: str
    assistant_message: str = ""
    summary: str = ""
    intent: str = ""
    intent_confidence: float = 0.0
    route: str = ""
    reply_mode: str = ""
    answer_confidence: float = 0.0
    needs_handoff: bool = False
    created_at: str = ""


@dataclass
class SessionSnapshot:
    session_id: str
    user_id: str
    channel: str = "streamlit"
    started_at: str = ""
    updated_at: str = ""
    ended_at: str | None = None
    status: SessionStatus = SessionStatus.OPEN
    end_reason: str = ""
    turn_count: int = 0

    last_intent: str = ""
    last_route: str = ""
    last_reply_mode: str = ""
    last_answer_confidence: float = 0.0
    last_needs_handoff: bool = False
    final_summary: str = ""
    turns: list = field(default_factory=list)

    def touch(self) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        if not self.started_at:
            self.started_at = now
        self.updated_at = now

    def append_turn(
        self,
        user_message: str,
        assistant_message: str,
        *,
        intent: str = "",
        intent_confidence: float = 0.0,
        route: str = "",
        reply_mode: str = "",
        answer_confidence: float = 0.0,
        needs_handoff: bool = False,
    ) -> SessionTurn:
        turn = SessionTurn(
            turn_index=len(self.turns) + 1,
            user_message=user_message,
            assistant_message=assistant_message,
            intent=intent,
            intent_confidence=intent_confidence,
            route=route,
            reply_mode=reply_mode,
            answer_confidence=answer_confidence,
            needs_handoff=needs_handoff,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        self.turns.append({
            "turn_index": turn.turn_index,
            "user_message": turn.user_message,
            "assistant_message": turn.assistant_message,
            "summary": turn.summary,
            "intent": turn.intent,
            "intent_confidence": turn.intent_confidence,
            "route": turn.route,
            "reply_mode": turn.reply_mode,
            "answer_confidence": turn.answer_confidence,
            "needs_handoff": turn.needs_handoff,
            "created_at": turn.created_at,
        })
        self.turn_count = len(self.turns)
        return turn
