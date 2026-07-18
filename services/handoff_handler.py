"""通用人工转接处理器。"""

from __future__ import annotations

from services.models import IntentResult, ReplyMode, RouteType, WeChatMessageResponse


class ManualHandoffHandler:
    """处理非投诉场景下的低置信人工转接。"""

    def handle(
        self,
        user_id: str,
        message: str,
        intent: IntentResult | None,
        *,
        answer_confidence: float,
        confidence_reason: str,
    ) -> WeChatMessageResponse:
        _ = message
        answer = "这块我先帮你转给同事确认一下，确认清楚后马上回你哈。"
        response = WeChatMessageResponse(
            user_id=user_id,
            route=RouteType.MANUAL_HANDOFF,
            reply_mode=ReplyMode.HANDOFF,
            answer=answer,
            answer_confidence=answer_confidence,
            answer_supported=False,
            needs_handoff=True,
            confidence_reason=confidence_reason,
        )
        if intent is not None:
            response.intent = intent.category.value
            response.intent_confidence = intent.confidence
            response.action = intent.action
            response.intent_probabilities = intent.probabilities
        return response
