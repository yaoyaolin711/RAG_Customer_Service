"""交易类 Mock 处理（预留真实 API）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from services.models import IntentResult, ReplyMode, RouteType, WeChatMessageResponse


@dataclass
class TransactionResult:
    answer: str
    api_payload: dict[str, Any] = field(default_factory=dict)
    mock: bool = True


class TransactionHandler:
    """交易类问题 Mock 回复，后期对接真实订单 API。"""

    def handle(
        self,
        user_id: str,
        message: str,
        intent: IntentResult,
    ) -> WeChatMessageResponse:
        result = self.mock_query(user_id, message)
        return WeChatMessageResponse(
            user_id=user_id,
            route=RouteType.TRANSACTION,
            reply_mode=ReplyMode.TRANSACTION,
            answer=result.answer,
            intent=intent.category.value,
            intent_confidence=intent.confidence,
            action=intent.action,
            intent_probabilities=intent.probabilities,
        )

    def mock_query(self, user_id: str, message: str) -> TransactionResult:
        # TODO: 对接真实订单/物流 API
        answer = "亲，我这边帮你查一下订单和物流，稍等我确认后马上回你哈。"
        if any(kw in message for kw in ("退款", "退货")):
            answer = "退款这块我帮你对接售后同事处理，确认了马上跟你说。"
        return TransactionResult(
            answer=answer,
            api_payload={"user_id": user_id, "query": message, "status": "pending"},
            mock=True,
        )
