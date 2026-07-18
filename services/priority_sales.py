"""已废弃：达人高意向跟进，已由意图识别分流取代。"""

from __future__ import annotations

import warnings

from services.models import ReplyMode, RouteType, WeChatMessageResponse


class PrioritySalesService:
    """遗留模块，不再接入主流程。"""

    def _build_answer(self, message: str) -> str:
        _ = message
        return "亲，有什么商品或订单问题直接说就行，我马上帮你看～"

    def handle(self, user_id: str, message: str, **_) -> WeChatMessageResponse:
        warnings.warn("PrioritySalesService 已废弃", DeprecationWarning, stacklevel=2)
        return WeChatMessageResponse(
            user_id=user_id,
            route=RouteType.RAG_AGENT,
            reply_mode=ReplyMode.CASUAL,
            answer=self._build_answer(message),
            sources=[],
        )
