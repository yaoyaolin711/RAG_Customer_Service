"""会话状态与结束触发判定。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from services.models import RouteType, SessionStatus, WeChatMessageResponse


_CLOSE_PHRASES = [
    "好的",
    "好滴",
    "ok",
    "OK",
    "收到",
    "明白了",
    "知道了",
    "行",
    "可以",
    "辛苦了",
    "谢谢",
    "先这样",
    "就这样",
    "没问题",
]

_CLOSE_RE = re.compile(r"(" + "|".join(re.escape(p) for p in _CLOSE_PHRASES) + r")", re.IGNORECASE)


@dataclass
class EndDecision:
    status: SessionStatus
    reason: str


def detect_end_trigger(
    *,
    user_text: str,
    response: WeChatMessageResponse,
    manual_clear: bool = False,
) -> EndDecision | None:
    """检测是否触发结束/转人工等会话状态变更。返回 None 表示不触发。"""

    if manual_clear:
        return EndDecision(status=SessionStatus.CLOSED, reason="manual_clear")

    # 路由事件：优先级最高
    if response.route in (RouteType.MANUAL_HANDOFF, RouteType.COMPLAINT_HANDOFF):
        return EndDecision(status=SessionStatus.HANDOFF_PENDING, reason=response.route.value)
    if response.route == RouteType.TRANSACTION:
        # 交易类通常需要后续跟进（人工或异步），此处先标记为 handoff_pending
        return EndDecision(status=SessionStatus.HANDOFF_PENDING, reason="transaction_pending")

    # 模型评估触发
    if getattr(response, "needs_handoff", False):
        return EndDecision(status=SessionStatus.HANDOFF_PENDING, reason="needs_handoff")

    # 用户结束语触发
    if user_text and _CLOSE_RE.search(user_text.strip()):
        return EndDecision(status=SessionStatus.RESOLVED, reason="user_close_phrase")

    return None

