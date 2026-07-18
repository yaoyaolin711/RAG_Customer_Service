"""已废弃：原 A/B/C 标签分流已移除，请使用 intent_router.resolve_route。"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from services.models import RouteType


@dataclass
class RouteDecision:
    route: RouteType
    reason: str


def route_by_user_tag(user_tag) -> RouteDecision:
    warnings.warn(
        "route_by_user_tag 已废弃，请使用 BERT 意图识别 + intent_router.resolve_route",
        DeprecationWarning,
        stacklevel=2,
    )
    return RouteDecision(
        route=RouteType.RAG_AGENT,
        reason=f"legacy tag={user_tag!r}，默认走 RAG",
    )
