"""意图 → 路由映射。"""

from __future__ import annotations

from dataclasses import dataclass

from services.complaint_signals import looks_like_complaint
from services.consult_signals import looks_like_product_consult
from services.models import IntentCategory, IntentResult, RouteType


@dataclass
class RouteDecision:
    route: RouteType
    reason: str


def resolve_route(intent: IntentResult) -> RouteDecision:
    text = intent.raw_text or ""

    # 投诉优先：即使置信度略低 / 被标成 fallback，也不进闲聊或缓存话术
    if intent.category == IntentCategory.COMPLAINT or looks_like_complaint(text):
        return RouteDecision(
            route=RouteType.COMPLAINT_HANDOFF,
            reason="投诉类 / 投诉强信号 → 建工单转人工",
        )

    if intent.category == IntentCategory.TRANSACTION and not intent.is_fallback:
        return RouteDecision(
            route=RouteType.TRANSACTION,
            reason="交易类 → 查询订单/物流",
        )

    # 商品+主题句被 BERT 标成其他类 / 低置信时，纠偏到 RAG（而非闲聊）
    if looks_like_product_consult(text) and intent.category in (
        IntentCategory.OTHER,
        IntentCategory.CONSULT,
    ):
        if intent.category == IntentCategory.OTHER or intent.is_fallback:
            return RouteDecision(
                route=RouteType.RAG_AGENT,
                reason="商品咨询纠偏 → RAG 知识库检索",
            )

    if intent.is_fallback:
        return RouteDecision(
            route=RouteType.FALLBACK,
            reason=f"置信度 {intent.confidence:.2f} 低于阈值",
        )

    mapping = {
        IntentCategory.CONSULT: (
            RouteType.RAG_AGENT,
            "咨询类 → RAG 知识库检索",
        ),
        IntentCategory.TRANSACTION: (
            RouteType.TRANSACTION,
            "交易类 → 查询订单/物流",
        ),
        IntentCategory.OTHER: (
            RouteType.CASUAL_CHAT,
            "其他类 → LLM 闲聊",
        ),
    }
    route, reason = mapping[intent.category]
    return RouteDecision(route=route, reason=reason)
