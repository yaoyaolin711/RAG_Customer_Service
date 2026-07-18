"""
店铺买家消息统一入口 — 意图识别主路由

用法:
    from services.wechat_handler import WeChatMessageHandler

    handler = WeChatMessageHandler()
    response = handler.handle_message(
        user_id="buyer_001",
        message="这款多久发货？",
    )
    reply_text = response.answer
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from services.bc_rag_agent import BCRagAgentService
from services.casual_handler import CasualHandler
from services.complaint_handler import ComplaintHandler
from services.complaint_signals import looks_like_complaint
from services.intent_classifier import classify_intent
from services.intent_router import resolve_route
from services.models import (
    IntentCategory,
    IntentResult,
    ReplyMode,
    RouteType,
    WeChatMessageRequest,
    WeChatMessageResponse,
)
from services.transaction_handler import TransactionHandler

logger = logging.getLogger(__name__)


@dataclass
class MessageStreamSession:
    """消息处理会话：即时回复或流式生成。"""

    instant: WeChatMessageResponse | None = None
    text_stream: Iterator[str] | None = None
    finalize: Callable[[str], WeChatMessageResponse] | None = None
    intent: IntentResult | None = None


class WeChatMessageHandler:
    """用户消息 → BERT 意图识别 → 分流处理。"""

    def __init__(self):
        self._bc_agent = BCRagAgentService()
        self._transaction = TransactionHandler()
        self._complaint = ComplaintHandler()
        self._casual = CasualHandler()

    def _empty_response(self, user_id: str) -> WeChatMessageResponse:
        return WeChatMessageResponse(
            user_id=user_id,
            route=RouteType.UNSUPPORTED,
            reply_mode=ReplyMode.CASUAL,
            answer="您好，请问有什么可以帮您？",
            sources=[],
        )

    @staticmethod
    def _allow_faq_cache(intent: IntentResult, message: str) -> bool:
        """投诉/交易绝不走 FAQ 缓存，避免「质量有问题」命中使用说明。"""
        if looks_like_complaint(message):
            return False
        if intent.category in (IntentCategory.COMPLAINT, IntentCategory.TRANSACTION):
            return False
        return True

    def _dispatch(
        self,
        user_id: str,
        message: str,
        intent: IntentResult,
        route: RouteType,
    ) -> WeChatMessageResponse:
        if route == RouteType.RAG_AGENT:
            if self._allow_faq_cache(intent, message):
                cache_resp = self._try_qa_cache(user_id, message, intent)
                if cache_resp is not None:
                    return cache_resp
            return self._bc_agent.handle(user_id, message, intent=intent)
        if route == RouteType.TRANSACTION:
            return self._transaction.handle(user_id, message, intent)
        if route == RouteType.COMPLAINT_HANDOFF:
            return self._complaint.handle(user_id, message, intent)
        return self._casual.handle(user_id, message, intent, fallback=(route == RouteType.FALLBACK))

    def _try_qa_cache(
        self,
        user_id: str,
        message: str,
        intent: IntentResult,
    ) -> WeChatMessageResponse | None:
        """咨询类：缓存 exact / 语义检索命中则直接返回答案。"""
        if not self._allow_faq_cache(intent, message):
            return None
        try:
            from services.qa_cache import hit_to_response, lookup_qa_cache

            hit = lookup_qa_cache(message)
            if hit is None:
                return None
            logger.info(
                "QA 缓存命中 user=%s match=%s faq_id=%s score=%.3f",
                user_id,
                hit.match_type,
                hit.faq_id,
                hit.score,
            )
            return hit_to_response(user_id, message, hit, intent=intent)
        except Exception:
            logger.exception("QA 缓存查询失败，回退 RAG")
            return None

    def handle_message(
        self,
        user_id: str,
        message: str,
        **extra,
    ) -> WeChatMessageResponse:
        request = WeChatMessageRequest(
            user_id=user_id,
            message=message.strip(),
            extra=extra,
        )

        if not request.message:
            return self._empty_response(user_id)

        intent = classify_intent(request.message)
        decision = resolve_route(intent)

        # 仅咨询/闲聊路径可 FAQ 优先；投诉/交易必须先分流
        if decision.route == RouteType.RAG_AGENT or decision.route in (
            RouteType.CASUAL_CHAT,
            RouteType.FALLBACK,
        ):
            cache_resp = self._try_qa_cache(user_id, request.message, intent)
            if cache_resp is not None:
                logger.info(
                    "FAQ缓存命中 user=%s intent=%s conf=%.3f",
                    user_id,
                    intent.category.value,
                    intent.confidence,
                )
                return cache_resp

        logger.info(
            "意图路由 user=%s intent=%s conf=%.3f route=%s reason=%s",
            user_id,
            intent.category.value,
            intent.confidence,
            decision.route.value,
            decision.reason,
        )
        return self._dispatch(user_id, request.message, intent, decision.route)

    def prepare_message_stream(
        self,
        user_id: str,
        message: str,
        **extra,
    ) -> MessageStreamSession:
        request = WeChatMessageRequest(
            user_id=user_id,
            message=message.strip(),
            extra=extra,
        )

        if not request.message:
            return MessageStreamSession(instant=self._empty_response(user_id))

        intent = classify_intent(request.message)
        decision = resolve_route(intent)
        route = decision.route

        if route in (RouteType.RAG_AGENT, RouteType.CASUAL_CHAT, RouteType.FALLBACK):
            cache_resp = self._try_qa_cache(user_id, request.message, intent)
            if cache_resp is not None:
                return MessageStreamSession(instant=cache_resp, intent=intent)

        if route == RouteType.RAG_AGENT:
            prepared = self._bc_agent.prepare_stream(user_id, request.message, intent=intent)

            def finalize(text: str) -> WeChatMessageResponse:
                return self._bc_agent.finalize_stream(prepared, text)

            return MessageStreamSession(
                text_stream=self._bc_agent.stream_answer(prepared),
                finalize=finalize,
                intent=intent,
            )

        if route in (RouteType.CASUAL_CHAT, RouteType.FALLBACK):
            prepared = self._casual.prepare_stream(
                user_id,
                request.message,
                intent,
                fallback=(route == RouteType.FALLBACK),
            )

            def finalize_casual(text: str) -> WeChatMessageResponse:
                return self._casual.finalize_stream(prepared, text)

            return MessageStreamSession(
                text_stream=self._casual.stream_answer(prepared),
                finalize=finalize_casual,
                intent=intent,
            )

        return MessageStreamSession(
            instant=self._dispatch(user_id, request.message, intent, route),
            intent=intent,
        )
