"""其他类 / 低置信兜底：LLM 闲聊（不检索）。"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from settings import LLM_MODEL_BASE_URL, LLM_MODEL_NAME, get_aliyun_api_key
from services.bc_rag_agent import CASUAL_SYSTEM_PROMPT
from services.models import IntentResult, ReplyMode, RouteType, WeChatMessageResponse
from services.rag_retriever import sanitize_user_reply

logger = logging.getLogger(__name__)

CASUAL_OTHER_PROMPT = """你是抖音店铺的智能客服，正在跟来买商品的买家聊天——要像真人客服尽快回消息。
对方的问题属于闲聊或与购物无关。

【怎么说】
- 口语化、短句、自然
- 可以用「亲」等，语气亲切
- 禁止套话、Markdown
- 回复就是发给买家的话，不要加括号备注

【说多少】
- 1～2 句话，尽量 60 字以内，方便尽快回复
- 可自然带一句（商品咨询/物流/售后相关），别硬推销

【说什么】
- 不编造具体价格、库存、物流时效
- 保持友好，适当引导到购物相关话题即可"""


class CasualHandler:
    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = init_chat_model(
                model=LLM_MODEL_NAME,
                model_provider="openai",
                api_key=get_aliyun_api_key(),
                base_url=LLM_MODEL_BASE_URL,
                temperature=0.3,
            )
        return self._llm

    def _build_messages(self, message: str, *, fallback: bool,
                        history: list[dict] | None = None,
                        recent_history_limit: int = 10) -> list:
        system = CASUAL_OTHER_PROMPT if not fallback else CASUAL_SYSTEM_PROMPT
        messages = [SystemMessage(content=system)]

        if history:
            for msg in history[-recent_history_limit:]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=message))
        return messages

    def prepare_stream(
        self,
        user_id: str,
        message: str,
        intent: IntentResult,
        *,
        fallback: bool = False,
        history: list[dict] | None = None,
        recent_history_limit: int = 10,
    ):
        from dataclasses import dataclass

        @dataclass
        class CasualStreamPrepare:
            user_id: str
            message: str
            intent: IntentResult
            fallback: bool
            history: list[dict] | None
            recent_history_limit: int
            llm_messages: list

        return CasualStreamPrepare(
            user_id=user_id,
            message=message,
            intent=intent,
            fallback=fallback,
            history=history,
            recent_history_limit=recent_history_limit,
            llm_messages=self._build_messages(message, fallback=fallback, history=history,
                                               recent_history_limit=recent_history_limit),
        )

    def stream_answer(self, prepared) -> Iterator[str]:
        for chunk in self._get_llm().stream(prepared.llm_messages):
            if chunk.content:
                yield chunk.content

    def finalize_stream(self, prepared, full_text: str) -> WeChatMessageResponse:
        route = RouteType.FALLBACK if prepared.fallback else RouteType.CASUAL_CHAT
        return WeChatMessageResponse(
            user_id=prepared.user_id,
            route=route,
            reply_mode=ReplyMode.CASUAL,
            answer=sanitize_user_reply(full_text.strip()),
            intent=prepared.intent.category.value,
            intent_confidence=prepared.intent.confidence,
            action=prepared.intent.action,
            intent_probabilities=prepared.intent.probabilities,
        )

    def handle(
        self,
        user_id: str,
        message: str,
        intent: IntentResult,
        *,
        fallback: bool = False,
        history: list[dict] | None = None,
        recent_history_limit: int = 10,
    ) -> WeChatMessageResponse:
        prepared = self.prepare_stream(user_id, message, intent, fallback=fallback,
                                       history=history, recent_history_limit=recent_history_limit)
        parts: list[str] = []
        for piece in self.stream_answer(prepared):
            parts.append(piece)
        return self.finalize_stream(prepared, "".join(parts))
