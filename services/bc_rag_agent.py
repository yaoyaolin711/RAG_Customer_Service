"""
RAG Agent — 咨询类知识库检索 + 生成
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from settings import (
    ANSWER_CONFIDENCE_THRESHOLD,
    ANSWER_RELEVANCE_THRESHOLD,
    LLM_MODEL_BASE_URL,
    LLM_MODEL_NAME,
    get_aliyun_api_key,
)
from services.answer_confidence import evaluate_answer_confidence
from services.handoff_handler import ManualHandoffHandler
from services.models import IntentResult, ReplyMode, RetrievedChunk, RouteType, WeChatMessageResponse
from services.qa_slot import build_multi_query_plan, detect_user_slots, is_hard_commitment, is_soft_consult
from services.rag_retriever import (
    filter_answer_chunks,
    filter_relevant_chunks,
    format_chunks_for_prompt,
    retrieve_with_strategy,
    sanitize_user_reply,
)
from services.semantic_match import score_question_answer

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """你是抖音店铺的智能客服，正在跟来买商品的买家聊天——要像真人客服尽快回消息。
用户提问后，你手头有一些内部参考材料，请基于材料回答商品信息、规格参数、活动价格、物流发货、售后规则等问题。

【怎么说】
- 口语化、短句、自然，不要书面腔
- 可以用「亲」等亲切称呼，语气友好但不油
- 禁止套话：不要「首先/其次/综上所述/希望以上内容对您有帮助」
- 不要 Markdown、不要编号列表，除非用户明确要对比多个选项
- 回复就是发给买家的话，不要加括号备注，不要暴露任何内部逻辑或文件名

【说多少】
- 优先 1～3 句话讲清楚，整体尽量控制在 80 字以内，方便尽快回复
- 简单问题 1 句即可；稍复杂也尽量不超过 120 字
- 只答用户问的点，不主动扩写背景、不重复用户原话

【说什么】
- 仅使用参考材料中的事实，不编造价格、库存、物流时效、售后政策
- 为缩短字数时，只能删冗余修饰和套话，不能删或改关键事实（数字、条件、流程、限制必须保留）
- 若参考材料不够，口语化说「这块我帮你确认下马上回你」，别装懂

参考材料：
{context}
"""

CASUAL_SYSTEM_PROMPT = """你是抖音店铺的智能客服，正在跟来买商品的买家聊天——要像真人客服尽快回消息。
当前问题在参考材料里没查到直接相关的内容。

【怎么说】
- 口语化、短句、自然
- 可以用「亲」等，语气亲切
- 禁止套话、Markdown，不要假装已经查到了资料
- 回复就是发给买家的话，不要加括号备注

【说多少】
- 1～2 句话，尽量 60 字以内，方便尽快回复
- 简单回应后可自然带一句（商品/物流/售后相关），别硬推销

【说什么】
- 不编造具体价格、库存、物流时效
- 对方若提到查单、退款、发货，口语化表示「我帮你查一下」即可"""


@dataclass
class BCRagStreamPrepare:
    """RAG 检索完成后，待流式生成的 LLM 上下文。"""

    user_id: str
    message: str
    reply_mode: ReplyMode
    relevant_chunks: list[RetrievedChunk]
    llm_messages: list
    intent: IntentResult | None = None
    query_strategy: str = "direct"
    query_strategy_name: str = "直接检索"
    query_strategy_reason: str = ""
    retrieval_queries: list[str] | None = None


def _attach_intent(response: WeChatMessageResponse, intent: IntentResult | None) -> WeChatMessageResponse:
    if intent is None:
        return response
    response.intent = intent.category.value
    response.intent_confidence = intent.confidence
    response.action = intent.action
    response.intent_probabilities = intent.probabilities
    return response


class BCRagAgentService:
    """咨询类 RAG + 未命中闲聊。"""

    def __init__(self):
        self._llm = None
        self._handoff = ManualHandoffHandler()

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

    def prepare_stream(
        self,
        user_id: str,
        message: str,
        intent: IntentResult | None = None,
    ) -> BCRagStreamPrepare:
        slots = detect_user_slots(message)
        forced_plan = build_multi_query_plan(message, slots)
        retrieval = retrieve_with_strategy(message, plan=forced_plan)
        all_chunks = retrieval.chunks
        plan = retrieval.strategy
        relevant_chunks = filter_answer_chunks(filter_relevant_chunks(all_chunks))

        if relevant_chunks:
            context = format_chunks_for_prompt(relevant_chunks)
            llm_messages = [
                SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context)),
                HumanMessage(content=message),
            ]
            reply_mode = ReplyMode.RAG
        else:
            llm_messages = [
                SystemMessage(content=CASUAL_SYSTEM_PROMPT),
                HumanMessage(content=message),
            ]
            reply_mode = ReplyMode.CASUAL
            relevant_chunks = []

        return BCRagStreamPrepare(
            user_id=user_id,
            message=message,
            reply_mode=reply_mode,
            relevant_chunks=relevant_chunks,
            llm_messages=llm_messages,
            intent=intent,
            query_strategy=plan.strategy.value,
            query_strategy_name=plan.strategy_name,
            query_strategy_reason=plan.reason,
            retrieval_queries=list(plan.queries),
        )

    def stream_answer(self, prepared: BCRagStreamPrepare) -> Iterator[str]:
        for chunk in self._get_llm().stream(prepared.llm_messages):
            if chunk.content:
                yield chunk.content

    def finalize_stream(
        self,
        prepared: BCRagStreamPrepare,
        full_text: str,
    ) -> WeChatMessageResponse:
        answer = sanitize_user_reply(full_text.strip())
        slots = detect_user_slots(prepared.message)
        soft_no_hit = (
            prepared.reply_mode == ReplyMode.CASUAL
            and not prepared.relevant_chunks
            and is_soft_consult(prepared.message, slots)
            and not is_hard_commitment(prepared.message)
        )

        # 软咨询空召回：允许保守生成，跳过转人工闸
        if soft_no_hit:
            response = WeChatMessageResponse(
                user_id=prepared.user_id,
                route=RouteType.RAG_AGENT,
                reply_mode=ReplyMode.CASUAL,
                answer=answer,
                sources=[],
                answer_confidence=0.45,
                answer_supported=False,
                needs_handoff=False,
                confidence_reason="soft_consult_no_hit",
                query_strategy=prepared.query_strategy,
                query_strategy_name=prepared.query_strategy_name,
                query_strategy_reason=prepared.query_strategy_reason,
            )
            return _attach_intent(response, prepared.intent)

        # 1) 快速问-答语义校验（BGE）：明显答非所问直接转人工
        try:
            qa_score = score_question_answer(prepared.message, answer)
        except Exception:
            logger.exception("问-答语义校验失败")
            qa_score = 0.0
        if qa_score < ANSWER_RELEVANCE_THRESHOLD:
            response = self._handoff.handle(
                user_id=prepared.user_id,
                message=prepared.message,
                intent=prepared.intent,
                answer_confidence=qa_score,
                confidence_reason=(
                    f"问-答语义相关度过低({qa_score:.3f}<{ANSWER_RELEVANCE_THRESHOLD:.2f})，"
                    "停止输出并转人工"
                ),
            )
            response.sources = prepared.relevant_chunks
            response.query_strategy = prepared.query_strategy
            response.query_strategy_name = prepared.query_strategy_name
            response.query_strategy_reason = prepared.query_strategy_reason
            return response

        # 2) LLM 证据一致性评估
        judge = evaluate_answer_confidence(
            question=prepared.message,
            chunks=prepared.relevant_chunks,
            answer=answer,
        )
        if judge.needs_handoff or judge.confidence < ANSWER_CONFIDENCE_THRESHOLD:
            response = self._handoff.handle(
                user_id=prepared.user_id,
                message=prepared.message,
                intent=prepared.intent,
                answer_confidence=judge.confidence,
                confidence_reason=judge.reason,
            )
            response.sources = prepared.relevant_chunks
            response.query_strategy = prepared.query_strategy
            response.query_strategy_name = prepared.query_strategy_name
            response.query_strategy_reason = prepared.query_strategy_reason
            return response
        response = WeChatMessageResponse(
            user_id=prepared.user_id,
            route=RouteType.RAG_AGENT,
            reply_mode=prepared.reply_mode,
            answer=answer,
            sources=prepared.relevant_chunks,
            answer_confidence=judge.confidence,
            answer_supported=judge.supported,
            needs_handoff=False,
            confidence_reason=f"{judge.reason} | qa_relevance={qa_score:.3f}",
            query_strategy=prepared.query_strategy,
            query_strategy_name=prepared.query_strategy_name,
            query_strategy_reason=prepared.query_strategy_reason,
        )
        return _attach_intent(response, prepared.intent)

    def handle(
        self,
        user_id: str,
        message: str,
        intent: IntentResult | None = None,
    ) -> WeChatMessageResponse:
        prepared = self.prepare_stream(user_id, message, intent=intent)
        parts: list[str] = []
        for piece in self.stream_answer(prepared):
            parts.append(piece)
        return self.finalize_stream(prepared, "".join(parts))
