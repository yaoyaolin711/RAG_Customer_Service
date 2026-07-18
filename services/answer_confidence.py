"""基于问题 + 召回片段 + 答案的二次置信评估。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from settings import LLM_MODEL_BASE_URL, LLM_MODEL_NAME, get_aliyun_api_key
from services.models import RetrievedChunk
from services.qa_slot import detect_user_slots, is_hard_commitment, is_soft_consult
from services.rag_retriever import format_chunks_for_prompt

logger = logging.getLogger(__name__)

ANSWER_JUDGE_PROMPT = """你是一个客服回答质检器，只负责评估候选答案的可信度，不负责改写答案。

你必须只依据以下三部分内容判断：
1. 用户原问题
2. 召回资料
3. 候选答案

评估规则：
- 如果候选答案的主题与用户问题明显无关（例如问发货/包邮，却答清洗/材质/功效），必须判定 needs_handoff=true，confidence ≤ 0.2
- 如果候选答案包含召回资料中没有明确支持的具体事实、数字、政策、承诺、时效，必须降分
- 如果答案只是保守表达“我帮你确认下/我去核实下”，不算幻觉
- 如果召回资料不足以支持候选答案，应判定需要转人工
- 输出必须是 JSON，不能输出 Markdown，不能输出额外解释

请返回如下 JSON：
{{
  "supported": true,
  "confidence": 0.0,
  "needs_handoff": false,
  "reason": "一句话说明原因"
}}
"""


@dataclass
class AnswerConfidenceResult:
    confidence: float
    supported: bool
    needs_handoff: bool
    reason: str


class AnswerConfidenceJudge:
    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = init_chat_model(
                model=LLM_MODEL_NAME,
                model_provider="openai",
                api_key=get_aliyun_api_key(),
                base_url=LLM_MODEL_BASE_URL,
                temperature=0,
            )
        return self._llm

    @staticmethod
    def _fallback_result(reason: str, *, needs_handoff: bool = True) -> AnswerConfidenceResult:
        return AnswerConfidenceResult(
            confidence=0.0,
            supported=False,
            needs_handoff=needs_handoff,
            reason=reason,
        )

    @staticmethod
    def _normalize_result(payload: dict) -> AnswerConfidenceResult:
        confidence = payload.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return AnswerConfidenceResult(
            confidence=confidence,
            supported=bool(payload.get("supported", False)),
            needs_handoff=bool(payload.get("needs_handoff", False)),
            reason=str(payload.get("reason", "")).strip() or "未提供评估原因",
        )

    @staticmethod
    def _parse_json(text: str) -> dict:
        raw = text.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = next((p for p in parts if "{" in p and "}" in p), raw)
            raw = raw[raw.find("{") : raw.rfind("}") + 1]
        return json.loads(raw)

    def evaluate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        answer: str,
    ) -> AnswerConfidenceResult:
        if not answer.strip():
            return self._fallback_result("答案为空，无法确认内容是否可信")
        if not chunks:
            slots = detect_user_slots(question)
            if is_soft_consult(question, slots) and not is_hard_commitment(question):
                return AnswerConfidenceResult(
                    confidence=0.45,
                    supported=False,
                    needs_handoff=False,
                    reason="无召回片段，软咨询允许保守回复",
                )
            return self._fallback_result("没有有效召回片段，答案缺少证据支持")

        evidence = format_chunks_for_prompt(chunks)
        judge_input = (
            f"用户原问题：\n{question.strip()}\n\n"
            f"召回资料：\n{evidence}\n\n"
            f"候选答案：\n{answer.strip()}\n"
        )
        try:
            result = self._get_llm().invoke(
                [
                    SystemMessage(content=ANSWER_JUDGE_PROMPT),
                    HumanMessage(content=judge_input),
                ]
            )
            content = getattr(result, "content", result)
            payload = self._parse_json(str(content))
            return self._normalize_result(payload)
        except Exception as exc:
            logger.exception("答案置信评估失败")
            return self._fallback_result(f"评估失败：{exc}")


_judge: AnswerConfidenceJudge | None = None


def get_answer_confidence_judge() -> AnswerConfidenceJudge:
    global _judge
    if _judge is None:
        _judge = AnswerConfidenceJudge()
    return _judge


def evaluate_answer_confidence(
    question: str,
    chunks: list[RetrievedChunk],
    answer: str,
) -> AnswerConfidenceResult:
    return get_answer_confidence_judge().evaluate(question, chunks, answer)
