from typing import Dict
from pydantic import Field
from app.agents.tools.base import BaseTool, ToolInput, ToolOutput, tool


class EvaluateConfidenceInput(ToolInput):
    question: str = Field(..., description="用户原问题")
    answer: str = Field(..., description="需要评估的候选答案")
    contexts: list[str] = Field(default_factory=list, description="RAG 检索到的知识库片段（纯文本列表）")


@tool(name="evaluate_answer_confidence", description="评估候选答案的可信度，返回置信度分数和是否需要转人工。必须在生成答案后调用。")
class EvaluateConfidenceTool(BaseTool):
    input_model = EvaluateConfidenceInput
    name = "evaluate_answer_confidence"
    description = "评估候选答案的可信度，返回置信度分数和是否需要转人工。必须在生成答案后调用。"

    def execute(self, input_data: Dict) -> ToolOutput:
        from services.answer_confidence import evaluate_answer_confidence
        from services.models import RetrievedChunk

        question = (input_data.get("question") or "").strip()
        answer = (input_data.get("answer") or "").strip()
        contexts = input_data.get("contexts") or []

        if not question:
            return ToolOutput(success=False, error="question 不能为空")
        if not answer:
            return ToolOutput(success=False, error="answer 不能为空")

        chunks = [
            RetrievedChunk(content=c[:500], source="", chunk_id="", page=0, score=0.0)
            for c in contexts if c.strip()
        ]
        try:
            result = evaluate_answer_confidence(question, chunks, answer)
            return ToolOutput(success=True, result={
                "confidence": result.confidence,
                "supported": result.supported,
                "needs_handoff": result.needs_handoff,
                "reason": result.reason,
            })
        except Exception as e:
            return ToolOutput(success=False, error=f"置信评估失败: {e}")
