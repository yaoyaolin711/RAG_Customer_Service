from typing import Dict
from pydantic import Field
from app.agents.tools.base import BaseTool, ToolInput, ToolOutput, tool


class QaCacheLookupInput(ToolInput):
    message: str = Field(..., description="用户消息")


@tool(name="qa_cache_lookup", description="查询 FAQ 缓存（Redis 精确匹配 + BGE 语义检索），命中直接返回标准答案。")
class QaCacheLookupTool(BaseTool):
    input_model = QaCacheLookupInput
    name = "qa_cache_lookup"
    description = "查询 FAQ 缓存（Redis 精确匹配 + BGE 语义检索），命中直接返回标准答案。"

    def execute(self, input_data: Dict) -> ToolOutput:
        from services.qa_cache import lookup_qa_cache

        message = (input_data.get("message") or "").strip()
        if not message:
            return ToolOutput(success=False, error="message 不能为空")

        try:
            hit = lookup_qa_cache(message)
            if hit is None:
                return ToolOutput(success=True, result={"hit": False})
            return ToolOutput(success=True, result={
                "hit": True,
                "faq_id": hit.faq_id,
                "question": hit.question_text,
                "answer": hit.answer,
                "score": hit.score,
                "match_type": hit.match_type,
                "qa_relevance": hit.qa_relevance,
            })
        except Exception as e:
            return ToolOutput(success=False, error=f"FAQ 缓存查询失败: {e}")
