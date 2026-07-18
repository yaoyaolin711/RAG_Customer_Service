from typing import Dict
from pydantic import Field
from app.agents.tools.base import BaseTool, ToolInput, ToolOutput, tool


class SearchKnowledgeBaseInput(ToolInput):
    query: str = Field(..., description="用户问题，用于在知识库中做向量检索")


@tool(name="search_knowledge_base", description="检索 RAG 知识库，查找与买家问题相关的 FAQ、商品信息、物流售后等文档片段。回复前必须先调用此工具。")
class SearchKnowledgeBaseTool(BaseTool):
    input_model = SearchKnowledgeBaseInput
    name = "search_knowledge_base"
    description = "检索 RAG 知识库，查找与买家问题相关的 FAQ、商品信息、物流售后等文档片段。回复前必须先调用此工具。"

    def _chunk_to_dict(self, c):
        return {
            "content": c.content,
            "source": c.source,
            "chunk_id": c.chunk_id,
            "page": c.page,
            "score": c.score,
            "section": c.section,
            "chunk_type": c.chunk_type,
            "question": c.question,
        }

    def execute(self, input_data: Dict) -> ToolOutput:
        from app.agents.tools.rag_tools import search_knowledge_base as _search

        query = (input_data.get("query") or "").strip()
        if not query:
            return ToolOutput(success=False, error="query 不能为空")
        try:
            result = _search(query)
            if isinstance(result.get("chunks"), list):
                result["chunks"] = [self._chunk_to_dict(c) for c in result["chunks"]]
            return ToolOutput(success=True, result=result)
        except Exception as e:
            return ToolOutput(success=False, error=f"知识库检索失败: {e}")
