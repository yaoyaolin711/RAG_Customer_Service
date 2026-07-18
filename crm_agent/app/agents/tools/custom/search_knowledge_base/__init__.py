from .executor import SearchKnowledgeBaseTool
from .schema import TOOL_SCHEMA

SEARCH_KNOWLEDGE_BASE_TOOL = SearchKnowledgeBaseTool()

__all__ = ["SEARCH_KNOWLEDGE_BASE_TOOL", "TOOL_SCHEMA"]
