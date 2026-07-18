from typing import Dict
from app.agents.tools.talent_tools import TOOL_SCHEMAS as TALENT_SCHEMAS
from app.agents.tools.rag_tools import RAG_TOOL_SCHEMAS

TOOL_SCHEMAS: Dict[str, dict] = {}
TOOL_SCHEMAS.update(TALENT_SCHEMAS)
TOOL_SCHEMAS.update(RAG_TOOL_SCHEMAS)
