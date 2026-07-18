from typing import Dict, Optional, List, Tuple
from app.agents.tools.base import BaseTool
from app.agents.tools.schemas import TOOL_SCHEMAS


class ToolRegistry:
    _tools: Dict[str, BaseTool] = {}
    _schemas: Dict[str, dict] = TOOL_SCHEMAS.copy()
    _custom_tools: Dict[str, bool] = {}
    _agent_tools: Dict[str, str] = {}
    _mcp_tools: Dict[str, str] = {}

    @classmethod
    def register(cls, tool: BaseTool, schema: dict = None, is_custom: bool = False, is_agent: str = None, is_mcp: str = None):
        if not tool.name:
            raise ValueError("Tool must have a name")
        cls._tools[tool.name] = tool
        if is_custom:
            cls._custom_tools[tool.name] = True
        if is_agent:
            cls._agent_tools[tool.name] = is_agent
        if is_mcp:
            cls._mcp_tools[tool.name] = is_mcp
        if schema:
            cls._schemas[tool.name] = schema

    @classmethod
    def get(cls, name: str) -> Optional[BaseTool]:
        return cls._tools.get(name)

    @classmethod
    def list_all(cls) -> List[str]:
        return list(cls._tools.keys())

    @classmethod
    def get_all(cls) -> Dict[str, BaseTool]:
        return cls._tools.copy()

    @classmethod
    def get_schema(cls, name: str) -> Optional[dict]:
        return cls._schemas.get(name)

    @classmethod
    def get_tools_schemas(cls, tool_names: List[str]) -> List[dict]:
        result = []
        for name in tool_names:
            schema = cls._schemas.get(name)
            if schema:
                result.append(schema)
        return result

    @classmethod
    def clear(cls):
        cls._tools.clear()
        cls._schemas.clear()
        cls._custom_tools.clear()
        cls._agent_tools.clear()
        cls._mcp_tools.clear()
