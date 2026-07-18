from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ToolInput(BaseModel):
    pass


class ToolOutput(BaseModel):
    success: bool = True
    result: Optional[Any] = None
    error: Optional[str] = None


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    input_model: type = ToolInput

    @abstractmethod
    def execute(self, input_data: Dict) -> ToolOutput:
        pass

    def __call__(self, input_data: Dict) -> ToolOutput:
        try:
            validated = self.input_model(**input_data)
            return self.execute(validated.model_dump())
        except Exception as e:
            return ToolOutput(success=False, error=str(e))


def tool(name: str, description: str):
    def decorator(cls):
        cls.name = name
        cls.description = description
        return cls
    return decorator
