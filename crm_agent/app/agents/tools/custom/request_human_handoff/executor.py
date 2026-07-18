from typing import Dict
from pydantic import Field
from app.agents.tools.base import BaseTool, ToolInput, ToolOutput, tool


class RequestHandoffInput(ToolInput):
    reason: str = Field(..., description="为什么需要转人工（如：用户要求、知识库无相关信息、超出能力范围）")


@tool(name="request_human_handoff", description="当你无法解答用户问题、用户要求转人工、或需要人工介入时调用此工具。调用后会返回转人工话术给用户。")
class RequestHumanHandoffTool(BaseTool):
    input_model = RequestHandoffInput
    name = "request_human_handoff"
    description = "当你无法解答用户问题、用户要求转人工、或需要人工介入时调用此工具。"

    def execute(self, input_data: Dict) -> ToolOutput:
        reason = (input_data.get("reason") or "").strip()
        return ToolOutput(success=True, result={
            "handoff": True,
            "reason": reason or "未提供原因",
            "message": "已通知人工客服处理，请稍等哈。",
        })
