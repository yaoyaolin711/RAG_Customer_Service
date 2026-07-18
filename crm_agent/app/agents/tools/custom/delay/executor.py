import time
from typing import Dict
from pydantic import Field
from agents.tools.base import BaseTool, ToolInput, ToolOutput, tool

class DelayInput(ToolInput):
    seconds: float = Field(..., ge=0, le=180, description="等待秒数（最大 180 秒）")
    reason: str = Field(default="", description="等待原因")

@tool(name="delay", description="等待指定的秒数，用于控制操作节奏、避免触发频控")
class DelayTool(BaseTool):
    input_model = DelayInput
    name = "delay"
    description = "等待指定的秒数，用于控制操作节奏、避免触发频控"

    def execute(self, input_data: Dict) -> ToolOutput:
        try:
            seconds = max(0.0, min(180.0, float(input_data.get("seconds", 0))))
            reason = input_data.get("reason", "")
            if seconds <= 0:
                return ToolOutput(success=True, result="无需等待")
            tag = f" ({reason})" if reason else ""
            time.sleep(seconds)
            return ToolOutput(success=True, result=f"已等待 {seconds:.1f} 秒{tag}")
        except Exception as e:
            return ToolOutput(success=False, error=str(e))
