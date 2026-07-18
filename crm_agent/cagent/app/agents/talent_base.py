import json
from typing import Dict, Any, Optional, List
from app.llm import llm
from app.config import config
from app.agents.tools.registry import ToolRegistry
from app.agents.tools.base import ToolOutput


class TalentAgentState:
    def __init__(self):
        self.task: str = ""
        self.tools_used: List[Dict] = []
        self.iterations: int = 0
        self.max_iterations: int = 10
        self.final_output: Any = None
        self.messages: List[Dict] = []


class TalentBaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.agent_config = config.agents.get(name, {})
        self.system_prompt = self.agent_config.get("system_prompt", "")
        self.description = self.agent_config.get("description", "")
        self.tools: List[str] = self.agent_config.get("tools", [])
        streaming_config = config.get("streaming", {})
        self.max_iterations: int = streaming_config.get("max_tool_calls", 10)
        self._summary_interval = self.agent_config.get("summary_interval",
                                                        config.get("context", {}).get("summary_interval", 5))

    def _update_progress_summary(self, state):
        if self._summary_interval <= 0:
            return
        if state.iterations - getattr(state, '_last_summary_iteration', 0) < self._summary_interval:
            return

        recent_tools = state.tools_used[-self._summary_interval:]
        if not recent_tools:
            return

        lines = []
        for t in recent_tools:
            tool_name = t.get("tool", "")
            status = "成功" if t.get("success") else "失败"
            args_str = json.dumps(t.get("input", {}), ensure_ascii=False)[:150]
            result_val = t.get("result", "")
            if isinstance(result_val, dict):
                result_str = json.dumps(result_val, ensure_ascii=False)[:200]
            else:
                result_str = str(result_val)[:200]
            lines.append(f"- 调用 {tool_name}: {status}")
            lines.append(f"  参数: {args_str}")
            if result_str:
                lines.append(f"  结果: {result_str}")

        summary_content = f"【执行进展 - 已完成 {state.iterations} 轮工具调用】\n" + "\n".join(lines)

        if len(summary_content) > 3000:
            summary_content = summary_content[:3000] + "\n...(截断)"

        if hasattr(state, '_progress_summary_index') and state._progress_summary_index is not None:
            state.messages[state._progress_summary_index]["content"] = summary_content
        else:
            insert_at = 1
            state.messages.insert(insert_at, {"role": "user", "content": summary_content})
            state._progress_summary_index = insert_at

        state._last_summary_iteration = state.iterations

    def get_tools(self) -> List[dict]:
        return ToolRegistry.get_tools_schemas(self.tools)

    def execute_tool(self, tool_name: str, input_data: Dict) -> ToolOutput:
        tool = ToolRegistry.get(tool_name)
        if not tool:
            return ToolOutput(success=False, error=f"Tool '{tool_name}' not found")
        try:
            return tool(input_data)
        except Exception as e:
            return ToolOutput(success=False, error=str(e))

    def build_messages(self, task: str = "", context: Optional[Dict] = None) -> list:
        system_content = self.system_prompt

        if context:
            parts = []

            prior_summaries = context.get("prior_summaries", [])
            if prior_summaries:
                summary_parts = ["【历史摘要】"]
                l2 = context.get("l2_summary")
                if l2:
                    summary_parts.append(f"- {l2.get('summary', '')}")
                    newer = [s for s in prior_summaries
                             if s.get("start_msg_id", 0) > l2.get("end_msg_id", 0)]
                    for s in newer:
                        summary_parts.append(f"- {s.get('summary', '')}")
                else:
                    for s in prior_summaries[:3]:
                        summary_parts.append(f"- {s.get('summary', '')}")
                parts.append("\n".join(summary_parts))

            profile = context.get("talent_profile")
            if profile:
                parts.append(f"【达人画像】{profile}")

            if parts:
                system_content += "\n\n" + "\n\n".join(parts)

        messages = [{"role": "system", "content": system_content}]

        if context:
            for msg in context.get("recent_history", []):
                if msg.get("role") != "tool":
                    messages.append(msg)

        if task:
            messages.append({"role": "user", "content": task})
        return messages

    def invoke(self, input_data: Dict) -> Dict:
        task = input_data.get("task", "")
        context = input_data.get("context", {})

        state = TalentAgentState()
        state.task = task
        state.messages = self.build_messages(task, context)
        state._last_summary_iteration = 0
        state._progress_summary_index = None

        tools = self.get_tools()
        tool_call_count = 0

        while tool_call_count < self.max_iterations:
            try:
                response = llm.invoke(state.messages, tools=tools,
                                      session_id=context.get("session_id"),
                                      agent_name=self.name)
            except Exception as e:
                return {"agent": self.name, "output": {}, "success": False, "error": str(e)}

            if isinstance(response, str):
                response = json.loads(response) if response.startswith("{") else {"choices": [{"message": {"content": response}}]}

            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                content = message.get("content", "")
                if content:
                    state.final_output = content
                break

            state.messages.append(message)

            for call in tool_calls:
                tool_call_count += 1
                state.iterations += 1

                func = call.get("function", {})
                tool_name = func.get("name", "")
                arguments = func.get("arguments", "{}")

                if tool_name not in self.tools:
                    state.messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": f"错误: 工具 '{tool_name}' 不存在或未启用"
                    })
                    continue

                try:
                    args = json.loads(arguments) if isinstance(arguments, str) else arguments
                except:
                    args = {}

                result = self.execute_tool(tool_name, args)

                state.tools_used.append({
                    "tool": tool_name,
                    "input": args,
                    "success": result.success,
                    "result": result.result,
                    "error": result.error,
                })

                if isinstance(result.result, dict):
                    result_content = json.dumps(result.result, ensure_ascii=False, indent=2)
                else:
                    result_content = str(result.result)

                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": result_content if result.success else f"错误: {result.error}",
                }
                state.messages.append(tool_message)

                if not result.success:
                    state.messages.append({
                        "role": "system",
                        "content": f"工具 '{tool_name}' 执行失败，请修正后重试。"
                    })

            self._update_progress_summary(state)

            if tool_call_count >= self.max_iterations:
                break

        if not state.final_output:
            state.final_output = message.get("content", "")

        return {
            "agent": self.name,
            "output": {
                "result": state.final_output,
                "tools_used": state.tools_used,
                "iterations": state.iterations,
            },
            "success": True,
            "error": None,
        }
