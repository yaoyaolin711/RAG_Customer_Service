import httpx
import json
import time
from typing import Iterator, Optional, Dict, Any
from app.config import config
from settings import get_deepseek_key
from app.utils.retry import retry


class QwenClient:
    def __init__(self):
        llm_config = config.llm
        active_provider = llm_config.get("active_provider", "aliyun")

        provider_config = llm_config.get(active_provider, {})
        self.provider = provider_config.get("provider", active_provider)
        self.base_url = provider_config.get("base_url", "")
        self.api_key = provider_config.get("api_key") or get_deepseek_key()
        self.model = provider_config.get("model", "qwen-plus")

        self.temperature = llm_config.get("temperature", 0.7)
        self.max_tokens = llm_config.get("max_tokens", 4096)
        self.timeout = llm_config.get("timeout", 120.0)
        self.max_retries = llm_config.get("max_retries", 3)

        context_config = config.context
        self.context_max_tokens = context_config.get("max_tokens", 4000)

    def _truncate_messages(self, messages: list, max_tokens: int = None) -> list:
        return messages

    @retry(max_attempts=3, delay=2.0, backoff=2.0, exceptions=[httpx.HTTPError, httpx.TimeoutException])
    def invoke(self, messages: list, tools: list = None, tool_choice: str = "auto",
               session_id: str = None, user_id: str = None, agent_name: str = None,
               model: str = None, base_url: str = None, api_key: str = None) -> dict:
        start_time = time.time()

        _model = model or self.model
        _base_url = base_url or self.base_url
        _api_key = api_key or self.api_key

        headers = {"Content-Type": "application/json"}
        if _api_key:
            headers["Authorization"] = f"Bearer {_api_key}"

        payload = {
            "model": _model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # 私信回复不需要长思考链，关闭后语气更自然、响应更快
        if agent_name != "unified_reply":
            payload["thinking"] = {"type": "enabled"}

        if tools:
            payload["tools"] = tools

        if tool_choice and tools:
            payload["tool_choice"] = tool_choice

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{_base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                if response.status_code != 200:
                    body = response.text[:2000]
                    print(f"[LLM Error] {response.status_code}: {body}")
                response.raise_for_status()
                result = response.json()

                return result
        except Exception as e:
            raise

    def extract_content(self, response: dict) -> str:
        if "choices" not in response:
            return str(response)

        choice = response["choices"][0]
        message = choice.get("message", {})

        if message.get("tool_calls"):
            return json.dumps({
                "tool_calls": message["tool_calls"],
                "content": message.get("content", "")
            }, ensure_ascii=False)

        return message.get("content", "")

    def stream(self, messages: list, tools: list = None, tool_choice: str = "auto",
               session_id: str = None, user_id: str = None, agent_name: str = None) -> Iterator[Dict[str, Any]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "enabled"},
            "stream": True
        }

        if tools:
            payload["tools"] = tools

        if tool_choice and tools:
            payload["tool_choice"] = tool_choice

        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status_code != 200:
                        body = response.read().decode()[:2000]
                        print(f"[LLM Stream Error] {response.status_code}: {body}")
                        yield {"error": f"API error {response.status_code}: {body[:200]}"}
                        return
                    for line in response.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue

                        data = line[6:].strip()
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            yield chunk
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield {"error": str(e), "type": "error"}
            return


llm = QwenClient()
