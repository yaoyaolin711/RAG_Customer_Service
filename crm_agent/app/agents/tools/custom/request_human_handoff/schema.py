TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "request_human_handoff",
        "description": "当你无法解答用户问题、用户要求转人工、或需要人工介入时调用此工具。调用后会返回转人工话术给用户。",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "为什么需要转人工（如：用户要求、知识库无相关信息、超出能力范围）"
                }
            },
            "required": ["reason"]
        }
    }
}
