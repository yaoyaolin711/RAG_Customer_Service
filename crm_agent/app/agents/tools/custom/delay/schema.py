TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "delay",
        "description": "等待指定的秒数，用于控制操作节奏、避免触发频控",
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 180,
                    "description": "等待秒数（最大 180 秒）"
                },
                "reason": {
                    "type": "string",
                    "description": "等待原因说明（可选）"
                }
            },
            "required": ["seconds"]
        }
    }
}
