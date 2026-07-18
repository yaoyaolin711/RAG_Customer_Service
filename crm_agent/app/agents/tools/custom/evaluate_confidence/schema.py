TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "evaluate_answer_confidence",
        "description": "评估候选答案的可信度，返回置信度分数和是否需要转人工。必须在生成答案后调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "用户原问题"
                },
                "answer": {
                    "type": "string",
                    "description": "需要评估的候选答案"
                },
                "contexts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "RAG 检索到的知识库片段（纯文本列表）"
                }
            },
            "required": ["question", "answer"]
        }
    }
}
