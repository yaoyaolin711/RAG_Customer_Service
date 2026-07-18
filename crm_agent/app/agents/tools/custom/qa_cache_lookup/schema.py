TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "qa_cache_lookup",
        "description": "查询 FAQ 缓存（Redis 精确匹配 + BGE 语义检索），命中直接返回标准答案，不命中可以调 search_knowledge_base 查 RAG。",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "用户消息"
                }
            },
            "required": ["message"]
        }
    }
}
