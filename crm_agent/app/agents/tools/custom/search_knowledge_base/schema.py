TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "检索 RAG 知识库，查找与买家问题相关的 FAQ、商品信息、物流售后等文档片段。回复前必须先调用此工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用户问题，用于在知识库中做向量检索"
                }
            },
            "required": ["query"]
        }
    }
}
