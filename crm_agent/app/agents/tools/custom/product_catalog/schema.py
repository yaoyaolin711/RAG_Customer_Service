TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "product_catalog",
        "description": "查询店铺产品目录，按分类列出商品名称和简介。支持关键词筛选和分页。",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "筛选关键词（空则返回全部分类）"
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "偏移量，用于分页"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "返回条数，最大 50"
                }
            }
        }
    }
}
