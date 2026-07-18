import json
from pathlib import Path
from typing import Dict
from pydantic import Field
from app.agents.tools.base import BaseTool, ToolInput, ToolOutput, tool


class ProductCatalogInput(ToolInput):
    keyword: str = Field("", description="筛选关键词（空则返回全部分类）")
    offset: int = Field(0, ge=0, description="偏移量，用于分页")
    limit: int = Field(10, ge=1, le=50, description="返回条数，最大 50")


_PRODUCTS: list[dict] | None = None


def _get_products() -> list[dict]:
    global _PRODUCTS
    if _PRODUCTS is None:
        path = Path(__file__).parent / "products.json"
        if path.exists():
            try:
                _PRODUCTS = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                _PRODUCTS = []
        else:
            _PRODUCTS = []
    return _PRODUCTS


@tool(name="product_catalog", description="查询店铺产品目录，按分类列出商品名称和简介。支持关键词筛选和分页。")
class ProductCatalogTool(BaseTool):
    input_model = ProductCatalogInput
    name = "product_catalog"
    description = "查询店铺产品目录，按分类列出商品名称和简介。支持关键词筛选和分页。"

    def execute(self, input_data: Dict) -> ToolOutput:
        keyword = (input_data.get("keyword") or "").strip().lower()
        offset = max(0, int(input_data.get("offset", 0)))
        limit = max(1, min(50, int(input_data.get("limit", 10))))

        items = _get_products()
        if keyword:
            items = [
                p for p in items
                if keyword in p["name"].lower()
                or keyword in p["category"].lower()
                or keyword in p["desc"].lower()
            ]

        total = len(items)
        page = items[offset:offset + limit]

        return ToolOutput(success=True, result={
            "total": total,
            "offset": offset,
            "limit": limit,
            "count": len(page),
            "has_more": offset + limit < total,
            "items": page,
        })
