from .executor import ProductCatalogTool
from .schema import TOOL_SCHEMA

PRODUCT_CATALOG_TOOL = ProductCatalogTool()

__all__ = ["PRODUCT_CATALOG_TOOL", "TOOL_SCHEMA"]
