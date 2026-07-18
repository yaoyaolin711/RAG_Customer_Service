import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from settings import API_PORT

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.response import api_fail, api_ok, format_validation_errors, unified_json_response
from app.api.talents import router as talents_router
from app.api.v1 import router as v1_router
from app.api.wechat import router as wechat_router

# 仓库根目录 web/（crm_agent/crm_agent → 上两级）
_WEB_DIR = _ROOT.parent / "web"

API_DESCRIPTION = """
## RAG Agent 统一接口

面向 **抖音店铺买家** 的智能问答与客服 Agent：知识库 RAG 检索 + 意图分流 + 统一回复。
目标：尽快回复买家（约 15 秒内）；买家一视同仁，无标签分层。

### 统一响应格式

```json
{
  "data": { ... },
  "state": { "code": 0, "message": "ok" }
}
```

- `state.code = 0`：成功
- `state.code ≠ 0`：失败（HTTP 状态码仍为 200，含参数校验失败）

### 推荐使用

| 接口 | 说明 |
|------|------|
| `POST /api/v1/chat` | **主入口** — Unified Agent（RAG 检索 + 历史对话 + 生成） |
| `GET /api/v1/health` | 健康检查 |
| `GET /api/v1/meta` | 服务元信息 |
| `/console/` | IM 模拟台（静态前端） |
| `GET /docs` | Swagger 交互文档 |

### 兼容接口

| 接口 | 说明 |
|------|------|
| `POST /api/wechat/chat` | 旧路径兼容，与 `/api/v1/chat` 相同，均走 UnifiedReplyAgent |
| `POST /api/talents/{id}/simulate` | 旧路径兼容，模拟买家消息，响应字段略有差异 |

### 错误码

- `state.code = 0`：成功
- `state.code = 400`：请求参数错误（含 Pydantic 校验失败）
- `state.code = 500`：服务内部错误
"""

app = FastAPI(
    title="RAG Agent API",
    version="2.1.0",
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router, prefix="/api/v1", tags=["统一接口 v1"])
app.include_router(wechat_router, prefix="/api/wechat", tags=["旧路径兼容"])
app.include_router(talents_router, prefix="/api/talents", tags=["旧路径兼容（模拟）"])


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    """Pydantic 请求体/参数校验失败 → 统一 {data, state} 格式。"""
    return unified_json_response(
        api_fail(400, format_validation_errors(exc.errors()))
    )


@app.exception_handler(StarletteHTTPException)
async def starlette_http_exception_handler(
    request: Request, exc: StarletteHTTPException
):
    """404 等路由层 HTTP 异常（Starlette 与 FastAPI HTTPException 为不同类）。"""
    return unified_json_response(api_fail(exc.status_code, str(exc.detail)))


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return unified_json_response(api_fail(exc.status_code, str(exc.detail)))


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """未预期异常兜底。"""
    return unified_json_response(
        api_fail(500, f"{type(exc).__name__}: {exc}")
    )


@app.get("/", summary="服务首页", tags=["系统"])
def root():
    return api_ok({
        "count": 4,
        "service": "RAG Agent",
        "version": "2.1.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
        "api": "/api/v1",
        "console": "/console/",
        "endpoints": {
            "chat": "POST /api/v1/chat",
            "health": "GET /api/v1/health",
            "meta": "GET /api/v1/meta",
            "console": "/console/",
        },
    })


if _WEB_DIR.is_dir():
    app.mount(
        "/console",
        StaticFiles(directory=str(_WEB_DIR), html=True),
        name="console",
    )


@app.on_event("startup")
async def warmup_models():
    import logging as _logging

    _logger = _logging.getLogger(__name__)
    _logger.info("预热：加载模型 ...")

    try:
        from vectorstore import get_milvus_client

        get_milvus_client()
    except Exception as e:
        _logger.warning("Milvus 预热失败: %s", e)

    try:
        from embedding import get_bge_m3_model

        get_bge_m3_model()
    except Exception as e:
        _logger.warning("Embedding 预热失败: %s", e)

    try:
        from reranker import _ensure_loaded as _ensure_reranker

        _ensure_reranker()
    except Exception as e:
        _logger.warning("Reranker 预热失败: %s", e)

    try:
        from services.intent_classifier import get_intent_classifier

        get_intent_classifier()._ensure_loaded()
    except Exception as e:
        _logger.warning("意图模型预热失败: %s", e)

    _logger.info("模型预热完成")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=API_PORT, reload=True)
