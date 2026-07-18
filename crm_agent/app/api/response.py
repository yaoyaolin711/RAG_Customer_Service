"""统一 API 响应格式。"""

from __future__ import annotations

from typing import Any


def _default_data() -> dict[str, int]:
    return {"count": 0}


def api_ok(data: Any = None, message: str = "ok") -> dict:
    """成功响应：{"data": ..., "state": {"code": 0, "message": "ok"}}"""
    if data is None:
        data = _default_data()
    return {"data": data, "state": {"code": 0, "message": message}}


def api_fail(code: int, message: str, data: Any = None) -> dict:
    """失败响应：state.code 非 0。"""
    if data is None:
        data = _default_data()
    elif isinstance(data, dict) and "count" not in data:
        data = {"count": 0, **data}
    return {"data": data, "state": {"code": code, "message": message}}


def format_validation_errors(errors: list[dict]) -> str:
    """将 Pydantic/FastAPI 校验错误转为可读 message。"""
    parts: list[str] = []
    for err in errors:
        loc = err.get("loc") or ()
        field = ".".join(str(x) for x in loc if x not in ("body", "query", "path"))
        msg = err.get("msg") or "校验失败"
        parts.append(f"{field}: {msg}" if field else msg)
    return "; ".join(parts) or "请求参数校验失败"


def unified_json_response(payload: dict) -> Any:
    """统一 HTTP 200 + JSON 响应（业务成败看 state.code）。"""
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=200, content=payload)
