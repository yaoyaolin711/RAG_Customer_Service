"""浼氳瘽缁撴潫鐘舵€佸垽瀹氬崟娴嬶紙涓嶄緷璧栧閮ㄦ湇鍔★級銆?""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
for _P in [_REPO_ROOT / "crm_agent" / "crm_agent", _REPO_ROOT / "RAG_mode" / "mode"]:
    if (_P / "settings.py").exists():
        sys.path.append(str(_P))
        break

from services.models import ReplyMode, RouteType, SessionStatus, WeChatMessageResponse
from services.session_status import detect_end_trigger


def _resp(route: RouteType) -> WeChatMessageResponse:
    return WeChatMessageResponse(
        user_id="u1",
        route=route,
        reply_mode=ReplyMode.CASUAL,
        answer="ok",
    )


def test_manual_clear_closes():
    d = detect_end_trigger(user_text="x", response=_resp(RouteType.RAG_AGENT), manual_clear=True)
    assert d is not None
    assert d.status == SessionStatus.CLOSED


def test_close_phrase_resolves():
    d = detect_end_trigger(user_text="濂界殑璋㈣阿", response=_resp(RouteType.RAG_AGENT))
    assert d is not None
    assert d.status == SessionStatus.RESOLVED


def test_handoff_route_sets_handoff_pending():
    d = detect_end_trigger(user_text="x", response=_resp(RouteType.MANUAL_HANDOFF))
    assert d is not None
    assert d.status == SessionStatus.HANDOFF_PENDING


def test_transaction_sets_handoff_pending():
    d = detect_end_trigger(user_text="x", response=_resp(RouteType.TRANSACTION))
    assert d is not None
    assert d.status == SessionStatus.HANDOFF_PENDING

