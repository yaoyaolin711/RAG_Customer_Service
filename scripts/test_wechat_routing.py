"""鎰忓浘璺敱涓?BERT 鍒嗙被鍣ㄦ祴璇曘€?""

from __future__ import annotations

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

from services.intent_classifier import classify_intent
from services.intent_router import resolve_route
from services.models import IntentCategory, RouteType


class TestIntentRouter:
    def test_consult_route(self):
        intent = classify_intent("杩欐鏈夎繍璐归櫓鍚楋紵")
        decision = resolve_route(intent)
        assert intent.category == IntentCategory.CONSULT
        assert decision.route == RouteType.RAG_AGENT

    def test_transaction_route(self):
        intent = classify_intent("鎴戠殑璁㈠崟浠€涔堟椂鍊欏彂璐?)
        decision = resolve_route(intent)
        assert intent.category == IntentCategory.TRANSACTION
        assert decision.route == RouteType.TRANSACTION

    def test_complaint_route(self):
        intent = classify_intent("瀹㈡湇鎬佸害澶樊浜嗘垜瑕佹姇璇?)
        decision = resolve_route(intent)
        assert intent.category == IntentCategory.COMPLAINT
        assert decision.route == RouteType.COMPLAINT_HANDOFF

    def test_other_route(self):
        intent = classify_intent("浠婂ぉ澶╂皵涓嶉敊")
        decision = resolve_route(intent)
        assert intent.category == IntentCategory.OTHER
        assert decision.route in (RouteType.CASUAL_CHAT, RouteType.FALLBACK)


class TestWeChatHandlerIntent:
    def test_complaint_instant_handoff(self):
        from services.wechat_handler import WeChatMessageHandler

        handler = WeChatMessageHandler()
        session = handler.prepare_message_stream("u1", "浣犱滑鏈嶅姟澶樊浜嗚鎶曡瘔")
        assert session.instant is not None
        assert session.instant.route == RouteType.COMPLAINT_HANDOFF
        assert session.instant.ticket_id is not None

    def test_transaction_instant(self):
        from services.wechat_handler import WeChatMessageHandler

        handler = WeChatMessageHandler()
        session = handler.prepare_message_stream("u1", "甯垜鏌ヤ竴涓嬬墿娴佸埌鍝簡")
        assert session.instant is not None
        assert session.instant.route == RouteType.TRANSACTION

    def test_consult_goes_rag(self):
        from services.wechat_handler import WeChatMessageHandler

        handler = WeChatMessageHandler()
        session = handler.prepare_message_stream("u1", "杩欐灏虹爜鎬庝箞閫夛紵澶氫箙鍙戣揣锛?)
        assert session.text_stream is not None
        assert session.intent is not None
        assert session.intent.category == IntentCategory.CONSULT


if __name__ == "__main__":
    t1 = TestIntentRouter()
    for name in ("test_consult_route", "test_transaction_route", "test_complaint_route", "test_other_route"):
        getattr(t1, name)()
        print("OK", name)
    try:
        t2 = TestWeChatHandlerIntent()
        for name in (
            "test_complaint_instant_handoff",
            "test_transaction_instant",
            "test_consult_goes_rag",
        ):
            getattr(t2, name)()
            print("OK", name)
    except ModuleNotFoundError as e:
        print("Handler tests skipped (missing deps):", e)

