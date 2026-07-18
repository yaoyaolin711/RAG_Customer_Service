"""浼氳瘽閫愯疆瀛樺偍鍗曟祴锛圫essionSnapshot + Redis 寰€杩旓級銆?""

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

from services.models import SessionSnapshot, SessionStatus
from services.session_store_redis import get_redis_session_store


def test_append_turn_syncs_turn_count():
    snap = SessionSnapshot(session_id="test_session", user_id="u1")
    snap.append_turn("闂涓€", "鍥炵瓟涓€", intent="鍜ㄨ绫?, route="rag_agent")
    snap.append_turn("闂浜?, "鍥炵瓟浜?, intent="浜ゆ槗绫?, route="transaction")
    snap.append_turn("闂涓?, "鍥炵瓟涓?, intent="鎶曡瘔绫?, route="complaint_handoff")

    assert snap.turn_count == 3
    assert len(snap.turns) == 3
    assert snap.turns[0]["user_message"] == "闂涓€"
    assert snap.turns[0]["assistant_message"] == "鍥炵瓟涓€"
    assert snap.turns[2]["turn_index"] == 3


def test_redis_turns_roundtrip():
    store = get_redis_session_store()
    if not store.ping():
        print("SKIP: Redis not available")
        return

    snap = store.create("wx_test_turns_user")
    snap.append_turn("q1", "a1", intent="鍜ㄨ绫?, route="rag_agent")
    snap.append_turn("q2", "a2", route="transaction")
    store.update(snap)

    loaded = store.get(snap.session_id)
    assert loaded is not None
    assert len(loaded.turns) == 2
    assert loaded.turn_count == 2
    assert loaded.turns[0]["user_message"] == "q1"
    assert loaded.turns[1]["assistant_message"] == "a2"

    loaded.status = SessionStatus.CLOSED
    store.mark_closed(loaded, status=SessionStatus.CLOSED, reason="test_cleanup")


def test_empty_session_has_no_turns():
    snap = SessionSnapshot(session_id="empty", user_id="u1")
    assert snap.turn_count == 0
    assert snap.turns == []


if __name__ == "__main__":
    test_empty_session_has_no_turns()
    test_append_turn_syncs_turn_count()
    test_redis_turns_roundtrip()
    print("All session turn tests passed.")

