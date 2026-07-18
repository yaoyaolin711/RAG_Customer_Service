"""Redis/MySQL 浼氳瘽鎸佷箙鍖栧啋鐑熸祴璇曪紙闇€瑕佹湰鍦版湇鍔″凡鍚姩锛夈€?""

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
from services.session_store_mysql import get_mysql_session_archive
from services.session_store_redis import get_redis_session_store


def main():
    redis_store = get_redis_session_store()
    print("Redis ping:", redis_store.ping())

    snap = redis_store.create("wx_test_user_001", channel="streamlit")
    snap.append_turn("娴嬭瘯闂", "娴嬭瘯鍥炵瓟", intent="鍜ㄨ绫?, route="rag_agent")
    snap.last_route = "rag_agent"
    redis_store.update(snap)
    print("Redis created session:", snap.session_id, "turns=", len(snap.turns))

    archive = get_mysql_session_archive()
    archive.upsert_session(snap)
    print("MySQL upsert ok")

    snap.status = SessionStatus.RESOLVED
    snap.end_reason = "smoke_test"
    snap.ended_at = snap.updated_at
    snap.final_summary = "smoke test summary"
    redis_store.mark_closed(snap, status=SessionStatus.RESOLVED, reason="smoke_test")
    archive.finalize_session(snap)
    print("Finalize ok")


if __name__ == "__main__":
    main()

