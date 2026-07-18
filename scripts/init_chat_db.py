"""初始化本地会话历史 SQLite 库并写入示例数据。

用法（从仓库根目录）:
    python scripts/init_chat_db.py
    python scripts/init_chat_db.py --reset
"""

from __future__ import annotations

import argparse
import sqlite3
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

from app.services.chat_history import (  # noqa: E402
    EXPORT_DB_PATH,
    _insert_message,
    ensure_db,
)

# 涓?API.md 绀轰緥涓€鑷?
DEMO_SELF = "buyer_demo_001"
DEMO_CONTACT = "dy_session_001"

SEED_TURNS: list[tuple[str, str]] = [
    ("浣犲ソ锛屾兂闂笅灏虹爜鎬庝箞閫?, "浜蹭綘璇翠笅骞虫椂绌垮澶э紝鎴戝府浣犲涓€涓嬪昂鐮佽〃鍝?),
    ("杩欐澶氫箙鍙戣揣锛熸湁杩愯垂闄╁悧锛?, "涓€鑸笅鍗曞悗48灏忔椂鍐呭彂锛岃繍璐归櫓浠ュ晢璇﹂〉涓哄噯鍝?),
    ("鏀寔涓冨ぉ鏃犵悊鐢卞悧锛?, "鏀寔鐨勶紝绛炬敹鍚庢寜骞冲彴娴佺▼鐢宠灏辫"),
]


def _reset_db(path: Path) -> None:
    if path.is_file():
        path.unlink()


def seed_demo_data() -> int:
    ensure_db()
    conn = sqlite3.connect(str(EXPORT_DB_PATH))
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE username = ?",
            (DEMO_CONTACT,),
        ).fetchone()[0]
        if count:
            return int(count)

        for i, (incoming, outgoing) in enumerate(SEED_TURNS):
            day = f"2026-07-0{6 - i}"
            _insert_message(
                conn,
                contact_username=DEMO_CONTACT,
                sender_username=DEMO_CONTACT,
                content=incoming,
                dt=f"{day} 10:{10 + i * 2:02d}:00",
            )
            _insert_message(
                conn,
                contact_username=DEMO_CONTACT,
                sender_username=DEMO_SELF,
                content=outgoing,
                dt=f"{day} 10:{11 + i * 2:02d}:00",
            )
        conn.commit()
        return len(SEED_TURNS) * 2
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="鍒濆鍖?exported_chats.db")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="鍒犻櫎宸叉湁搴撳悗閲嶆柊鍒涘缓",
    )
    args = parser.parse_args()

    if args.reset:
        _reset_db(EXPORT_DB_PATH)

    ensure_db()
    inserted = seed_demo_data()
    print(f"鏁版嵁搴撹矾寰? {EXPORT_DB_PATH}")
    print(f"绀轰緥鑱旂郴浜? {DEMO_CONTACT}锛堟垜鏂?ID: {DEMO_SELF}锛?)
    print(f"绀轰緥娑堟伅鏉℃暟: {inserted}")


if __name__ == "__main__":
    main()


