"""
搴楅摵涔板娑堟伅鎰忓浘鍒嗘祦 + RAG Agent 妯℃嫙娴嬭瘯

鐢ㄦ硶:
  python scripts/wechat_simulate.py --message "杩欐鏃犵嚎鑰虫満澶氬皯閽憋紵"
  python scripts/wechat_simulate.py --message "鎴戠殑璁㈠崟浠€涔堟椂鍊欏彂璐?
  python scripts/wechat_simulate.py --message "瀹㈡湇鎬佸害澶樊浜嗘垜瑕佹姇璇?
  python scripts/wechat_simulate.py --message "浠婂ぉ澶╂皵涓嶉敊"
"""
import argparse
import logging
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
for _P in [os.path.join(_REPO_ROOT, "crm_agent", "crm_agent"), os.path.join(_REPO_ROOT, "RAG_mode", "mode")]:
    if os.path.isfile(os.path.join(_P, "settings.py")):
        sys.path.append(_P)
        break

from services.wechat_handler import WeChatMessageHandler
from vectorstore import check_milvus_connection

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="鎰忓浘鍒嗘祦 + RAG Agent 妯℃嫙锛堝簵閾轰拱瀹讹級")
    parser.add_argument("--user-id", default="buyer_test_001", help="涔板鐢ㄦ埛 ID")
    parser.add_argument("--message", required=True, help="涔板娑堟伅")
    args = parser.parse_args()

    check_milvus_connection()
    handler = WeChatMessageHandler()
    response = handler.handle_message(args.user_id, args.message)

    print("\n=== 澶勭悊缁撴灉 ===")
    print(f"鐢ㄦ埛: {response.user_id}")
    print(f"鎰忓浘: {response.intent} (缃俊搴?{response.intent_confidence:.4f})")
    print(f"鍔ㄤ綔: {response.action}")
    print(f"鍒嗘祦: {response.route.value}")
    print(f"鍥炲妯″紡: {response.reply_mode.value}")
    if response.ticket_id:
        print(f"鎶曡瘔宸ュ崟: #{response.ticket_id}")
    if response.sources:
        print(f"妫€绱㈠懡涓? {len(response.sources)} 鏉?)
        for s in response.sources:
            print(f"  - [{s.score:.2f}] {s.source}: {s.content[:60]}...")
    else:
        print("妫€绱㈠懡涓? 0 鏉?)
    print(f"\n鍥炲:\n{response.answer}")


if __name__ == "__main__":
    main()

