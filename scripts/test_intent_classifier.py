"""鎰忓浘鍒嗙被鍣ㄥ姞杞戒笌鎺ㄧ悊鍐掔儫娴嬭瘯銆?""

from __future__ import annotations

import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
for _P in [_REPO_ROOT / "crm_agent" / "crm_agent", _REPO_ROOT / "RAG_mode" / "mode"]:
    if (_P / "settings.py").exists():
        sys.path.append(str(_P))
        break

from services.intent_classifier import classify_intent, get_intent_classifier


def main():
    samples = [
        "杩欎釜澶氬皯閽?,
        "鎴戠殑璁㈠崟浠€涔堟椂鍊欏彂璐?,
        "瀹㈡湇鎬佸害澶樊浜嗘垜瑕佹姇璇?,
        "浠婂ぉ澶╂皵涓嶉敊",
    ]
    t0 = time.time()
    clf = get_intent_classifier()
    first = classify_intent(samples[0])
    load_ms = (time.time() - t0) * 1000
    print(f"棣栨鎺ㄧ悊锛堝惈鍔犺浇锛? {load_ms:.0f}ms")
    print(f"  {samples[0]!r} -> {first.category.value} conf={first.confidence:.4f}")

    for text in samples[1:]:
        r = clf.classify(text)
        print(f"  {text!r} -> {r.category.value} conf={r.confidence:.4f} action={r.action}")

    print("OK")


if __name__ == "__main__":
    main()

