"""妲戒綅澶氶棶銆佽蒋/纭富棰樸€佸瓙闂 plan銆佹剰鍥剧籂鍋忓啋鐑熸祴璇曘€?""

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

from services.answer_confidence import AnswerConfidenceJudge
from services.consult_signals import looks_like_product_consult
from services.intent_router import resolve_route
from services.models import IntentCategory, IntentResult, RouteType
from services.qa_slot import (
    build_multi_query_plan,
    detect_user_slots,
    is_hard_commitment,
    is_soft_consult,
)
from services.query_strategy import QueryStrategy


def test_detect_multi_slots_sangfu():
    slots = detect_user_slots("妗戜紡鑼剁畝浠嬪拰浣跨敤鏂规硶")
    assert "绠€浠? in slots and "浣跨敤鏂规硶" in slots
    assert len(slots) >= 2
    # 鎸夐娆″嚭鐜帮細绠€浠嬪湪鍓?
    assert slots.index("绠€浠?) < slots.index("浣跨敤鏂规硶")


def test_detect_spec_and_crowd():
    slots = detect_user_slots("鏈変粈涔堣鏍硷紵閫傚悎浠€涔堜汉缇わ紵")
    assert slots == ["閫傚悎浜虹兢", "瑙勬牸"] or set(slots) == {"瑙勬牸", "閫傚悎浜虹兢"}
    assert len(slots) >= 2


def test_weak_intro_only():
    assert detect_user_slots("浜旀寚姣涙涓冨懗鑼舵€庝箞鏍?) == ["绠€浠?]
    # 鏁堟灉鎬庝箞鏍?鈫?鍔熸晥锛堝急銆屾€庝箞鏍枫€嶄笉鍙﹀姞绠€浠嬶級
    assert detect_user_slots("鏁堟灉鎬庝箞鏍?) == ["鍔熸晥"]


def test_build_multi_query_plan():
    text = "妗戜紡鑼剁畝浠嬪拰浣跨敤鏂规硶"
    slots = detect_user_slots(text)
    plan = build_multi_query_plan(text, slots)
    assert plan is not None
    assert plan.strategy == QueryStrategy.MULTI_QUERY
    assert len(plan.queries) >= 2
    joined = " ".join(plan.queries)
    assert "绠€浠? in joined and "浣跨敤鏂规硶" in joined
    assert any("妗戜紡鑼? in q for q in plan.queries)


def test_soft_hard():
    assert is_soft_consult("妗戜紡鑼剁殑鍔熸晥鏄粈涔?)
    assert not is_hard_commitment("妗戜紡鑼剁殑鍔熸晥鏄粈涔?)
    assert is_hard_commitment("妗戜紡鑼跺灏戦挶鍖呴偖鍚?)
    assert not is_soft_consult("妗戜紡鑼跺灏戦挶鍖呴偖鍚?)


def test_empty_chunks_soft_no_handoff():
    judge = AnswerConfidenceJudge()
    result = judge.evaluate("妗戜紡鑼剁殑鍔熸晥鏄粈涔?, [], "浜诧紝鐩墠璧勬枡閲岃繖鍧楁垜鍐嶅府浣犲涓€涓嬶綖")
    assert result.needs_handoff is False
    assert result.confidence == 0.45


def test_empty_chunks_hard_handoff():
    judge = AnswerConfidenceJudge()
    result = judge.evaluate("妗戜紡鑼跺灏戦挶鍖呴偖鍚?, [], "浜诧紝浠锋牸杩欏潡鎴戝府浣犵‘璁や笅")
    assert result.needs_handoff is True


def test_product_consult_route_other():
    assert looks_like_product_consult("妗戜紡鑼剁畝浠?)
    intent = IntentResult(
        category=IntentCategory.OTHER,
        confidence=0.43,
        action="chat_or_fallback",
        raw_text="妗戜紡鑼剁畝浠?,
        is_fallback=False,
    )
    decision = resolve_route(intent)
    assert decision.route == RouteType.RAG_AGENT
    assert "绾犲亸" in decision.reason


def test_complaint_still_priority():
    intent = IntentResult(
        category=IntentCategory.OTHER,
        confidence=0.5,
        action="chat",
        raw_text="璐ㄩ噺鏈夐棶棰樻垜瑕佹姇璇?,
        is_fallback=False,
    )
    decision = resolve_route(intent)
    assert decision.route == RouteType.COMPLAINT_HANDOFF


if __name__ == "__main__":
    test_detect_multi_slots_sangfu()
    test_detect_spec_and_crowd()
    test_weak_intro_only()
    test_build_multi_query_plan()
    test_soft_hard()
    test_empty_chunks_soft_no_handoff()
    test_empty_chunks_hard_handoff()
    test_product_consult_route_other()
    test_complaint_still_priority()
    print("ALL_PASS")

