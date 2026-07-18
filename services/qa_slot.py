"""用户问题槽位 ↔ FAQ 副类对齐：轻量规则，不走模型。

用途：缓存命中前否决「商品名很像、主题不对」的假高分（如问规格命中简介）；
多槽时拆子问题检索；软/硬主题分流空召回是否转人工。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from services.qa_variants import clean_sub_class

if TYPE_CHECKING:
    from services.query_strategy import QueryStrategyPlan

# 强制走 RAG 的「要细节/要参数」说法（即使有简介缓存也拒收）
_DETAIL_PHRASES: tuple[str, ...] = (
    "详细说说",
    "详细讲讲",
    "详细介绍一下",
    "能详细点",
    "能详细一点",
    "具体参数",
    "详细参数",
    "具体规格",
    "详细规格",
    "再说详细点",
    "讲详细点",
)

# 显式主题词：出现即计入对应槽位（可与其它槽并存）
_EXPLICIT_SLOT_MARKERS: list[tuple[str, tuple[str, ...]]] = [
    ("使用方法", ("食用方法", "使用方法", "怎么喝", "怎么用", "怎么泡", "怎么煮", "煮多久", "怎么吃", "怎么使用", "怎么冲", "用法")),
    ("适合人群", ("适合什么人", "适合谁", "适合人群", "适用人群", "哪些人能", "哪些人不能", "孕妇能", "孕妇可以", "哺乳期", "小孩能", "儿童能", "老人能")),
    ("注意事项", ("注意事项", "有什么注意", "忌口", "能不能一起", "可以一起", "禁忌", "副作用吗")),
    ("材质", ("什么材质", "是什么面料", "什么面料", "材质是", "材质吗", "材质安全", "面料", "材质")),
    ("功效", ("有什么功效", "有什么用", "什么功效", "功效是", "能改善", "有用吗", "功效", "效果怎么样", "什么效果")),
    ("规格", ("有什么规格", "什么规格", "哪些规格", "规格是", "规格呢", "规格吗", "多少克", "几盒", "几袋", "多少袋", "多少盒", "容量", "净含量", "几克", "规格", "尺寸", "码数", "多少斤")),
    ("简介", ("简介", "产品介绍", "商品介绍")),
]

# 弱槽短语：仅在未命中任何显式槽时，才把整句算作「简介」
_WEAK_INTRO_PHRASES: tuple[str, ...] = (
    "介绍一下",
    "介绍下",
    "是什么",
    "什么产品",
    "怎么样",
    "咋样",
    "了解一下",
    "说说这款",
    "讲讲这款",
)

# 兼容旧触发表结构（测试/文档）；实际走 _EXPLICIT + _WEAK
_SLOT_TRIGGERS: list[tuple[str, tuple[str, ...]]] = [
    *[(s, ps) for s, ps in _EXPLICIT_SLOT_MARKERS if s != "简介"],
    ("简介", ("简介", *_WEAK_INTRO_PHRASES)),
]

# 槽位 → 副类兼容关键词（副类清理后 contains 任一即兼容）
_SLOT_SUB_KEYS: dict[str, tuple[str, ...]] = {
    "规格": ("规格", "码数", "尺码", "尺寸", "净含量", "容量", "重量", "几克"),
    "简介": ("简介", "卖点", "产品介绍", "商品介绍"),
    "使用方法": ("使用方法", "食用方法", "每日食用", "用法", "怎么用", "使用推荐", "使用时间"),
    "适合人群": ("适合人群", "适用人群", "适合谁", "孕妇", "哺乳", "不适宜", "禁用"),
    "材质": ("材质", "面料", "填充"),
    "功效": ("功效", "作用", "效果", "改善"),
    "注意事项": ("注意事项", "注意", "忌口", "禁忌", "过敏", "副作用"),
}

_SOFT_SLOTS: frozenset[str] = frozenset(
    {"简介", "功效", "使用方法", "适合人群", "材质", "注意事项"}
)

# 硬承诺主题：空召回仍转人工
_HARD_COMMITMENT_PHRASES: tuple[str, ...] = (
    "多少钱",
    "价格",
    "价位",
    "几块",
    "优惠",
    "便宜",
    "打折",
    "活动价",
    "包邮",
    "运费",
    "邮费",
    "运费险",
    "发货",
    "几天到",
    "多久到",
    "物流",
    "快递",
    "到哪了",
    "退款",
    "退货",
    "仅退款",
    "售后",
    "订单",
    "发票",
    "库存",
    "断货",
    "预售",
    "质保",
    "保修",
)

# 硬互斥：这些副类关键词出现时，绝不能当作「规格」命中
_SPEC_BLOCK_SUBS: tuple[str, ...] = (
    "简介",
    "功效",
    "作用",
    "口感",
    "卖点",
    "使用方法",
    "食用方法",
    "适合人群",
    "适用人群",
    "注意事项",
    "材质",
)

_JOIN_TRIM_RE = re.compile(r"^[\s的之，,、和与及]+|[\s的之，,、和与及]+$")


def force_rag_detail(message: str) -> bool:
    """用户明确要细节/参数时，拒收简介类缓存，交给 RAG。"""
    text = (message or "").strip()
    if not text:
        return False
    return any(p in text for p in _DETAIL_PHRASES)


def _find_first_marker(text: str, phrases: tuple[str, ...]) -> tuple[int, str] | None:
    """返回 (起始下标, 命中短语)；取最长短语中最早出现者。"""
    best: tuple[int, str] | None = None
    for p in sorted(phrases, key=len, reverse=True):
        if not p:
            continue
        idx = text.find(p)
        if idx < 0:
            continue
        if best is None or idx < best[0] or (idx == best[0] and len(p) > len(best[1])):
            best = (idx, p)
    return best


def detect_user_slots(message: str) -> list[str]:
    """从用户问题识别需求槽位（按首次出现顺序去重）。

    显式主题词始终计入；弱介绍短语仅在无显式槽时计入「简介」。
    """
    text = (message or "").strip()
    if not text:
        return []

    hits: list[tuple[int, str]] = []
    seen: set[str] = set()
    for slot, phrases in _EXPLICIT_SLOT_MARKERS:
        found = _find_first_marker(text, phrases)
        if found is None:
            continue
        idx, _ = found
        if slot not in seen:
            seen.add(slot)
            hits.append((idx, slot))

    if not hits:
        for p in sorted(_WEAK_INTRO_PHRASES, key=len, reverse=True):
            if p and p in text:
                return ["简介"]
        return []

    hits.sort(key=lambda x: x[0])
    return [slot for _, slot in hits]


def extract_product_prefix(message: str, slots: list[str] | None = None) -> str:
    """取第一个显式主题词之前的文本作为商品前缀。"""
    text = (message or "").strip()
    if not text:
        return ""
    slot_list = slots if slots is not None else detect_user_slots(text)
    if not slot_list:
        return ""

    earliest = len(text)
    for slot, phrases in _EXPLICIT_SLOT_MARKERS:
        if slot not in slot_list:
            continue
        found = _find_first_marker(text, phrases)
        if found is not None:
            earliest = min(earliest, found[0])

    if earliest >= len(text):
        return ""
    prefix = _JOIN_TRIM_RE.sub("", text[:earliest].strip())
    # 去掉末尾问答语气碎片
    prefix = re.sub(r"[？?！!。．.]+$", "", prefix).strip()
    if len(prefix) < 2 or len(prefix) > 40:
        return ""
    return prefix


def build_multi_query_plan(message: str, slots: list[str] | None = None):
    """多槽时构造强制子问题检索 plan；不足 2 槽返回 None。"""
    from services.query_strategy import STRATEGY_NAMES, QueryStrategy, QueryStrategyPlan

    text = (message or "").strip()
    slot_list = slots if slots is not None else detect_user_slots(text)
    if len(slot_list) < 2:
        return None

    use_slots = slot_list[:3]
    prefix = extract_product_prefix(text, use_slots)
    queries: list[str] = []
    for slot in use_slots:
        if prefix:
            q = f"{prefix} {slot}"
        else:
            q = f"{text} {slot}"
        if q not in queries:
            queries.append(q)

    if len(queries) < 2:
        return None

    return QueryStrategyPlan(
        strategy=QueryStrategy.MULTI_QUERY,
        strategy_name=STRATEGY_NAMES[QueryStrategy.MULTI_QUERY.value],
        original_query=text,
        queries=queries,
        keywords=[],
        reason=f"槽位规则强制子问题检索 slots={use_slots}",
    )


def is_hard_commitment(message: str) -> bool:
    """价格/物流/退款/优惠等硬承诺主题。"""
    text = (message or "").strip()
    if not text:
        return False
    return any(p in text for p in _HARD_COMMITMENT_PHRASES)


def is_soft_consult(message: str, slots: list[str] | None = None) -> bool:
    """仅软咨询槽且无硬主题：空召回可保守生成不转人工。"""
    text = (message or "").strip()
    if not text or is_hard_commitment(text):
        return False
    slot_list = slots if slots is not None else detect_user_slots(text)
    if not slot_list:
        # 无槽但像商品介绍宽问：不算硬主题时视为软咨询宽匹配
        return any(p in text for p in _WEAK_INTRO_PHRASES) or "简介" in text
    return all(s in _SOFT_SLOTS for s in slot_list)


def sub_compatible(slot: str | None, sub_class: str) -> bool:
    """用户槽位与 FAQ 副类是否对齐。slot 为空时一律放行（宽问不强制否决）。"""
    if not slot:
        return True
    sub = clean_sub_class(sub_class or "")
    if not sub:
        return False

    # 规格硬否决：简介/功效等一律不兼容
    if slot == "规格":
        for blocked in _SPEC_BLOCK_SUBS:
            if blocked in sub and "规格" not in sub:
                return False

    keys = _SLOT_SUB_KEYS.get(slot)
    if not keys:
        # 未知槽位：宽松放行，避免误拦
        return True
    return any(k in sub for k in keys)


def slot_allows_item(slots: list[str], sub_class: str) -> bool:
    """候选 FAQ 是否通过槽位闸。槽位为空放行；多槽位由调用方提前 miss。"""
    if not slots:
        return True
    return sub_compatible(slots[0], sub_class)
