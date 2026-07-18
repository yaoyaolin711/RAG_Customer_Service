"""商品咨询强信号：意图被标成「其他类」时纠偏到 RAG。"""

from __future__ import annotations

from services.complaint_signals import looks_like_complaint
from services.qa_slot import detect_user_slots


# 显式主题词（与 qa_slot 显式标记对齐的短清单，便于快速判断）
_PRODUCT_THEME_WORDS: tuple[str, ...] = (
    "简介",
    "功效",
    "规格",
    "使用方法",
    "食用方法",
    "适合人群",
    "适用人群",
    "注意事项",
    "材质",
    "介绍一下",
    "介绍下",
    "怎么喝",
    "怎么用",
    "怎么吃",
)


def looks_like_product_consult(message: str) -> bool:
    """含商品主题槽位/关键词的咨询句；投诉强信号排除。"""
    text = (message or "").strip()
    if not text:
        return False
    if looks_like_complaint(text):
        return False
    if detect_user_slots(text):
        return True
    return any(w in text for w in _PRODUCT_THEME_WORDS)
