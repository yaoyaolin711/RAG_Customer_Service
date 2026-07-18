"""投诉/交易强信号：避免 FAQ 缓存抢走分流。"""

from __future__ import annotations

# 明确投诉表达（子集命中即可），咨询式「质量怎么样」不在此列
COMPLAINT_HINTS = (
    "质量有问题",
    "质量太差",
    "质量不行",
    "质量很差",
    "假货",
    "伪劣",
    "投诉",
    "维权",
    "消协",
    "12315",
    "曝光",
    "骗子",
    "欺诈",
    "太气人",
    "气死了",
    "差评",
    "完全不能用",
    "坏了没用",
    "破损严重",
    "一直不处理",
    "拒不处理",
    "态度太差",
    "骗人",
    "坑人",
    "退款拖",
    "退款一直",
)


def looks_like_complaint(text: str) -> bool:
    msg = (text or "").strip()
    if not msg:
        return False
    return any(k in msg for k in COMPLAINT_HINTS)
