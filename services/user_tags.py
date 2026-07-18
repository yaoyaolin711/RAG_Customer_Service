"""遗留模块：达人 A/B/C 标签升级已剥离，店铺买家一视同仁。"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum


class UserTag(str, Enum):
    """兼容旧枚举；业务侧不再使用分层。"""

    A = "A"
    B = "B"
    C = "C"


@dataclass
class TagUpgradeEvent:
    """兼容旧结构；店铺主链路不再产生升级事件。"""

    user_id: str
    from_tag: UserTag
    to_tag: UserTag
    trigger_keywords: list[str]
    applied: bool
    message: str = ""


def extract_mentioned_products(message: str) -> list[str]:
    """已废弃：不再按达人品类词抽取。"""
    _ = message
    return []


def extract_intent_topics(message: str) -> list[str]:
    """已废弃：不再抽取达人商务话题。"""
    _ = message
    return []


def should_use_rag_for_message(message: str, trigger_keywords: list[str] | None = None) -> bool:
    """已废弃：咨询类一律由意图路由决定是否走 RAG。"""
    _ = (message, trigger_keywords)
    return bool((message or "").strip())


def detect_upgrade_keywords(message: str) -> list[str]:
    """已废弃：不再检测升标签关键词。"""
    _ = message
    return []


def should_upgrade_to_a(user_tag: UserTag, message: str) -> tuple[bool, list[str]]:
    """已废弃：店铺买家不升级标签。"""
    _ = (user_tag, message)
    return False, []


def try_upgrade_user_tag(
    user_id: str,
    user_tag: UserTag,
    message: str,
    repo=None,
) -> tuple[UserTag, TagUpgradeEvent | None]:
    """已废弃：始终返回原标签，不产生升级事件。"""
    _ = (user_id, message, repo)
    warnings.warn(
        "try_upgrade_user_tag 已废弃，店铺买家不进行标签升级",
        DeprecationWarning,
        stacklevel=2,
    )
    return user_tag, None
