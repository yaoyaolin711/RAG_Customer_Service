"""用户问题文本归一化：提升 exact 命中率。"""

from __future__ import annotations

import re
import unicodedata

# 常见客服语气词 / 填充词（仅用于 exact key，不用于改写语义）
_MODAL_RE = re.compile(
    r"(您|你|我|呢|啊|呀|吧|哦|噢|哈|啦|么|吗|嘛|亲+|亲亲|宝宝|宝|"
    r"请问|问一下|帮我|麻烦|一下|怎么样|如何)"
)
_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize_question(text: str) -> str:
    """去空白/标点/部分语气词，便于 Redis exact 匹配。"""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).strip().lower()
    if not s:
        return ""
    s = _MODAL_RE.sub("", s)
    s = _PUNCT_RE.sub("", s)
    return s
