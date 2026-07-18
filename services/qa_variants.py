"""从主类/类型/副类生成可匹配问法（variants），供 exact / BM25 使用。"""

from __future__ import annotations

import re

_PAREN_RE = re.compile(r"[（(].*?[）)]")
_TRAIL_RE = re.compile(r"[？?；;。.!！]+$")

# (副类关键词片段, 买家常见问法) —— 匹配时取最长关键词，避免「发货」误贴到「发货延期」
_THEME_QUESTIONS: list[tuple[tuple[str, ...], list[str]]] = [
    (("问候",), ["你好", "在吗", "有人吗", "客服在吗", "在不在"]),
    (("发货时效",), ["多久发货", "什么时候发货", "几天发货", "今天能发货吗", "发货要多久"]),
    (("发货延期", "停发"), ["为什么还没发货", "发货延期了吗", "什么时候能恢复发货"]),
    (("预售",), ["预售什么时候发", "预售多久发货", "是预售吗"]),
    (("快递",), ["用什么快递", "发什么快递", "什么物流", "能指定快递吗", "是哪家快递"]),
    (("运费险",), ["有运费险吗", "包运费险吗", "退货运费谁出", "运费险怎么用"]),
    (("运费",), ["运费多少", "包邮吗", "邮费谁出", "运费怎么算"]),
    (("催件", "催促物流", "物流停滞", "物流问题"), ["物流到哪了", "怎么还没到", "快递到哪了", "帮我催一下物流"]),
    (("改地址",), ["能改地址吗", "改收货地址", "地址填错了"]),
    (("发票",), ["能开发票吗", "怎么开发票", "发票怎么开"]),
    (("不满意退货", "要求退货", "退货引导", "退货"), ["怎么退货", "可以退货吗", "退货流程", "不想要了怎么退"]),
    (("仅退款", "全额退款"), ["能仅退款吗", "不退货可以退款吗", "全额退款"]),
    (("补发",), ["能补发吗", "什么时候补发", "帮我补发"]),
    (("包裹破损", "收到坏", "破损"), ["收到坏了", "包裹破损", "东西坏了怎么办"]),
    (("尺码", "码数", "尺寸"), ["怎么选尺码", "尺码怎么选", "码数建议", "尺寸多少"]),
    (("填充材质", "材质", "面料"), ["什么材质", "是什么面料", "材质安全吗"]),
    (("使用方法", "食用方法", "每日食用"), ["怎么用", "怎么使用", "用法", "怎么吃"]),
    (("适用人群", "适合人群", "不适宜", "禁用"), ["适合什么人", "哪些人不能用", "孕妇能用吗"]),
    (("功效", "作用", "效果", "改善"), ["有什么功效", "有什么用", "效果怎么样", "能改善什么"]),
    (("储存方法", "储存", "养护", "收纳", "保养"), ["怎么保存", "怎么养护", "怎么收纳", "怎么保养"]),
    (("洗护说明", "清洁方式", "水洗", "机洗", "清洗"), ["能洗吗", "怎么清洗", "可以机洗吗", "怎么清洁"]),
    (("赠品",), ["有赠品吗", "送什么赠品", "带不带赠品"]),
    (("议价", "差价", "半价", "更便宜"), ["能便宜点吗", "有优惠吗", "能便宜吗"]),
    (("催付",), ["还没付款", "怎么付款", "订单待付款"]),
    (("一年质保", "质保"), ["有质保吗", "质保多久", "保修吗"]),
    (("过敏", "副作用"), ["会过敏吗", "有副作用吗", "过敏怎么办"]),
    (("价位区别", "规格区别", "规格", "区别"), ["有什么区别", "规格有什么不一样", "怎么选"]),
    (("重量推荐", "重量选择", "重量"), ["多重", "重量多少", "怎么选重量"]),
]


def _best_theme_questions(sub_clean: str) -> list[str]:
    """按副类匹配模板问法：精确命中优先；细分主题（运费险超重）不抢父主题问法。"""
    if not sub_clean:
        return []
    for keys, questions in _THEME_QUESTIONS:
        if sub_clean in keys:
            return list(questions)

    best_len = 0
    best_qs: list[str] = []
    for keys, questions in _THEME_QUESTIONS:
        for k in keys:
            if k and k in sub_clean and len(k) > best_len:
                best_len = len(k)
                best_qs = list(questions)
    # 副类明显长于关键词 → 视为细分主题，不用父主题全局问法
    if best_qs and len(sub_clean) - best_len >= 2:
        return []
    return best_qs


def clean_sub_class(sub: str) -> str:
    text = (sub or "").strip()
    if not text:
        return ""
    text = _PAREN_RE.sub("", text).strip()
    text = _TRAIL_RE.sub("", text).strip()
    return text


def build_question_variants(main_class: str, qa_type: str, sub_class: str) -> list[str]:
    """生成去重后的可匹配问法列表（不含完整话术正文）。"""
    main = (main_class or "").strip()
    typ = (qa_type or "").strip()
    sub = (sub_class or "").strip()
    sub_clean = clean_sub_class(sub)

    out: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        text = (value or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        out.append(text)

    add(sub)
    add(sub_clean)
    if typ:
        add(f"{typ}{sub_clean}" if sub_clean else typ)
    if main and main != "通用":
        add(f"{main}{sub_clean}")
        add(f"{main} {sub_clean}")
        add(f"{main}的{sub_clean}")

    hay = f"{sub} {sub_clean}"
    theme_qs = _best_theme_questions(sub_clean)
    for q in theme_qs:
        # 全局常见问法只挂在「通用」话术上，避免末条商品 exact 抢键
        if main == "通用" or not main:
            add(q)
        elif sub_clean in {
            "使用方法",
            "食用方法",
            "材质",
            "功效",
            "作用",
            "适用人群",
            "适合人群",
            "规格",
            "尺码",
            "赠品",
        }:
            add(f"{main}{q}")
            add(f"{main} {q}")

    # 副类本身已是问句时保留；否则补弱模板（仅主题本身，不生成全局抢占问法）
    if sub_clean and not any(ch in sub_clean for ch in "吗么哪好多怎能否"):
        add(f"{sub_clean}吗")
        add(f"请问{sub_clean}")

    return out


def build_search_text(
    main_class: str,
    qa_type: str,
    sub_class: str,
    variants: list[str],
    *,
    max_chars: int = 900,
) -> str:
    """BM25 语料：主题 + 问法，不拼接话术正文。"""
    parts = [main_class, qa_type, sub_class, *variants]
    text = " ".join(p.strip() for p in parts if p and str(p).strip())
    return text[:max_chars].strip()
