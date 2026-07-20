"""
FAQ 种子数据导入脚本 — 将常见问答对写入 MySQL + Redis

用法:
  python scripts/seed_faq.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
for _p in [
    _REPO_ROOT / "crm_agent",
    _REPO_ROOT / "RAG_mode" / "mode",
]:
    if (_p / "settings.py").is_file():
        sys.path.append(str(_p))
        break

from services.faq_store_mysql import MySQLFaqStore
from services.faq_store_redis import RedisFaqStore
from services.qa_variants import build_question_variants, build_search_text

# ============================================================
# FAQ 种子数据 — 按实际店铺业务修改 answer
# ============================================================
FAQ_SEED = [
    # ---- 商品咨询 ----
    ("商品咨询", "售前", "运费险", "有的亲，本店商品都赠送运费险，退货无忧~"),
    ("商品咨询", "售前", "发货时效", "一般下单后24-48小时内发货，具体以仓库实际为准。"),
    ("商品咨询", "售前", "预售", "预售商品按页面标注时间发货，我们会尽快安排，请耐心等待~"),
    ("商品咨询", "售前", "快递", "默认发中通/圆通快递，如需指定快递请联系客服备注。"),
    ("商品咨询", "售前", "运费", "全场满99元包邮，不满则运费按页面显示收取。"),
    ("商品咨询", "售前", "改地址", "下单后请尽快联系客服修改地址，若已发货则无法更改。"),
    ("商品咨询", "售前", "发票", "可以开发票的，下单时备注发票抬头和税号，随货发出。"),
    ("商品咨询", "售前", "赠品", "目前店铺有满赠活动，具体赠品以页面展示为准哦~"),
    ("商品咨询", "售前", "议价", "亲，价格已经很优惠了，暂时没有议价空间，但可以关注店铺优惠券~"),
    ("商品咨询", "售前", "催付", "请尽快完成付款，我们会在付款后第一时间安排发货~"),
    ("商品咨询", "售前", "质保", "本店商品享受一年质保，质保期内非人为损坏免费维修。"),

    # ---- 商品详情 ----
    ("商品咨询", "售前", "尺码", "建议您参考详情页的尺码表，根据自身尺寸选择，也可以联系客服给您推荐~"),
    ("商品咨询", "售前", "材质", "本商品材质详情页有详细说明，保证品质，您可以放心购买。"),
    ("商品咨询", "售前", "功效", "产品功效已在页面详细说明，坚持使用效果更佳哦~"),
    ("商品咨询", "售前", "使用方法", "使用方法请参考商品详情页或随货附带的说明书。"),
    ("商品咨询", "售前", "适用人群", "一般人群均可使用，特殊体质建议咨询医生后使用。"),
    ("商品咨询", "售前", "储存方法", "请置于阴凉干燥处保存，避免阳光直射。"),
    ("商品咨询", "售前", "规格区别", "不同规格主要在数量/大小上有区别，您可以根据自身需求选择。"),
    ("商品咨询", "售前", "洗护说明", "请参考商品标签上的洗护说明，建议按照说明进行清洗保养。"),
    ("商品咨询", "售前", "过敏", "如果您有过敏史，建议先少量试用或咨询医生，确认无碍后再使用。"),

    # ---- 物流 ----
    ("物流", "售中", "催件", "我帮您查一下物流进度，稍等哈~"),
    ("物流", "售中", "物流停滞", "您别急，我帮您联系物流公司核实一下情况，尽快给您回复。"),
    ("物流", "售中", "补发", "如果确实丢件了，我们会尽快为您补发，请放心~"),

    # ---- 售后 ----
    ("售后", "售后", "退货", "亲，支持七天无理由退货，您可以在订单页面申请退货，按指引操作即可。"),
    ("售后", "售后", "仅退款", "商品问题可以申请仅退款，我们会尽快审核处理。"),
    ("售后", "售后", "包裹破损", "收到包裹破损请拍照联系客服，我们会为您处理退换或赔偿。"),
    ("售后", "售后", "不满意退货", "如果不满意，可以在订单页面申请退货退款，我们会及时处理。"),

    # ---- 通用问候 ----
    ("通用", "通用", "问候", "您好呀~欢迎光临，请问有什么可以帮您的？"),
    ("通用", "通用", "在吗", "在的呢，亲有什么问题随时问我哈~"),
]

# ============================================================
# 构建完整记录
# ============================================================

def build_rows() -> list[dict]:
    rows = []
    for main_class, qa_type, sub_class, answer in FAQ_SEED:
        variants = build_question_variants(main_class, qa_type, sub_class)
        search_text = build_search_text(main_class, qa_type, sub_class, variants)
        question_text = f"{main_class} {sub_class}" if main_class != "通用" else sub_class
        rows.append({
            "main_class": main_class,
            "qa_type": qa_type,
            "sub_class": sub_class,
            "question_text": question_text,
            "question_variants": variants,
            "search_text": search_text,
            "answer": answer,
            "source": "seed_faq.py",
        })
    return rows


# ============================================================
# 去重：同 (main_class, sub_class) 多话术 → question_text 追加 #n
# ============================================================

def deduplicate(rows: list[dict]) -> list[dict]:
    seen: dict[str, int] = {}
    for row in rows:
        base = row["question_text"]
        n = seen.get(base, 0) + 1
        seen[base] = n
        if n > 1:
            row["question_text"] = f"{base} #{n}"
            extra = f"{base} 话术{n}"
            variants = list(row["question_variants"])
            if extra not in variants:
                variants.append(extra)
            row["question_variants"] = variants
            row["search_text"] = build_search_text(
                row["main_class"], row["qa_type"], row["sub_class"], variants
            )
    return rows


# ============================================================
# Main
# ============================================================

def main() -> None:
    rows = deduplicate(build_rows())
    print(f"FAQ 种子数据: {len(rows)} 条")

    # 写入 MySQL
    mysql = MySQLFaqStore()
    n = mysql.replace_all(rows)
    print(f"MySQL `faq_qa_pairs`: {n} 行")

    # 加载到 Redis
    db_rows = mysql.list_all()
    redis = RedisFaqStore()
    if not redis.ping():
        print("Redis 连接失败，请检查 Redis 是否启动")
        return
    redis.clear_index()
    loaded = redis.load_all_questions(db_rows, preload_answers=True)
    print(f"Redis faq:qa:* 索引: {loaded} 条 (answers 已预热)")

    # 验证
    from services.qa_cache import lookup_qa_cache
    samples = [
        "有运费险吗",
        "多久发货",
        "怎么退货",
        "运费多少",
        "用什么快递",
        "你好",
        "在吗",
        "能开发票吗",
        "尺码怎么选",
        "什么时候发货",
    ]
    print("\n=== 验证 ===")
    for q in samples:
        hit = lookup_qa_cache(q)
        if hit is None:
            print(f"  [{q}] -> MISS")
        else:
            print(f"  [{q}] -> {hit.match_type} score={hit.score:.3f} ans={hit.answer[:50]}")


if __name__ == "__main__":
    main()
