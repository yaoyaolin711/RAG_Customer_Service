"""验证本地 BGE-Reranker-v2-m3 可在 GPU 加载，并打印显存占用。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    import torch

    from reranker import get_reranker, is_rerank_enabled, rerank_pairs
    from settings import (
        RERANKER_BATCH_SIZE,
        RERANKER_DEVICE,
        RERANKER_PATH,
        RERANKER_USE_FP16,
    )

    print(f"rerank_enabled={is_rerank_enabled()}")
    print(f"path={RERANKER_PATH}")
    print(f"device={RERANKER_DEVICE} fp16={RERANKER_USE_FP16} batch={RERANKER_BATCH_SIZE}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated() / (1024**2)
    else:
        before = 0.0

    get_reranker()
    if torch.cuda.is_available():
        after_load = torch.cuda.memory_allocated() / (1024**2)
        peak_load = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        after_load = peak_load = 0.0

    query = "多久发货有没有运费险"
    passages = [
        "一般下单后 48 小时内发货，具体以物流更新为准。",
        "今日天气晴朗，适合户外运动。",
        "收件地址与联系人请在下单页核实。",
        "退货政策：七天无理由退货，不影响二次销售。",
        "运费险规则以商品详情页说明为准。",
        "护膝产品材质为氯丁橡胶，尺码参考尺寸表。",
        "物流一般三到五天送达，偏远地区可能更久。",
        "包邮活动限指定地区，详情见活动页。",
    ]
    scores = rerank_pairs(query, passages)
    ranked = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)

    if torch.cuda.is_available():
        after_infer = torch.cuda.memory_allocated() / (1024**2)
        peak_infer = torch.cuda.max_memory_allocated() / (1024**2)
        print(
            f"vram_mb before={before:.1f} after_load={after_load:.1f} "
            f"after_infer={after_infer:.1f} peak={peak_infer:.1f}"
        )
        if peak_infer > 4200:
            print("WARNING: peak VRAM > 4.2GB，可再调小 RERANKER_BATCH_SIZE / MAX_LENGTH")
        else:
            print("OK: peak VRAM within ~4GB budget")
    else:
        print("running on CPU (no CUDA)")

    print("top scores:")
    for score, text in ranked[:3]:
        print(f"  {score:.4f} | {text}")


if __name__ == "__main__":
    main()
