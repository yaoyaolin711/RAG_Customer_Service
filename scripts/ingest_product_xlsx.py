"""
鐭ヨ瘑搴撶粨鏋勫寲鍏ュ簱锛堜骇鍝佽祫鏂欏簱 + 瀹㈡湇琛屼负瑙勮寖锛?

杞殧绂伙細鍚?collection锛岀敤 chunk_type / kb / reason=kb=... 鍖哄垎銆?
- 浜у搧璧勬枡搴?鈫?kb=product, chunk_type=product_card
- 琛屼负瑙勮寖   鈫?kb=policy,  chunk_type=policy_norm

鐢ㄦ硶:
  python scripts/ingest_product_xlsx.py
  python scripts/ingest_product_xlsx.py --product path.xlsx --policy path.md
  python scripts/ingest_product_xlsx.py --clear-all
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
for _P in [os.path.join(_REPO_ROOT, "crm_agent", "crm_agent"), os.path.join(_REPO_ROOT, "RAG_mode", "mode")]:
    if os.path.isfile(os.path.join(_P, "settings.py")):
        sys.path.append(_P)
        break

from rag.pipeline import IngestPipeline
from rag.structured_policy_md import load_structured_policy_md
from rag.structured_xlsx import load_structured_xlsx
from settings import DATA_PATH, MILVUS_PATH, RAG_COLLECTION_NAME
from vectorstore import check_milvus_connection, get_collection_chunk_types, get_collection_count

DEFAULT_PRODUCT = r"c:\Users\Administrator\Desktop\宸ヤ綔鍙癨寰呭垏鍒嗘枃妗瀹㈡湇鍥㈤槦_浜у搧璧勬枡搴?xlsx"
DEFAULT_POLICY = r"c:\Users\Administrator\Desktop\宸ヤ綔鍙癨寰呭垏鍒嗘枃妗瀹㈡湇宀椾綅鑱岃矗涓庤涓鸿鑼?.md"


def _copy_into_data(src: str) -> str:
    os.makedirs(DATA_PATH, exist_ok=True)
    target = os.path.join(DATA_PATH, Path(src).name)
    if os.path.abspath(src) != os.path.abspath(target):
        shutil.copy2(src, target)
        print(f"宸插鍒跺埌: {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="浜у搧璧勬枡搴?+ 瀹㈡湇琛屼负瑙勮寖鍏ュ簱锛堣蒋闅旂锛?)
    parser.add_argument("--product", default=DEFAULT_PRODUCT, help="浜у搧璧勬枡搴?xlsx")
    parser.add_argument("--policy", default=DEFAULT_POLICY, help="瀹㈡湇宀椾綅鑱岃矗涓庤涓鸿鑼?md")
    parser.add_argument("--faq", default="", help="锛堝凡寮冪敤锛夊嬁鍐嶇亴 FAQ 杩?Milvus锛涜瘽鏈蛋缂撳瓨灞?)
    parser.add_argument(
        "--clear-all",
        action="store_true",
        default=True,
        help="娓呯┖ Milvus 鍏ㄩ儴 collection 鍚庨噸寤猴紙榛樿寮€鍚級",
    )
    parser.add_argument("--no-clear-all", action="store_true", help="涓嶆竻绌哄叏閮紝浠呭垹鏈?collection")
    parser.add_argument("--rebuild", action="store_true", default=True, help="閲嶅缓 rag_collection")
    parser.add_argument("--no-rebuild", action="store_true", help="涓嶅垹 collection锛岀洿鎺?upsert")
    args = parser.parse_args()

    clear_all = False if args.no_clear_all else True
    rebuild = False if args.no_rebuild else True

    paths: list[str] = []
    for label, src in (("浜у搧璧勬枡搴?, args.product), ("琛屼负瑙勮寖", args.policy)):
        if not src:
            continue
        if not os.path.isfile(src):
            raise FileNotFoundError(f"鎵句笉鍒皗label}: {src}")
        paths.append(_copy_into_data(src))

    if args.faq:
        print(f"蹇界暐 --faq锛堣瘽鏈凡璧?Redis/MySQL 缂撳瓨锛屼笉杩?Milvus锛? {args.faq}")

    if not paths:
        raise ValueError("鏈寚瀹氫换浣曞叆搴撴枃浠?)

    print(f"Milvus: {MILVUS_PATH}")
    print(f"Collection: {RAG_COLLECTION_NAME}")
    print("=== 鍒囧垎棰勮 ===")
    type_counts: dict[str, int] = {}
    for path in paths:
        suffix = Path(path).suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            chunks = load_structured_xlsx(path)
        else:
            chunks = load_structured_policy_md(path)
        for c in chunks:
            ct = str(c.metadata.get("chunk_type", ""))
            type_counts[ct] = type_counts.get(ct, 0) + 1
            kb = str(c.metadata.get("kb", ""))
            reason = str(c.metadata.get("reason", ""))
            assert kb, f"缂哄皯 kb 杞殧绂? {path}"
            assert reason.startswith("kb="), f"缂哄皯 reason 杞殧绂? {path}"
        n, ct, kb = len(chunks), chunks[0].metadata.get("chunk_type"), chunks[0].metadata.get("kb")
        print(f"{Path(path).name}: {n} chunks, type={ct}, kb={kb}")
        print("--- sample ---")
        sample_text = chunks[0].page_content[:240].encode("unicode_escape").decode("ascii")
        print(sample_text)
        print("--------------")
    print("type_counts:", type_counts)

    check_milvus_connection()
    pipeline = IngestPipeline()
    result = pipeline.run(
        file_paths=paths,
        structured=True,
        rebuild=rebuild,
        clear_all=clear_all,
    )

    count = get_collection_count(result.collection_name)
    types = get_collection_chunk_types(result.collection_name, sample_limit=2000)
    print("\n=== 鍏ュ簱鎽樿 ===")
    print(f"Collection: {result.collection_name}")
    print(f"Chunks written: {result.chunk_count}")
    print(f"Milvus row_count: {count}")
    print(f"chunk_type sample: {types}")
    print(f"Embedding: {result.embedding_model}")


if __name__ == "__main__":
    main()

