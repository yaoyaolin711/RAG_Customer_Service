"""
BD 鎻愮ず璇?docx 缁撴瀯鍖栧叆搴?

鐢ㄦ硶:
  python scripts/ingest_bd_docx.py
  python scripts/ingest_bd_docx.py --file "path/to/BD绛涢€夋彁绀鸿瘝.docx"
"""
import argparse
import os
import shutil
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
for _P in [os.path.join(_REPO_ROOT, "crm_agent", "crm_agent"), os.path.join(_REPO_ROOT, "RAG_mode", "mode")]:
    if os.path.isfile(os.path.join(_P, "settings.py")):
        sys.path.append(_P)
        break

from rag.pipeline import IngestPipeline
from settings import DATA_PATH
from vectorstore import check_milvus_connection

DEFAULT_DOCX = r"d:\xwechat_files\wxid_i3hlr9ja1jug22_804f\msg\file\2026-07\BD绛涢€夋彁绀鸿瘝.docx"


def main():
    parser = argparse.ArgumentParser(description="BD 鎻愮ず璇?docx 缁撴瀯鍖栧叆搴?)
    parser.add_argument("--file", default=DEFAULT_DOCX, help="docx 鏂囦欢璺緞")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        raise FileNotFoundError(f"鎵句笉鍒版枃妗? {args.file}")

    os.makedirs(DATA_PATH, exist_ok=True)
    target = os.path.join(DATA_PATH, "BD绛涢€夋彁绀鸿瘝.docx")
    if os.path.abspath(args.file) != os.path.abspath(target):
        shutil.copy2(args.file, target)
        print(f"宸插鍒舵枃妗ｅ埌: {target}")

    check_milvus_connection()
    pipeline = IngestPipeline()
    result = pipeline.run(
        file_paths=[target],
        structured=True,
        rebuild=True,
        clear_all=False,
    )

    print("\n=== 鍏ュ簱鎽樿 ===")
    print(f"Collection: {result.collection_name}")
    print(f"Chunks: {result.chunk_count}")
    print(f"Embedding: {result.embedding_model}")


if __name__ == "__main__":
    main()

