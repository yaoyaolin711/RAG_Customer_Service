# scripts/build_vectorstore.py
"""
姝よ剼鏈礋璐ｏ細
1. 鍔犺浇 data/ 鐩綍涓嬬殑閿︿笧鍟嗗煄鐭ヨ瘑搴撴枃妗ｏ紙TXT锛?
2. 浣跨敤鏈湴 BGE-M3 鍚戦噺鍖栵紙绋犲瘑 + 绋€鐤忥級
3. 鍐欏叆 Milvus 鍚戦噺鏁版嵁搴?
"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
for _P in [os.path.join(_REPO_ROOT, "crm_agent", "crm_agent"), os.path.join(_REPO_ROOT, "RAG_mode", "mode")]:
    if os.path.isfile(os.path.join(_P, "settings.py")):
        sys.path.append(_P)
        break

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PATH,
    MILVUS_COLLECTION_NAME,
    MILVUS_URI,
)
from embedding import embed_documents_hybrid_batch
from vectorstore import check_milvus_connection, delete_collection, ensure_hybrid_collection, upsert_chunks

KNOWLEDGE_FILES = [
    "product_guide.txt",
    "after_sales_policy.txt",
    "promotion_member.txt",
]


def load_txt(filename: str) -> dict:
    file_path = os.path.join(DATA_PATH, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"page_content": content, "metadata": {"source": filename}}


def main():
    check_milvus_connection()

    try:
        delete_collection(MILVUS_COLLECTION_NAME)
        print(f"宸插垹闄ゆ棫闆嗗悎: {MILVUS_COLLECTION_NAME}")
    except Exception:
        pass

    ensure_hybrid_collection(MILVUS_COLLECTION_NAME)

    raw_documents = [load_txt(name) for name in KNOWLEDGE_FILES]
    print(f"鍏卞姞杞戒簡 {len(raw_documents)} 涓枃妗ｃ€?)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "銆?, "锛?, "锛?, " ", ""],
    )

    docs: list[Document] = []
    for item in raw_documents:
        for split in text_splitter.split_text(item["page_content"]):
            docs.append(Document(page_content=split, metadata=item["metadata"]))
    print(f"鍏卞垎鍧椾簡 {len(docs)} 涓枃妗ｃ€?)

    ids = [f"doc_{i}" for i in range(len(docs))]
    texts = [d.page_content for d in docs]
    metadatas = [
        {
            "source": d.metadata.get("source", ""),
            "chunk_id": ids[i],
            "chunk_index": i,
            "page": 0,
            "section": "",
            "chunk_type": "",
        }
        for i, d in enumerate(docs)
    ]

    print("寮€濮?BGE-M3 绋犲瘑+绋€鐤忓悜閲忓寲...")
    dense_vectors, sparse_vectors = embed_documents_hybrid_batch(texts)
    upsert_chunks(MILVUS_COLLECTION_NAME, ids, texts, dense_vectors, sparse_vectors, metadatas)

    print(f"鍚戦噺搴撴瀯寤哄畬鎴? uri={MILVUS_URI}, collection={MILVUS_COLLECTION_NAME}")


if __name__ == "__main__":
    main()

