"""
RAG 妫€绱㈡祴璇曠ず渚?

婕旂ず: Milvus 娣峰悎鍚戦噺妫€绱?鈫?鍙€?LLM 鐢熸垚鍥炵瓟

鐢ㄦ硶:
  python scripts/query_rag.py "閿︿笧 Pro 鏃犵嚎鑰虫満澶氬皯閽憋紵"
  python scripts/query_rag.py "婊″灏戝厤杩愯垂锛? --with-llm
"""
import argparse
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

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rag.index_manifest import check_index_consistency
from settings import (
    LLM_MODEL_BASE_URL,
    LLM_MODEL_NAME,
    RAG_COLLECTION_NAME,
    TOP_K,
    get_aliyun_api_key,
)
from vectorstore import check_milvus_connection, get_rag_vector_store


def retrieve(query: str, k: int = TOP_K):
    vector_store = get_rag_vector_store()
    docs = vector_store.similarity_search(query, k=k)
    return docs


def format_context(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "-")
        page = doc.metadata.get("page", "-")
        parts.append(
            f"[{i}] source={source}, page={page}, chunk_id={chunk_id}\n{doc.page_content}"
        )
    return "\n\n".join(parts)


def answer_with_llm(query: str, context: str) -> str:
    llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        api_key=get_aliyun_api_key(),
        base_url=LLM_MODEL_BASE_URL,
        temperature=0.2,
    )
    messages = [
        SystemMessage(
            content="浣犳槸鐭ヨ瘑搴撻棶绛斿姪鎵嬨€備粎鏍规嵁鎻愪緵鐨勬绱笂涓嬫枃鍥炵瓟锛屼笉瓒冲垯璇存槑鏃犳硶鍥炵瓟銆?
        ),
        HumanMessage(content=f"涓婁笅鏂?\n{context}\n\n闂: {query}"),
    ]
    return llm.invoke(messages).content


def main():
    parser = argparse.ArgumentParser(description="RAG 妫€绱㈡祴璇?)
    parser.add_argument("query", help="妫€绱㈤棶棰?)
    parser.add_argument("--k", type=int, default=TOP_K, help="杩斿洖 top-k 鏉?)
    parser.add_argument("--with-llm", action="store_true", help="妫€绱㈠悗璋冪敤 LLM 鐢熸垚鍥炵瓟")
    args = parser.parse_args()

    check_milvus_connection()
    consistent, msg = check_index_consistency()
    print(msg)
    if not consistent:
        print("璀﹀憡: index 鍙兘杩囨湡锛屾绱㈢粨鏋滀粎渚涘弬鑰冦€傝鍏?rebuild銆?)

    print(f"\n鏌ヨ: {args.query}")
    print(f"Collection: {RAG_COLLECTION_NAME}\n")

    docs = retrieve(args.query, k=args.k)
    if not docs:
        print("鏈绱㈠埌鐩稿叧鏂囨。锛岃鍏堣繍琛? python scripts/ingest_documents.py --rebuild")
        return

    context = format_context(docs)
    print("=== 妫€绱㈢粨鏋?===")
    print(context)

    if args.with_llm:
        print("\n=== LLM 鍥炵瓟 ===")
        try:
            answer = answer_with_llm(args.query, context)
            print(answer)
        except Exception as e:
            print(f"LLM 璋冪敤澶辫触: {e}")
            print("鎻愮ず: 璇疯缃幆澧冨彉閲?DEEPSEEK_KEY")


if __name__ == "__main__":
    main()

