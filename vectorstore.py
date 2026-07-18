"""Milvus 向量库连接与混合检索封装（稠密 + 稀疏）。"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

from embedding import DENSE_DIM, embed_query_hybrid
from settings import (
    MILVUS_COLLECTION_NAME,
    MILVUS_TOKEN,
    MILVUS_URI,
    RAG_COLLECTION_NAME,
)

_client: MilvusClient | None = None

OUTPUT_FIELDS = [
    "text",
    "source",
    "chunk_id",
    "chunk_index",
    "page",
    "section",
    "chunk_type",
    "question",
    "keyword",
    "reason",
    "dense_vector",
]


def close_milvus_client() -> None:
    """关闭全局客户端，便于 Windows 下物理清理 Lite 文件。"""
    global _client
    if _client is None:
        return
    try:
        _client.close()
    except Exception:
        pass
    _client = None


def _patch_windows_rename() -> None:
    """Milvus Lite 在 Windows 上用 os.rename(tmp, manifest) 会触发 WinError 183。"""
    import os

    if getattr(os, "_rag_rename_patched", False):
        return
    _old_rename = os.rename

    def _rename(src, dst, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            return _old_rename(src, dst, *args, **kwargs)
        except OSError as exc:
            winerr = getattr(exc, "winerror", None)
            if winerr == 183 or getattr(exc, "errno", None) in {17, 183}:
                return os.replace(src, dst)
            raise

    os.rename = _rename  # type: ignore[assignment]
    os._rag_rename_patched = True  # type: ignore[attr-defined]


def get_milvus_client() -> MilvusClient:
    global _client
    _patch_windows_rename()
    if _client is None:
        kwargs: dict[str, Any] = {"uri": MILVUS_URI}
        if MILVUS_TOKEN:
            kwargs["token"] = MILVUS_TOKEN
        _client = MilvusClient(**kwargs)
    return _client


def check_milvus_connection() -> None:
    """检查 Milvus 是否可用。"""
    client = get_milvus_client()
    client.list_collections()


def _ensure_loaded(client: MilvusClient, collection_name: str) -> None:
    try:
        state = client.get_load_state(collection_name)
        if state.get("state") != "Loaded":
            client.load_collection(collection_name)
    except Exception:
        client.load_collection(collection_name)


def ensure_hybrid_collection(collection_name: str) -> None:
    """创建支持稠密 + 稀疏向量的 Milvus collection。"""
    client = get_milvus_client()
    if client.has_collection(collection_name):
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=256)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=DENSE_DIM)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("source", DataType.VARCHAR, max_length=512)
    schema.add_field("section", DataType.VARCHAR, max_length=512)
    schema.add_field("chunk_type", DataType.VARCHAR, max_length=128)
    schema.add_field("question", DataType.VARCHAR, max_length=1000)
    schema.add_field("keyword", DataType.VARCHAR, max_length=512)
    schema.add_field("reason", DataType.VARCHAR, max_length=1000)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("page", DataType.INT64)

    client.create_collection(collection_name, schema=schema)

    for field, index_type, metric in [
        ("dense_vector", "AUTOINDEX", "COSINE"),
        ("sparse_vector", "SPARSE_INVERTED_INDEX", "IP"),
    ]:
        try:
            _prepare_windows_manifest(collection_name)
            idx = client.prepare_index_params()
            idx.add_index(field, index_type=index_type, metric_type=metric)
            client.create_index(collection_name, idx)
        except Exception as exc:
            # Windows Milvus Lite 常见 WinError 183；删掉旧 manifest 再试一次
            msg = str(exc)
            if "WinError 183" in msg or "当文件已存在时" in msg:
                _prepare_windows_manifest(collection_name, force=True)
                try:
                    idx = client.prepare_index_params()
                    idx.add_index(field, index_type=index_type, metric_type=metric)
                    client.create_index(collection_name, idx)
                    continue
                except Exception as exc2:
                    print(f"  索引 {field} 创建警告（可忽略）: {exc2}")
            else:
                print(f"  索引 {field} 创建警告（可忽略）: {exc}")


def _prepare_windows_manifest(collection_name: str, *, force: bool = False) -> None:
    """缓解 Windows 下 milvus Lite manifest.json.tmp → manifest.json 的 rename 冲突。"""
    from pathlib import Path

    base = Path(MILVUS_URI) / "collections" / collection_name
    if not base.exists():
        return
    man = base / "manifest.json"
    tmp = base / "manifest.json.tmp"
    try:
        if force and man.exists():
            man.unlink()
        if tmp.exists() and man.exists():
            # 保留已有 manifest，去掉冲突的 tmp，避免 rename 目标已存在
            tmp.unlink()
        elif tmp.exists() and not man.exists():
            tmp.rename(man)
    except OSError:
        pass


def delete_collection(collection_name: str) -> None:
    client = get_milvus_client()
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)


def list_collections() -> list[str]:
    return get_milvus_client().list_collections()


def upsert_chunks(
    collection_name: str,
    ids: list[str],
    texts: list[str],
    dense_vectors: list[list[float]],
    sparse_vectors: list[dict[int, float]],
    metadatas: list[dict],
    *,
    write_batch: int = 64,
) -> None:
    """批量 upsert 文档（稠密 + 稀疏向量）。"""
    client = get_milvus_client()
    ensure_hybrid_collection(collection_name)

    for start in range(0, len(ids), write_batch):
        end = start + write_batch
        rows = []
        for i in range(start, min(end, len(ids))):
            meta = metadatas[i]
            row = {
                "chunk_id": ids[i],
                "text": texts[i],
                "dense_vector": dense_vectors[i],
                "sparse_vector": sparse_vectors[i],
                "source": str(meta.get("source", "")),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "page": int(meta.get("page", 0)),
                "section": str(meta.get("section", "")),
                "chunk_type": str(meta.get("chunk_type", "")),
                "question": str(meta.get("question", "")),
                "keyword": str(meta.get("keyword", "")),
                "reason": str(meta.get("reason", "")),
            }
            rows.append(row)
        client.upsert(collection_name, rows)
        try:
            client.flush(collection_name)
        except Exception:
            pass
        print(f"  已写入 {min(end, len(ids))}/{len(ids)}")


def _dense_relevance_score(query_dense: list[float], doc_dense: list[float]) -> float:
    return float(sum(a * b for a, b in zip(query_dense, doc_dense)))


def _hit_to_document(hit: dict) -> Document:
    entity = hit.get("entity", hit)
    metadata = {
        "source": entity.get("source", "unknown"),
        "chunk_id": entity.get("chunk_id", ""),
        "chunk_index": entity.get("chunk_index", 0),
        "page": entity.get("page", 0),
        "section": entity.get("section", ""),
        "chunk_type": entity.get("chunk_type", ""),
        "question": entity.get("question", ""),
        "keyword": entity.get("keyword", ""),
        "reason": entity.get("reason", ""),
    }
    return Document(page_content=entity.get("text", ""), metadata=metadata)


def hybrid_search_with_scores(
    query: str,
    collection_name: str,
    k: int = 3,
) -> list[tuple[Document, float]]:
    """混合检索（稠密 + 稀疏 RRF 融合），返回 (Document, 稠密余弦相似度)。"""
    client = get_milvus_client()
    _ensure_loaded(client, collection_name)

    dense_q, sparse_q = embed_query_hybrid(query)

    try:
        req_dense = AnnSearchRequest(
            data=[dense_q],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=k,
        )
        req_sparse = AnnSearchRequest(
            data=[sparse_q],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=k,
        )
        results = client.hybrid_search(
            collection_name,
            [req_dense, req_sparse],
            ranker=RRFRanker(),
            limit=k,
            output_fields=OUTPUT_FIELDS,
        )
        hits = results[0] if results else []
    except Exception:
        results = client.search(
            collection_name,
            data=[dense_q],
            anns_field="dense_vector",
            limit=k,
            output_fields=OUTPUT_FIELDS,
        )
        hits = results[0] if results else []

    output: list[tuple[Document, float]] = []
    for hit in hits:
        doc = _hit_to_document(hit)
        doc_dense = hit.get("entity", hit).get("dense_vector")
        if doc_dense:
            score = _dense_relevance_score(dense_q, doc_dense)
        else:
            score = float(hit.get("distance", 0.0))
        output.append((doc, score))
    return output


def get_collection_count(collection_name: str) -> int:
    client = get_milvus_client()
    if not client.has_collection(collection_name):
        return 0
    stats = client.get_collection_stats(collection_name)
    return int(stats.get("row_count", 0))


def get_collection_chunk_types(collection_name: str, sample_limit: int = 100) -> dict[str, int]:
    client = get_milvus_client()
    if not client.has_collection(collection_name):
        return {}
    _ensure_loaded(client, collection_name)
    rows = client.query(
        collection_name,
        filter="chunk_id != ''",
        output_fields=["chunk_type"],
        limit=sample_limit,
    )
    types: dict[str, int] = {}
    for row in rows:
        t = row.get("chunk_type") or "unknown"
        types[t] = types.get(t, 0) + 1
    return types


class MilvusRAGStore:
    """兼容 LangChain 向量库接口的 Milvus 封装。"""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        return [doc for doc, _ in hybrid_search_with_scores(query, self.collection_name, k=k)]

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4
    ) -> list[tuple[Document, float]]:
        return hybrid_search_with_scores(query, self.collection_name, k=k)


def get_vector_store(collection_name: str | None = None) -> MilvusRAGStore:
    return MilvusRAGStore(collection_name or MILVUS_COLLECTION_NAME)


def get_rag_vector_store() -> MilvusRAGStore:
    return get_vector_store(RAG_COLLECTION_NAME)
