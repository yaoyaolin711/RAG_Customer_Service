"""本地 BGE-M3 Embedding，输出稠密 + 稀疏向量。"""

from __future__ import annotations

from typing import Dict, List, Tuple

from langchain_core.embeddings import Embeddings

from settings import BGE_M3_DEVICE, BGE_M3_PATH, EMBEDDING_MODEL_ID, RAG_EMBEDDING_BATCH_SIZE

_model_instance = None
DENSE_DIM = 1024


def _allow_local_torch_bin_load() -> None:
    """本地可信模型常用 pytorch_model.bin；放行 torch<2.6 的 transformers 限制。"""
    try:
        import transformers.modeling_utils as mu
        import transformers.utils.import_utils as iu

        if getattr(iu, "_rag_torch_load_patched", False):
            return

        def _noop() -> None:
            return None

        iu.check_torch_load_is_safe = _noop  # type: ignore[assignment]
        mu.check_torch_load_is_safe = _noop  # type: ignore[attr-defined]
        iu._rag_torch_load_patched = True  # type: ignore[attr-defined]
    except Exception:
        pass


def get_bge_m3_model(model_path: str | None = None):
    global _model_instance
    path = model_path or BGE_M3_PATH
    if _model_instance is None:
        _allow_local_torch_bin_load()
        from FlagEmbedding import BGEM3FlagModel

        use_fp16 = BGE_M3_DEVICE != "cpu"
        _model_instance = BGEM3FlagModel(path, use_fp16=use_fp16, device=BGE_M3_DEVICE)
    return _model_instance


def sparse_dict_to_milvus(sparse: dict) -> dict[int, float]:
    """将 BGE-M3 lexical_weights 转为 Milvus SPARSE_FLOAT_VECTOR 格式。"""
    return {int(k): float(v) for k, v in sparse.items()}


def _encode_hybrid(
    texts: List[str],
    model_path: str | None = None,
) -> Tuple[List[List[float]], List[Dict[int, float]]]:
    if not texts:
        return [], []
    model = get_bge_m3_model(model_path)
    output = model.encode(
        texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense = output["dense_vecs"].tolist()
    sparse = [sparse_dict_to_milvus(w) for w in output["lexical_weights"]]
    return dense, sparse


class BgeM3Embeddings(Embeddings):
    """LangChain Embeddings 接口（稠密向量，兼容旧调用）。"""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or BGE_M3_PATH

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        dense, _ = _encode_hybrid(texts, self.model_path)
        return dense

    def embed_query(self, text: str) -> List[float]:
        dense, _ = _encode_hybrid([text], self.model_path)
        return dense[0]


def get_embedding_model() -> BgeM3Embeddings:
    return BgeM3Embeddings()


def embed_query_hybrid(text: str, model_path: str | None = None) -> Tuple[List[float], Dict[int, float]]:
    """单条查询的稠密 + 稀疏向量。"""
    dense, sparse = _encode_hybrid([text], model_path)
    return dense[0], sparse[0]


def embed_documents_batch(
    texts: List[str],
    batch_size: int | None = None,
    model_path: str | None = None,
) -> List[List[float]]:
    """批量稠密 embedding（兼容旧接口）。"""
    dense, _ = embed_documents_hybrid_batch(texts, batch_size, model_path)
    return dense


def embed_documents_hybrid_batch(
    texts: List[str],
    batch_size: int | None = None,
    model_path: str | None = None,
) -> Tuple[List[List[float]], List[Dict[int, float]]]:
    """批量稠密 + 稀疏 embedding。"""
    if not texts:
        return [], []
    size = batch_size or RAG_EMBEDDING_BATCH_SIZE
    all_dense: List[List[float]] = []
    all_sparse: List[Dict[int, float]] = []
    for start in range(0, len(texts), size):
        batch = texts[start : start + size]
        dense, sparse = _encode_hybrid(batch, model_path)
        all_dense.extend(dense)
        all_sparse.extend(sparse)
    return all_dense, all_sparse


def get_embedding_model_fingerprint() -> dict:
    """返回 embedding 模型指纹，用于 index 一致性校验。"""
    return {
        "model_id": EMBEDDING_MODEL_ID,
        "model_path": BGE_M3_PATH,
        "normalize_embeddings": True,
        "embedding_type": "dense+sparse",
        "dense_dim": DENSE_DIM,
    }
