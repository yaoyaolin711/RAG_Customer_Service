"""文本切分：基于 BGE-M3 tokenizer 计 token，适配中文文档。"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embedding import get_bge_m3_model
from settings import RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE


def _bge_token_length(text: str) -> int:
    """使用 BGE-M3 同款 tokenizer 统计 token 数，与 embedding 模型对齐。"""
    tokenizer = get_bge_m3_model().tokenizer
    return len(tokenizer.encode(text, add_special_tokens=False))


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        length_function=_bge_token_length,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        is_separator_regex=False,
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """切分文档并注入 chunk_id、page 等 metadata。"""
    splitter = get_text_splitter()
    chunks: list[Document] = []

    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        splits = splitter.split_documents([doc])

        for chunk_index, chunk in enumerate(splits):
            chunk.metadata["source"] = source
            chunk.metadata["chunk_id"] = _stable_chunk_id(source, chunk_index, page)
            chunk.metadata["chunk_index"] = chunk_index
            if page is not None:
                chunk.metadata["page"] = page
            chunks.append(chunk)

    return chunks


def _stable_chunk_id(source: str, chunk_index: int, page: int | None = None) -> str:
    """稳定可复现的 chunk_id，同文档同位置切分结果一致。"""
    import hashlib

    page_num = page if page is not None else -1
    raw = f"{source}|page={page_num}|idx={chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
