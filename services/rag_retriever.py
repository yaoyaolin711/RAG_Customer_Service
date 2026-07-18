"""RAG 知识库检索封装（策略选择 + 混合召回 + Reranker 重排 + 去重）。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document

from settings import (
    RAG_CANDIDATE_K,
    RAG_RELEVANCE_THRESHOLD,
    RERANK_RELEVANCE_THRESHOLD,
    TOP_K,
)
from services.models import RetrievedChunk
from services.query_strategy import QueryStrategy, QueryStrategyPlan, select_query_strategy
from vectorstore import get_rag_vector_store

try:
    from reranker import is_rerank_enabled, rerank_pairs
except Exception:  # pragma: no cover
    def is_rerank_enabled() -> bool:  # type: ignore[misc]
        return False

    def rerank_pairs(query: str, passages, *, model_path=None):  # type: ignore[misc]
        return [0.0] * len(list(passages))


INTERNAL_CHUNK_TYPES = frozenset({"upgrade_keyword", "section_header"})
# 软隔离：同一 collection 内按知识域区分
KB_CHUNK_TYPES: dict[str, frozenset[str]] = {
    "product": frozenset({"product_card"}),
    "script": frozenset({"script_faq"}),
    "policy": frozenset({"policy_norm"}),
}
_SYSTEM_HINT_RE = re.compile(r"[（(]\s*系统提示[：:][^）)]*[）)]")


@dataclass
class RetrieveResult:
    chunks: list[RetrievedChunk]
    strategy: QueryStrategyPlan


def _doc_to_chunk(doc: Document, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        content=doc.page_content,
        source=str(doc.metadata.get("source", "unknown")),
        chunk_id=str(doc.metadata.get("chunk_id", "")),
        page=int(doc.metadata.get("page", 0)),
        score=float(score),
        section=str(doc.metadata.get("section", "")),
        chunk_type=str(doc.metadata.get("chunk_type", "")),
        question=str(doc.metadata.get("question", "")),
    )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().lower())


def deduplicate_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    output: list[RetrievedChunk] = []
    for chunk in chunks:
        cid = (chunk.chunk_id or "").strip()
        if cid and cid in seen_ids:
            continue
        norm = _normalize_text(chunk.content)
        if norm and norm in seen_text:
            continue
        if cid:
            seen_ids.add(cid)
        if norm:
            seen_text.add(norm)
        output.append(chunk)
    return output


def _apply_rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    if not chunks:
        return []
    passages = [c.content for c in chunks]
    scores = rerank_pairs(query, passages)
    ranked = sorted(
        (
            RetrievedChunk(
                content=c.content,
                source=c.source,
                chunk_id=c.chunk_id,
                page=c.page,
                score=float(s),
                section=c.section,
                chunk_type=c.chunk_type,
                question=c.question,
            )
            for c, s in zip(chunks, scores)
        ),
        key=lambda x: x.score,
        reverse=True,
    )
    return deduplicate_chunks(ranked)[:top_k]


def _vector_retrieve(query: str, candidate_k: int) -> list[RetrievedChunk]:
    store = get_rag_vector_store()
    results = store.similarity_search_with_relevance_scores(query, k=candidate_k)
    return [_doc_to_chunk(doc, score) for doc, score in results]


def _merge_by_best_score(chunk_lists: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    best: dict[str, RetrievedChunk] = {}
    order: list[str] = []
    for chunks in chunk_lists:
        for c in chunks:
            key = (c.chunk_id or "").strip() or f"text:{_normalize_text(c.content)[:80]}"
            prev = best.get(key)
            if prev is None:
                best[key] = c
                order.append(key)
            elif c.score > prev.score:
                best[key] = c
    merged = [best[k] for k in order]
    return sorted(merged, key=lambda x: x.score, reverse=True)


def retrieve_with_strategy(
    query: str,
    k: int = TOP_K,
    *,
    plan: QueryStrategyPlan | None = None,
) -> RetrieveResult:
    """按策略检索；重排始终用用户原问对齐相关性。"""
    text = (query or "").strip()
    strategy_plan = plan or select_query_strategy(text)
    candidate_k = max(k, RAG_CANDIDATE_K) if is_rerank_enabled() else k

    if strategy_plan.strategy == QueryStrategy.MULTI_QUERY:
        per_query = [_vector_retrieve(q, candidate_k) for q in strategy_plan.queries]
        merged = _merge_by_best_score(per_query)
        merged = merged[: max(candidate_k * 2, k)]
        chunks = _apply_rerank(text, merged, top_k=k) if is_rerank_enabled() else deduplicate_chunks(merged)[:k]
        return RetrieveResult(chunks=chunks, strategy=strategy_plan)

    search_query = strategy_plan.queries[0] if strategy_plan.queries else text
    if strategy_plan.strategy == QueryStrategy.KEYWORD_BOOST and strategy_plan.keywords:
        kw_query = " ".join(strategy_plan.keywords)
        merged = _merge_by_best_score(
            [
                _vector_retrieve(search_query, candidate_k),
                _vector_retrieve(kw_query, candidate_k),
            ]
        )
        chunks = (
            _apply_rerank(text, merged, top_k=k)
            if is_rerank_enabled()
            else deduplicate_chunks(merged)[:k]
        )
        return RetrieveResult(chunks=chunks, strategy=strategy_plan)

    raw = _vector_retrieve(search_query, candidate_k)
    chunks = _apply_rerank(text, raw, top_k=k) if is_rerank_enabled() else raw[:k]
    return RetrieveResult(chunks=chunks, strategy=strategy_plan)


def retrieve_with_scores(query: str, k: int = TOP_K) -> list[RetrievedChunk]:
    return retrieve_with_strategy(query, k=k).chunks


def filter_relevant_chunks(
    chunks: list[RetrievedChunk],
    threshold: float | None = None,
) -> list[RetrievedChunk]:
    if threshold is not None:
        min_score = threshold
    elif is_rerank_enabled():
        min_score = RERANK_RELEVANCE_THRESHOLD
    else:
        min_score = RAG_RELEVANCE_THRESHOLD
    return [c for c in chunks if c.score >= min_score]


def filter_answer_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return [c for c in chunks if c.chunk_type not in INTERNAL_CHUNK_TYPES]


def filter_chunks_by_kb(
    chunks: list[RetrievedChunk],
    kb: str | None = None,
) -> list[RetrievedChunk]:
    """软隔离过滤：kb=product / script / policy；为空则不过滤。"""
    if not kb:
        return chunks
    allowed = KB_CHUNK_TYPES.get(kb.strip().lower())
    if not allowed:
        return chunks
    return [c for c in chunks if c.chunk_type in allowed]


def sanitize_user_reply(text: str) -> str:
    """去掉误生成的括号说明、Markdown、句末波浪号等，保证用户只看到真人话术。"""
    cleaned = _SYSTEM_HINT_RE.sub("", text)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[-*]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\d+[.、)]\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"[～~]+(?=\s*$)", "", cleaned)
    cleaned = re.sub(r"[～~]+(?=[。！？，,.])", "", cleaned)
    cleaned = re.sub(r"([。！？，,.])\s*[～~]+", r"\1", cleaned)
    return cleaned.strip()


def format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return ""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] 来源={c.source}, 相关度={c.score:.2f}\n{c.content}"
        )
    return "\n\n".join(parts)


def doc_to_retrieved(doc: Document, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        content=doc.page_content,
        source=str(doc.metadata.get("source", "unknown")),
        chunk_id=str(doc.metadata.get("chunk_id", "")),
        page=int(doc.metadata.get("page", 0)),
        score=score,
    )
