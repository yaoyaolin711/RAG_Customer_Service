"""RAG 知识库检索工具。"""

from __future__ import annotations

from typing import Any

from app.agents.tools.custom.search_knowledge_base import SEARCH_KNOWLEDGE_BASE_TOOL, TOOL_SCHEMA as SKB_SCHEMA
from app.agents.tools.custom.evaluate_confidence import EVALUATE_CONFIDENCE_TOOL, TOOL_SCHEMA as EC_SCHEMA
from app.agents.tools.custom.qa_cache_lookup import QA_CACHE_LOOKUP_TOOL, TOOL_SCHEMA as QA_SCHEMA
from app.agents.tools.custom.product_catalog import PRODUCT_CATALOG_TOOL, TOOL_SCHEMA as PC_SCHEMA
from app.agents.tools.custom.request_human_handoff import REQUEST_HANDOFF_TOOL, TOOL_SCHEMA as RH_SCHEMA
from services.rag_retriever import (
    filter_answer_chunks,
    filter_relevant_chunks,
    format_chunks_for_prompt,
    retrieve_with_strategy,
)
from settings import RAG_RELEVANCE_THRESHOLD

try:
    from reranker import is_rerank_enabled
    from settings import RERANK_RELEVANCE_THRESHOLD
except Exception:
    def is_rerank_enabled() -> bool:  # type: ignore[misc]
        return False

    RERANK_RELEVANCE_THRESHOLD = RAG_RELEVANCE_THRESHOLD


def search_knowledge_base(query: str) -> dict[str, Any]:
    """检索知识库，返回结构化结果（供 Agent 与工具共用）。"""
    retrieval = retrieve_with_strategy(query)
    relevant = filter_answer_chunks(filter_relevant_chunks(retrieval.chunks))
    threshold = RERANK_RELEVANCE_THRESHOLD if is_rerank_enabled() else RAG_RELEVANCE_THRESHOLD
    plan = retrieval.strategy
    return {
        "query": query,
        "hit": bool(relevant),
        "threshold": threshold,
        "count": len(relevant),
        "query_strategy": plan.strategy.value,
        "query_strategy_name": plan.strategy_name,
        "query_strategy_reason": plan.reason,
        "retrieval_queries": list(plan.queries),
        "keywords": list(plan.keywords),
        "sources": [
            {
                "content": c.content,
                "source": c.source,
                "score": c.score,
                "section": c.section,
                "chunk_type": c.chunk_type,
                "question": c.question,
            }
            for c in relevant
        ],
        "context": format_chunks_for_prompt(relevant),
        "chunks": relevant,
    }


RAG_TOOL_SCHEMAS: dict[str, dict] = {
    SEARCH_KNOWLEDGE_BASE_TOOL.name: SKB_SCHEMA,
    EVALUATE_CONFIDENCE_TOOL.name: EC_SCHEMA,
    QA_CACHE_LOOKUP_TOOL.name: QA_SCHEMA,
    PRODUCT_CATALOG_TOOL.name: PC_SCHEMA,
    REQUEST_HANDOFF_TOOL.name: RH_SCHEMA,
}
