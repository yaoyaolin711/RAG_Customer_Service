"""语义匹配：基于 BGE-M3 稠密向量余弦相似度。

用途：
1. 缓存 FAQ：用户问题 ↔ 标准问法（问-问）
2. 输出质检：用户问题 ↔ 最终答案（问-答）
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from embedding import embed_documents_batch, embed_query_hybrid

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(va, vb) / denom)


def embed_text(text: str) -> list[float]:
    dense, _ = embed_query_hybrid((text or "").strip() or " ")
    return dense


def embed_texts(texts: list[str]) -> list[list[float]]:
    cleaned = [(t or "").strip() or " " for t in texts]
    if not cleaned:
        return []
    return embed_documents_batch(cleaned)


def score_query_to_texts(query: str, texts: list[str]) -> list[float]:
    """一次编码 query + 多条候选，返回逐条余弦分。"""
    if not texts:
        return []
    q_vec = embed_text(query)
    cand_vecs = embed_texts(texts)
    return [cosine_similarity(q_vec, v) for v in cand_vecs]


def score_question_pair(query: str, candidate_question: str) -> float:
    scores = score_query_to_texts(query, [candidate_question])
    return scores[0] if scores else 0.0


def score_question_answer(question: str, answer: str) -> float:
    """问-答相关性（跨类型文本，分数通常低于问-问）。"""
    q = (question or "").strip()
    a = (answer or "").strip()
    if not q or not a:
        return 0.0
    return score_question_pair(q, a)


@lru_cache(maxsize=1)
def _warmup_note() -> str:
    return "semantic_match_ready"
