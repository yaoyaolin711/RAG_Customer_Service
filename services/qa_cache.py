"""咨询类多级缓存：Redis exact → BGE 全库语义检索 → MySQL → 回写 Redis。

已移除 BM25。FAQ 问法变体预先编码为稠密向量，查询时做余弦相似度召回。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from settings import (
    ANSWER_RELEVANCE_THRESHOLD,
    QA_ANSWER_TTL_SECONDS,
    QA_BM25_CANDIDATE_TOP_K,
    QA_CACHE_ENABLED,
    QA_SEMANTIC_THRESHOLD,
)
from services.faq_store_mysql import get_mysql_faq_store
from services.faq_store_redis import get_redis_faq_store
from services.models import IntentResult, ReplyMode, RetrievedChunk, RouteType, WeChatMessageResponse
from services.qa_normalize import normalize_question
from services.qa_slot import detect_user_slots, force_rag_detail, slot_allows_item
from services.semantic_match import embed_text, embed_texts, score_question_answer

logger = logging.getLogger(__name__)


@dataclass
class QaCacheHit:
    faq_id: int
    question_text: str
    answer: str
    score: float
    match_type: str  # exact | semantic
    main_class: str = ""
    qa_type: str = ""
    sub_class: str = ""
    source: str = ""
    qa_relevance: float = 0.0
    matched_phrase: str = ""


class FaqSemanticIndex:
    """进程内 FAQ 语义索引：每条问法变体一条向量。"""

    def __init__(self):
        self._items_by_id: dict[int, dict[str, Any]] = {}
        self._phrases: list[str] = []
        self._faq_ids: list[int] = []
        self._matrix: np.ndarray | None = None  # (N, D) L2-normalized

    @property
    def ready(self) -> bool:
        return self._matrix is not None and len(self._phrases) > 0

    def rebuild(self, items: list[dict[str, Any]]) -> int:
        self._items_by_id = {}
        phrases: list[str] = []
        faq_ids: list[int] = []
        seen_pair: set[tuple[int, str]] = set()

        for item in items:
            try:
                faq_id = int(item["id"])
            except (KeyError, TypeError, ValueError):
                continue
            self._items_by_id[faq_id] = item
            for phrase in _candidate_phrases(item):
                key = (faq_id, phrase)
                if key in seen_pair:
                    continue
                seen_pair.add(key)
                phrases.append(phrase)
                faq_ids.append(faq_id)

        self._phrases = phrases
        self._faq_ids = faq_ids
        if not phrases:
            self._matrix = None
            return 0

        logger.info("FAQ 语义索引编码中：%s 条问法 …", len(phrases))
        vectors = embed_texts(phrases)
        mat = np.asarray(vectors, dtype=np.float32)
        # L2 normalize for cosine = dot
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        self._matrix = mat / norms
        logger.info("FAQ 语义索引就绪：faqs=%s phrases=%s", len(self._items_by_id), len(phrases))
        return len(self._items_by_id)

    def search(self, query: str, *, top_k: int = 5) -> list[tuple[dict[str, Any], float, str]]:
        """返回 [(item, score, matched_phrase), ...]，按分数降序，同一 faq 只保留最高分。"""
        if not self.ready or self._matrix is None:
            return []
        q = (query or "").strip()
        if not q:
            return []
        q_vec = np.asarray(embed_text(q), dtype=np.float32)
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm <= 1e-9:
            return []
        q_vec = q_vec / q_norm
        scores = self._matrix @ q_vec  # (N,)

        # 每个 faq_id 取最高分
        best: dict[int, tuple[float, str]] = {}
        for i, score in enumerate(scores.tolist()):
            faq_id = self._faq_ids[i]
            s = float(score)
            prev = best.get(faq_id)
            if prev is None or s > prev[0]:
                best[faq_id] = (s, self._phrases[i])

        ranked = sorted(best.items(), key=lambda x: x[1][0], reverse=True)[: max(1, top_k)]
        out: list[tuple[dict[str, Any], float, str]] = []
        for faq_id, (score, phrase) in ranked:
            item = self._items_by_id.get(faq_id)
            if item is not None:
                out.append((item, score, phrase))
        return out


def _candidate_phrases(item: dict[str, Any]) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()

    def add(value: object) -> None:
        text = str(value or "").strip()
        if not text or text in seen:
            return
        # 过长噪声片段跳过
        if len(text) > 40:
            return
        seen.add(text)
        phrases.append(text)

    variants = item.get("question_variants") or []
    if isinstance(variants, str):
        try:
            variants = json.loads(variants)
        except Exception:
            variants = [variants]
    if isinstance(variants, list):
        for v in variants:
            add(v)

    add(item.get("sub_class"))
    # 内部标签「通用 发货时效」较长但有用，单独放宽
    qt = str(item.get("question_text") or "").strip()
    if qt and qt not in seen and len(qt) <= 48:
        seen.add(qt)
        phrases.append(qt)

    main = str(item.get("main_class") or "").strip()
    sub = str(item.get("sub_class") or "").strip()
    if main and sub:
        add(f"{main}{sub}")
        add(f"{main} {sub}")
    return phrases


_semantic_index = FaqSemanticIndex()
_semantic_ready = False


def refresh_faq_semantic_index() -> int:
    """从 Redis 重建语义向量索引。保留旧函数名兼容灌库脚本。"""
    global _semantic_ready
    store = get_redis_faq_store()
    n = _semantic_index.rebuild(store.get_index())
    _semantic_ready = n > 0
    return n


# 兼容旧调用名
def refresh_bm25_from_redis() -> int:
    return refresh_faq_semantic_index()


def ensure_semantic_ready() -> None:
    global _semantic_ready
    if _semantic_ready and _semantic_index.ready:
        return
    refresh_faq_semantic_index()


def ensure_bm25_ready() -> None:
    ensure_semantic_ready()


def _split_user_clauses(message: str) -> list[str]:
    parts = re.split(r"[？?！!；;。\n]+", message or "")
    return [p.strip() for p in parts if len(p.strip()) >= 2]


def _passes_qa_gate(message: str, answer: str) -> tuple[bool, float]:
    qa = score_question_answer(message, answer)
    return qa >= ANSWER_RELEVANCE_THRESHOLD, float(qa)


def lookup_qa_cache(message: str) -> QaCacheHit | None:
    """Exact → 全库 BGE 语义检索 → 问-答相关度门槛 → 返回答案。"""
    if not QA_CACHE_ENABLED:
        return None
    text = (message or "").strip()
    if not text:
        return None

    # 投诉强信号绝不走缓存（即便入口漏拦）
    try:
        from services.complaint_signals import looks_like_complaint

        if looks_like_complaint(text):
            logger.info("跳过 FAQ 缓存：投诉强信号 text=%r", text[:40])
            return None
    except Exception:
        pass

    # 主题对齐闸：详细/参数、多槽位 → 强制 miss 走 RAG
    slots = detect_user_slots(text)
    if force_rag_detail(text):
        logger.info("slot_miss reason=detail text=%r slots=%s", text[:40], slots)
        return None
    if len(slots) >= 2:
        logger.info("slot_miss reason=multi text=%r slots=%s", text[:40], slots)
        return None
    primary_slot = slots[0] if slots else None

    try:
        redis_store = get_redis_faq_store()
        mysql_store = get_mysql_faq_store()
    except Exception:
        logger.exception("QA 缓存存储不可用")
        return None

    index = redis_store.get_index()
    if not index:
        try:
            rows = mysql_store.list_all()
            if rows:
                redis_store.load_all_questions(rows, preload_answers=False)
                index = redis_store.get_index()
                refresh_faq_semantic_index()
        except Exception:
            logger.exception("从 MySQL 预热 Redis FAQ 索引失败")
            return None
    else:
        ensure_semantic_ready()

    norm = normalize_question(text)

    def finish_hit(
        hit: QaCacheHit,
        *,
        item: dict[str, Any],
        qq: float,
        matched: str,
        match_type: str,
    ) -> QaCacheHit | None:
        ok, qa = _passes_qa_gate(text, hit.answer)
        if not ok:
            logger.info(
                "语义候选问答应过低 faq_id=%s qq=%.3f qa=%.3f(th=%.2f) matched=%r",
                hit.faq_id,
                qq,
                qa,
                ANSWER_RELEVANCE_THRESHOLD,
                matched,
            )
            return None
        hit.score = qq
        hit.qa_relevance = qa
        hit.match_type = match_type
        hit.matched_phrase = matched
        return hit

    # 1) exact：整句 + 分句（归一化命中后校验槽位对齐 + 问-答相关度）
    for raw in [text, *_split_user_clauses(text)]:
        fid = redis_store.resolve_norm(normalize_question(raw)) if raw else None
        if fid is None:
            continue
        hint = redis_store.get_item(fid) or {}
        if not slot_allows_item(slots, str(hint.get("sub_class") or "")):
            logger.info(
                "slot_miss reason=sub_mismatch match=exact slot=%r sub=%r faq=%s",
                primary_slot,
                hint.get("sub_class"),
                fid,
            )
            continue
        hit = _resolve_answer(
            redis_store,
            mysql_store,
            fid,
            score=1.0,
            match_type="exact",
            user_norm=norm,
            item_hint=hint or None,
        )
        if not hit:
            continue
        if not slot_allows_item(slots, hit.sub_class):
            logger.info(
                "slot_miss reason=sub_mismatch match=exact slot=%r sub=%r faq=%s",
                primary_slot,
                hit.sub_class,
                hit.faq_id,
            )
            continue
        done = finish_hit(
            hit,
            item=hint,
            qq=1.0,
            matched=raw.strip(),
            match_type="exact",
        )
        if done:
            return done

    # 2) 全库语义检索（不再走 BM25）
    top_k = max(5, int(QA_BM25_CANDIDATE_TOP_K))
    # 对整句 + 分句分别搜，合并最优
    merged: dict[int, tuple[dict[str, Any], float, str]] = {}
    for q in [text, *_split_user_clauses(text)]:
        try:
            for item, score, phrase in _semantic_index.search(q, top_k=top_k):
                fid = int(item["id"])
                prev = merged.get(fid)
                if prev is None or score > prev[1]:
                    merged[fid] = (item, score, phrase)
        except Exception:
            logger.exception("FAQ 语义检索失败 query=%r", q[:80])
            continue

    if not merged:
        return None

    ranked_all = sorted(merged.values(), key=lambda x: x[1], reverse=True)
    # 主题对齐：先过滤副类不兼容，再取最高分
    filtered = [
        (item, score, phrase)
        for item, score, phrase in ranked_all
        if slot_allows_item(slots, str(item.get("sub_class") or ""))
    ]
    if not filtered:
        top_sub = ranked_all[0][0].get("sub_class") if ranked_all else ""
        logger.info(
            "slot_miss reason=sub_mismatch match=semantic slot=%r top_sub=%r text=%r",
            primary_slot,
            top_sub,
            text[:40],
        )
        return None

    best_item, best_score, best_matched = filtered[0]
    if best_score < QA_SEMANTIC_THRESHOLD:
        logger.info(
            "semantic miss: best_qq=%.3f threshold=%.3f matched=%r faq=%s",
            best_score,
            QA_SEMANTIC_THRESHOLD,
            best_matched,
            best_item.get("id"),
        )
        return None

    faq_id = int(best_item["id"])
    hit = _resolve_answer(
        redis_store,
        mysql_store,
        faq_id,
        score=best_score,
        match_type="semantic",
        user_norm=norm,
        item_hint=best_item,
    )
    if not hit:
        return None
    return finish_hit(
        hit,
        item=best_item,
        qq=best_score,
        matched=best_matched,
        match_type="semantic",
    )


def _resolve_answer(
    redis_store,
    mysql_store,
    faq_id: int,
    *,
    score: float,
    match_type: str,
    user_norm: str,
    item_hint: dict[str, Any] | None = None,
) -> QaCacheHit | None:
    item = item_hint or redis_store.get_item(faq_id) or {}
    answer = redis_store.get_answer(faq_id)
    if not answer:
        row = mysql_store.get_by_id(faq_id)
        if not row or not row.get("answer"):
            return None
        answer = str(row["answer"])
        item = {
            "id": faq_id,
            "main_class": row.get("main_class", ""),
            "qa_type": row.get("qa_type", ""),
            "sub_class": row.get("sub_class", ""),
            "question_text": row.get("question_text", ""),
            "question_variants": row.get("question_variants") or [],
            "source": row.get("source", ""),
        }
        redis_store.set_answer(faq_id, answer, ttl=QA_ANSWER_TTL_SECONDS)
        try:
            mysql_store.incr_hit(faq_id)
        except Exception:
            logger.exception("FAQ hit_count 更新失败 id=%s", faq_id)

    if user_norm:
        redis_store.cache_user_hit(user_norm, faq_id, answer, ttl=QA_ANSWER_TTL_SECONDS)

    return QaCacheHit(
        faq_id=faq_id,
        question_text=str(item.get("question_text") or ""),
        answer=answer,
        score=float(score),
        match_type=match_type,
        main_class=str(item.get("main_class") or ""),
        qa_type=str(item.get("qa_type") or ""),
        sub_class=str(item.get("sub_class") or ""),
        source=str(item.get("source") or "faq_cache"),
    )


def hit_to_response(
    user_id: str,
    message: str,
    hit: QaCacheHit,
    intent: IntentResult | None = None,
) -> WeChatMessageResponse:
    chunk = RetrievedChunk(
        content=(
            f"【缓存命中:{hit.match_type}】\n"
            f"适用商品：{hit.main_class}\n"
            f"场景：{hit.qa_type}\n"
            f"主题：{hit.sub_class}\n"
            f"标准问法：{hit.question_text}\n"
            f"匹配问法：{hit.matched_phrase}\n"
            f"问法相似度：{hit.score:.3f}\n"
            f"问答相关度：{hit.qa_relevance:.3f}\n"
            f"话术：{hit.answer}"
        ),
        source=hit.source or "faq_cache",
        chunk_id=f"faq-{hit.faq_id}",
        page=0,
        score=hit.score,
        section=hit.qa_type or "cache",
        chunk_type="script_faq",
        question=hit.matched_phrase or hit.question_text or hit.sub_class,
    )
    response = WeChatMessageResponse(
        user_id=user_id,
        route=RouteType.RAG_AGENT,
        reply_mode=ReplyMode.CACHE,
        answer=hit.answer,
        sources=[chunk],
        answer_confidence=float(hit.score),
        answer_supported=True,
        needs_handoff=False,
        confidence_reason=(
            f"qa_cache:{hit.match_type}, qq={hit.score:.3f}, qa={hit.qa_relevance:.3f}, "
            f"matched={hit.matched_phrase}"
        ),
        query_strategy=f"cache_{hit.match_type}",
        query_strategy_name=(
            "缓存精确匹配" if hit.match_type == "exact" else "缓存语义检索(BGE)"
        ),
        query_strategy_reason=(
            f"faq_id={hit.faq_id}, qq={hit.score:.3f}, qa={hit.qa_relevance:.3f}, "
            f"matched={hit.matched_phrase}"
        ),
    )
    if intent is not None:
        response.intent = intent.category.value
        response.intent_confidence = intent.confidence
        response.action = intent.action
        response.intent_probabilities = intent.probabilities
    return response
