"""本地 BGE-Reranker-v2-m3：GPU FP16 二次排序，显存控制在约 4GB。"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from settings import (
    RERANK_ENABLED,
    RERANKER_BATCH_SIZE,
    RERANKER_DEVICE,
    RERANKER_MAX_LENGTH,
    RERANKER_PATH,
    RERANKER_USE_FP16,
)

logger = logging.getLogger(__name__)

_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSequenceClassification | None = None
_device: str | None = None


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def _ensure_loaded(model_path: str | None = None) -> tuple[AutoTokenizer, AutoModelForSequenceClassification, str]:
    global _tokenizer, _model, _device
    if _model is not None and _tokenizer is not None and _device is not None:
        return _tokenizer, _model, _device

    path = model_path or RERANKER_PATH
    device = _resolve_device(RERANKER_DEVICE)
    use_fp16 = bool(RERANKER_USE_FP16) and device.startswith("cuda")

    logger.info(
        "加载 Reranker path=%s device=%s fp16=%s batch=%s max_len=%s",
        path,
        device,
        use_fp16,
        RERANKER_BATCH_SIZE,
        RERANKER_MAX_LENGTH,
    )

    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    dtype = torch.float16 if use_fp16 else torch.float32
    try:
        model = AutoModelForSequenceClassification.from_pretrained(path, dtype=dtype)
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=dtype)
    model.to(device)
    model.eval()

    _tokenizer = tokenizer
    _model = model
    _device = device
    return tokenizer, model, device


def get_reranker(model_path: str | None = None) -> AutoModelForSequenceClassification:
    """懒加载并返回模型（侧载 tokenizer）。"""
    _, model, _ = _ensure_loaded(model_path)
    return model


def rerank_pairs(
    query: str,
    passages: Sequence[str],
    *,
    model_path: str | None = None,
) -> list[float]:
    """对 (query, passage) 打分，返回与 passages 同序的 0~1 分数。"""
    if not passages:
        return []

    tokenizer, model, device = _ensure_loaded(model_path)
    batch_size = max(1, int(RERANKER_BATCH_SIZE))
    max_length = int(RERANKER_MAX_LENGTH)
    scores: list[float] = []

    with torch.inference_mode():
        for start in range(0, len(passages), batch_size):
            batch_passages = list(passages[start : start + batch_size])
            pairs = [[query, p] for p in batch_passages]
            encoded = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded, return_dict=True).logits.view(-1)
            probs = torch.sigmoid(logits.float()).detach().cpu().tolist()
            scores.extend(float(x) for x in probs)

    return scores


def is_rerank_enabled() -> bool:
    return bool(RERANK_ENABLED)
