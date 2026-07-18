"""MacBERT + LoRA 四分类意图识别。"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from services.models import IntentCategory, IntentResult
from settings import (
    INTENT_CONFIDENCE_THRESHOLD,
    INTENT_DEVICE,
    INTENT_MODEL_ADAPTER_PATH,
    INTENT_MODEL_BASE_PATH,
)

logger = logging.getLogger(__name__)

ID2LABEL = {0: "咨询类", 1: "交易类", 2: "投诉类", 3: "其他类"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
LABEL2INTENT = {v: IntentCategory(v) for v in ID2LABEL.values()}

ACTION_MAP = {
    IntentCategory.CONSULT: "knowledge_base_search",
    IntentCategory.TRANSACTION: "handle_transaction",
    IntentCategory.COMPLAINT: "escalate_to_human",
    IntentCategory.OTHER: "chat_or_fallback",
}

_classifier: Optional["IntentClassifier"] = None


class IntentClassifier:
    """单例懒加载意图分类器。"""

    def __init__(
        self,
        adapter_path: str | None = None,
        base_path: str | None = None,
        device: str | None = None,
        confidence_threshold: float | None = None,
    ):
        self.adapter_path = adapter_path or INTENT_MODEL_ADAPTER_PATH
        self.base_path = base_path or INTENT_MODEL_BASE_PATH
        self.device = device or INTENT_DEVICE
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else INTENT_CONFIDENCE_THRESHOLD
        )
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        logger.info(
            "加载意图模型 base=%s adapter=%s device=%s",
            self.base_path,
            self.adapter_path,
            self.device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_path,
            num_labels=4,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        self._model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self._model.to(self.device)
        self._model.eval()

    def classify(self, text: str) -> IntentResult:
        stripped = (text or "").strip()
        if not stripped:
            return IntentResult(
                category=IntentCategory.OTHER,
                confidence=0.0,
                action="fallback",
                raw_text=text or "",
                is_fallback=True,
            )

        self._ensure_loaded()
        t0 = time.time()
        inputs = self._tokenizer(
            stripped,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()

        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        category = LABEL2INTENT[ID2LABEL[pred_id]]
        latency = (time.time() - t0) * 1000

        if confidence < self.confidence_threshold:
            return IntentResult(
                category=category,
                confidence=confidence,
                action="fallback",
                probabilities={ID2LABEL[i]: float(probs[i]) for i in range(len(ID2LABEL))},
                raw_text=stripped,
                latency_ms=latency,
                is_fallback=True,
            )

        return IntentResult(
            category=category,
            confidence=confidence,
            action=ACTION_MAP[category],
            probabilities={ID2LABEL[i]: float(probs[i]) for i in range(len(ID2LABEL))},
            raw_text=stripped,
            latency_ms=latency,
            is_fallback=False,
        )


def get_intent_classifier() -> IntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier


def classify_intent(text: str) -> IntentResult:
    return get_intent_classifier().classify(text)
