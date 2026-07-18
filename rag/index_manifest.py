"""Index 一致性：记录 embedding 模型与切分参数，模型变更时提示重建。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from embedding import get_embedding_model_fingerprint
from settings import (
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_COLLECTION_NAME,
    RAG_INDEX_MANIFEST_PATH,
)


@dataclass
class IndexManifest:
    embedding_model_id: str
    embedding_model_path: str
    embedding_type: str
    normalize_embeddings: bool
    chunk_size: int
    chunk_overlap: int
    collection_name: str

    @classmethod
    def current(cls) -> "IndexManifest":
        fp = get_embedding_model_fingerprint()
        return cls(
            embedding_model_id=fp["model_id"],
            embedding_model_path=fp["model_path"],
            embedding_type=fp["embedding_type"],
            normalize_embeddings=fp["normalize_embeddings"],
            chunk_size=RAG_CHUNK_SIZE,
            chunk_overlap=RAG_CHUNK_OVERLAP,
            collection_name=RAG_COLLECTION_NAME,
        )

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path or RAG_INDEX_MANIFEST_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "IndexManifest | None":
        target = Path(path or RAG_INDEX_MANIFEST_PATH)
        if not target.exists():
            return None
        raw = target.read_text(encoding="utf-8-sig").strip()
        if not raw or raw == "{}":
            return None
        data = json.loads(raw)
        required = {
            "embedding_model_id",
            "embedding_model_path",
            "embedding_type",
            "normalize_embeddings",
            "chunk_size",
            "chunk_overlap",
            "collection_name",
        }
        if not required.issubset(data.keys()):
            return None
        return cls(**{k: data[k] for k in required})


def check_index_consistency(manifest_path: str | Path | None = None) -> tuple[bool, str]:
    """
    校验当前配置与已存 manifest 是否一致。
    返回 (is_consistent, message)。
    """
    existing = IndexManifest.load(manifest_path)
    current = IndexManifest.current()

    if existing is None:
        return True, "未发现已有 index manifest，将创建新索引。"

    mismatches: list[str] = []
    for field in (
        "embedding_model_id",
        "embedding_model_path",
        "chunk_size",
        "chunk_overlap",
        "collection_name",
    ):
        if getattr(existing, field) != getattr(current, field):
            mismatches.append(
                f"  - {field}: 已有={getattr(existing, field)!r}, 当前={getattr(current, field)!r}"
            )

    if mismatches:
        msg = (
            "⚠️  检测到 embedding 模型或切分参数已变更，现有 index 可能不兼容，请重建索引：\n"
            + "\n".join(mismatches)
            + "\n\n建议执行: python scripts/ingest_documents.py --rebuild"
        )
        return False, msg

    return True, "Index manifest 与当前配置一致。"
