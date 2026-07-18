"""多格式文档加载器。"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_documents_from_dir(data_dir: str | Path) -> list[Document]:
    """从目录递归加载 txt / md / pdf 文档。"""
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"文档目录不存在: {root}")

    documents: list[Document] = []
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        raise FileNotFoundError(f"目录中未找到支持的文档 ({', '.join(SUPPORTED_EXTENSIONS)}): {root}")

    for file_path in files:
        documents.extend(_load_single_file(file_path, root))

    return documents


def _load_single_file(file_path: Path, root: Path) -> list[Document]:
    suffix = file_path.suffix.lower()
    relative_source = str(file_path.relative_to(root)).replace("\\", "/")

    if suffix in {".txt", ".md"}:
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
    elif suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
    else:
        return []

    for doc in docs:
        doc.metadata["source"] = relative_source
        if "page" not in doc.metadata and suffix != ".pdf":
            doc.metadata["page"] = 0

    return docs


def load_documents_from_paths(paths: list[str]) -> list[Document]:
    """从指定文件路径列表加载。"""
    documents: list[Document] = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_dir():
            documents.extend(load_documents_from_dir(path))
        elif path.is_file():
            documents.extend(_load_single_file(path, path.parent))
        else:
            raise FileNotFoundError(f"路径不存在: {path_str}")
    return documents
