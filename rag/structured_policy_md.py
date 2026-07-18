"""客服岗位职责与行为规范 Markdown 结构化切分（软隔离）。

软隔离字段：
- kb=policy
- chunk_type=policy_norm
- reason=kb=policy
- page_content 前缀【客服规范】增强向量区分度
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document as LCDocument

KB_POLICY = "policy"
CHUNK_POLICY = "policy_norm"

_H2_RE = re.compile(r"^##\s+(.+)$")
_H3_RE = re.compile(r"^###\s+(.+)$")
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def load_structured_policy_md(file_path: str | Path) -> list[LCDocument]:
    """按 ## / ### 标题切块；保留编号条目上下文。"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文档不存在: {path}")
    if path.suffix.lower() not in {".md", ".markdown", ".txt"}:
        raise ValueError(f"仅支持 Markdown/文本: {path}")

    text = path.read_text(encoding="utf-8")
    source = path.name
    sections = _split_by_headings(text)
    if not sections:
        raise ValueError(f"行为规范未切分出有效章节: {source}")

    chunks: list[LCDocument] = []
    for idx, sec in enumerate(sections):
        title = sec["title"]
        body = sec["body"].strip()
        if not body:
            continue
        content = f"【客服规范】\n章节：{title}\n{body}"
        chunks.append(
            _make_chunk(
                content,
                source=source,
                chunk_index=idx,
                section=title[:200],
                question=title[:200],
                keyword="客服规范",
            )
        )

    if not chunks:
        raise ValueError(f"行为规范未切分出有效 chunk: {source}")
    return chunks


def _split_by_headings(text: str) -> list[dict[str, str]]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sections: list[dict[str, str]] = []
    h2 = ""
    h3 = ""
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        body = "\n".join(buf).strip()
        buf = []
        if not body:
            return
        title_parts = [p for p in (h2, h3) if p]
        title = " / ".join(title_parts) if title_parts else "总则"
        sections.append({"title": _clean_title(title), "body": body})

    for line in lines:
        m2 = _H2_RE.match(line.strip())
        m3 = _H3_RE.match(line.strip())
        if m2:
            flush()
            h2 = m2.group(1).strip()
            h3 = ""
            continue
        if m3:
            flush()
            h3 = m3.group(1).strip()
            continue
        # 跳过一级标题（文档名）
        if line.strip().startswith("# ") and not line.strip().startswith("##"):
            continue
        buf.append(line)

    flush()
    return sections


def _clean_title(title: str) -> str:
    title = _MD_BOLD_RE.sub(r"\1", title or "").strip()
    return re.sub(r"\s+", " ", title)


def _make_chunk(
    content: str,
    *,
    source: str,
    chunk_index: int,
    section: str,
    question: str = "",
    keyword: str = "",
) -> LCDocument:
    chunk_id = hashlib.sha256(
        f"{source}|{KB_POLICY}|{CHUNK_POLICY}|{chunk_index}|{content[:48]}".encode("utf-8")
    ).hexdigest()[:16]
    metadata = {
        "source": source,
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "page": 0,
        "section": section,
        "chunk_type": CHUNK_POLICY,
        "kb": KB_POLICY,
        "reason": f"kb={KB_POLICY}",
    }
    if question:
        metadata["question"] = question[:200]
    if keyword:
        metadata["keyword"] = keyword[:200]
    return LCDocument(page_content=content, metadata=metadata)
