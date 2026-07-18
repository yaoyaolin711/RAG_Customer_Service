"""Excel 结构化切分：产品资料库 + 商品话术 FAQ（一行一块，软隔离）。

软隔离字段：
- chunk_type: product_card | script_faq
- section / question / keyword: 便于检索侧按知识域过滤或加权
- page_content 前缀【产品资料】/【商品话术】增强向量区分度
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document as LCDocument

KB_PRODUCT = "product"
KB_SCRIPT = "script"

CHUNK_PRODUCT = "product_card"
CHUNK_SCRIPT = "script_faq"

SECTION_PRODUCT = "产品资料库"

# 产品卡写入字段：(Excel 列名候选, 展示标签)
_PRODUCT_FIELDS: list[tuple[tuple[str, ...], str]] = [
    (("产品名称",), "产品名称"),
    (("产品定价", "定价", "价格"), "价格"),
    (("在售状态🛒", "在售状态"), "在售状态"),
    (("产品材质", "材质"), "材质"),
    (("产品规格", "规格"), "规格"),
    (("尺寸/保质期", "尺寸", "保质期"), "尺寸/保质期"),
    (("产品重量（kg）", "产品重量", "重量"), "重量"),
    (("售后政策", "售后"), "售后政策"),
    (("赠品",), "赠品"),
    (("产品卖点🔥", "产品卖点", "卖点"), "卖点"),
    (("产品简介", "简介"), "简介"),
    (("使用方法", "用法"), "使用方法"),
]

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"
    "\U00002600-\U000027BF"
    "\U0000FE00-\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)


def load_structured_xlsx(file_path: str | Path) -> list[LCDocument]:
    """按表头自动识别产品资料库 / 话术 FAQ，一行一块。"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文档不存在: {path}")
    if path.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError(f"仅支持 Excel 文件: {path}")

    df = pd.read_excel(path, sheet_name=0)
    df.columns = [_normalize_header(c) for c in df.columns]
    source = path.name

    if _has_columns(df, "产品名称") and (
        _has_columns(df, "产品定价", "定价", "价格") or _has_columns(df, "产品简介", "简介")
    ):
        return _load_product_catalog(df, source)

    if _has_columns(df, "快捷话术") and _has_columns(df, "主类"):
        return _load_script_faq(df, source)

    raise ValueError(
        f"无法识别 Excel 结构（期望产品资料库或商品话术FAQ）: {path.name} "
        f"列={list(df.columns)}"
    )


def _normalize_header(value: object) -> str:
    text = str(value).strip() if value is not None and not (isinstance(value, float) and pd.isna(value)) else ""
    return _EMOJI_RE.sub("", text).strip()


def _has_columns(df: pd.DataFrame, *names: str) -> bool:
    cols = set(df.columns)
    return any(n in cols for n in names)


def _cell(row: pd.Series, *names: str) -> str:
    for name in names:
        if name not in row.index:
            continue
        value = row.get(name)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def _stable_chunk_id(*parts: object) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _make_chunk(
    content: str,
    *,
    source: str,
    chunk_index: int,
    section: str,
    chunk_type: str,
    kb: str,
    question: str = "",
    keyword: str = "",
) -> LCDocument:
    chunk_id = _stable_chunk_id(source, kb, chunk_type, chunk_index, content[:48])
    metadata = {
        "source": source,
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "page": 0,
        "section": section,
        "chunk_type": chunk_type,
        "kb": kb,
        # reason 复用为软隔离标签，便于现有 schema 持久化与召回回传
        "reason": f"kb={kb}",
    }
    if question:
        metadata["question"] = question[:200]
    if keyword:
        metadata["keyword"] = keyword[:200]
    return LCDocument(page_content=content, metadata=metadata)


def _load_product_catalog(df: pd.DataFrame, source: str) -> list[LCDocument]:
    chunks: list[LCDocument] = []
    chunk_index = 0

    for _, row in df.iterrows():
        name = _cell(row, "产品名称")
        if not name:
            continue

        lines = ["【产品资料】"]
        for candidates, label in _PRODUCT_FIELDS:
            value = _cell(row, *candidates)
            if not value:
                continue
            if label == "在售状态":
                value = _EMOJI_RE.sub("", value).strip() or value
            lines.append(f"{label}：{value}")

        if len(lines) <= 1:
            continue

        chunks.append(
            _make_chunk(
                "\n".join(lines),
                source=source,
                chunk_index=chunk_index,
                section=SECTION_PRODUCT,
                chunk_type=CHUNK_PRODUCT,
                kb=KB_PRODUCT,
                question=name,
                keyword=name,
            )
        )
        chunk_index += 1

    if not chunks:
        raise ValueError(f"产品资料库未切分出有效 chunk: {source}")
    return chunks


def _load_script_faq(df: pd.DataFrame, source: str) -> list[LCDocument]:
    chunks: list[LCDocument] = []
    chunk_index = 0

    for _, row in df.iterrows():
        script = _cell(row, "快捷话术")
        if not script:
            continue

        main = _cell(row, "主类") or "通用"
        typ = _cell(row, "类型") or "通用"
        sub = _cell(row, "副类") or "话术"

        content = (
            f"【商品话术】\n"
            f"适用商品：{main}\n"
            f"场景：{typ}\n"
            f"主题：{sub}\n"
            f"话术：{script}"
        )
        chunks.append(
            _make_chunk(
                content,
                source=source,
                chunk_index=chunk_index,
                section=typ,
                chunk_type=CHUNK_SCRIPT,
                kb=KB_SCRIPT,
                question=sub,
                keyword=main,
            )
        )
        chunk_index += 1

    if not chunks:
        raise ValueError(f"商品话术 FAQ 未切分出有效 chunk: {source}")
    return chunks
