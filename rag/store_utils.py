"""向量库清理工具。"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from settings import MILVUS_PATH, MILVUS_URI, RAG_INDEX_MANIFEST_PATH
from vectorstore import close_milvus_client, delete_collection, list_collections


def clear_all_collections() -> list[str]:
    """删除 Milvus 中所有 collection，返回已删除名称列表。

    Windows + Milvus Lite 下 drop_collection 可能因 manifest rename 失败；
    失败时回退为关闭连接并物理清空 milvus.db 目录（不碰 complaint_tickets.db）。
    """
    deleted: list[str] = []
    api_ok = True
    try:
        names = list(list_collections())
        for name in names:
            try:
                delete_collection(name)
                deleted.append(name)
                print(f"已删除 collection: {name}")
            except Exception as exc:
                api_ok = False
                print(f"API 删除失败 {name}: {exc}")
                break
    except Exception as exc:
        api_ok = False
        print(f"列举 collection 失败，改物理清空: {exc}")

    if not api_ok or _need_force_wipe():
        wiped = force_wipe_milvus_lite()
        if wiped:
            print(f"已物理清空 Milvus Lite: {wiped}")
        deleted = deleted or ["*force_wipe*"]
    return deleted


def _need_force_wipe() -> bool:
    """drop 后若 collection 目录仍残留，视为需要强制清空。"""
    root = Path(MILVUS_URI)
    if not root.exists():
        return False
    # milvus.db 可能是目录（Lite）
    collections = root / "collections"
    return collections.exists() and any(collections.iterdir())


def force_wipe_milvus_lite() -> str:
    """关闭客户端后删除 milvus.db（目录或文件），保留同目录下其它库文件。"""
    close_milvus_client()
    time.sleep(0.3)
    target = Path(MILVUS_URI)
    if not target.exists():
        return ""
    # 兼容 uri 指向 .db 文件或目录
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
        # Windows 偶发锁：再试一次
        if target.exists():
            time.sleep(0.5)
            shutil.rmtree(target, ignore_errors=True)
    else:
        target.unlink(missing_ok=True)
        # Lite 有时旁路同名目录
        sidecar = Path(str(target) + ".lock")
        if sidecar.exists():
            sidecar.unlink(missing_ok=True)
    return str(target)


def clear_index_manifest() -> None:
    manifest = Path(RAG_INDEX_MANIFEST_PATH)
    if manifest.exists():
        manifest.unlink()
        print(f"已删除 index manifest: {manifest}")
