"""仓库级共用：Redis / MySQL 会话存储配置。

两套入口（RAG_mode、crm_agent）均从此模块读取，避免各写一份。
优先级：环境变量 > 仓库根目录 `.env.session` > 下方默认值。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSION_ENV_FILE = Path(
    os.getenv("RAG_SESSION_ENV_FILE", str(REPO_ROOT / ".env.session"))
)

# 只加载会话相关配置；不覆盖已有环境变量（override=False）
if SESSION_ENV_FILE.is_file():
    load_dotenv(SESSION_ENV_FILE, override=False)

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "123")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(24 * 3600)))

MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "123")
# 会话等业务库（勿与 FAQ 混用）
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rag_app")
# 商品话术 FAQ 独立库（不影响会话表）
MYSQL_FAQ_DATABASE = os.getenv("MYSQL_FAQ_DATABASE", "rag_faq")
