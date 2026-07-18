# settings.py — 抖音店铺买家智能客服配置

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared_config.session_store import (  # noqa: E402
    MYSQL_DATABASE,
    MYSQL_FAQ_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    MYSQL_USER,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    SESSION_TTL_SECONDS,
)

DEEPSEEK_KEY_ENV = "DEEPSEEK_KEY"
DEEPSEEK_KEY_FILE = os.getenv(
    "DEEPSEEK_KEY_FILE",
    str(REPO_ROOT / "DEEPSEEK_KEY.txt"),
)

# LLM 配置（DeepSeek 兼容 OpenAI 接口）
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "deepseek-chat")
LLM_MODEL_BASE_URL = os.getenv("LLM_MODEL_BASE_URL", "https://api.deepseek.com/v1")

# 服务端口（避免 3/5/8 开头）
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "7121"))


def _read_key_from_file(path: str | Path) -> str | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    content = file_path.read_text(encoding="utf-8").strip()
    return content or None


def get_deepseek_key() -> str:
    key = os.getenv(DEEPSEEK_KEY_ENV) or _read_key_from_file(DEEPSEEK_KEY_FILE)
    if not key:
        raise ValueError(
            f"未找到 DeepSeek API Key，请设置环境变量 {DEEPSEEK_KEY_ENV} "
            f"或在 {DEEPSEEK_KEY_FILE} 中写入密钥"
        )
    return key


def get_aliyun_api_key() -> str:
    """兼容旧调用，实际读取 DEEPSEEK_KEY。"""
    return get_deepseek_key()


# 本地 Embedding 模型（BGE-M3，稠密 + 稀疏）
BGE_M3_PATH = os.getenv("BGE_M3_PATH", r"E:\model\BGE-M3\BGE-M3")
BGE_M3_DEVICE = os.getenv("BGE_M3_DEVICE", "cpu")  # 有 GPU 可改为 "cuda"

# 本地 Reranker（BGE-Reranker-v2-m3，GPU FP16，目标显存约 4GB）
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
RERANKER_PATH = os.getenv("RERANKER_PATH", r"E:\model\BGE-Reranker-v2-m3\BGE-Reranker-v2-m3")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cuda")  # cuda / cpu / auto
RERANKER_USE_FP16 = os.getenv("RERANKER_USE_FP16", "true").lower() in {"1", "true", "yes", "on"}
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "8"))
RERANKER_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "512"))
RAG_CANDIDATE_K = int(os.getenv("RAG_CANDIDATE_K", "20"))  # 重排前候选数
RERANK_RELEVANCE_THRESHOLD = float(os.getenv("RERANK_RELEVANCE_THRESHOLD", "0.2"))

# 查询策略选择器（仅咨询类 RAG）
QUERY_STRATEGY_ENABLED = os.getenv("QUERY_STRATEGY_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
QUERY_STRATEGY_DEFAULT = os.getenv("QUERY_STRATEGY_DEFAULT", "direct").strip().lower()


# Milvus 向量数据库（Docker Standalone）
MILVUS_PATH = r"E:\model\Milvus\Docker\volumes\milvus"
MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = ""
MILVUS_COLLECTION_NAME = "jincheng_mall"

# 知识库路径（仓库根目录 data/）
DATA_PATH = str(REPO_ROOT / "data")

# 文本分割参数（锦丞 Demo 旧配置，build_vectorstore.py 仍使用）
CHUNK_SIZE = 150
CHUNK_OVERLAP = 20

# RAG 入库专用配置
RAG_COLLECTION_NAME = "rag_collection"
RAG_CHUNK_SIZE = 600  # tokens，范围 500~800
RAG_CHUNK_OVERLAP = 80  # tokens
RAG_EMBEDDING_BATCH_SIZE = 32
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
RAG_INDEX_MANIFEST_PATH = os.path.join(MILVUS_PATH, "rag_index_manifest.json")
RAG_DATA_PATH = DATA_PATH  # 默认文档目录，可通过 CLI 覆盖

# 混合检索权重
MILVUS_DENSE_WEIGHT = 0.7
MILVUS_SPARSE_WEIGHT = 0.3

# 检索参数
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.45"))
ANSWER_CONFIDENCE_THRESHOLD = float(os.getenv("ANSWER_CONFIDENCE_THRESHOLD", "0.75"))
# 问-答语义相关性（BGE 余弦）。问题与答案语体不同，阈值通常低于问-问。
ANSWER_RELEVANCE_THRESHOLD = float(os.getenv("ANSWER_RELEVANCE_THRESHOLD", "0.50"))


# 知识库文档（文件名保留兼容已入库资源；对外展示用中性名称）
KB_DOC_NAME = "BD筛选提示词.docx"
KB_DOC_DISPLAY_NAME = "店铺知识库"

# 意图识别模型（MacBERT + LoRA，v4b 领域增强版）
INTENT_MODEL_ADAPTER_PATH = os.getenv(
    "INTENT_MODEL_ADAPTER_PATH",
    r"E:\model\AI_Project\AI_Project\Intent_Classification_Trainer\outputs\intent_model_v4b\best_model",
)
INTENT_MODEL_BASE_PATH = os.getenv(
    "INTENT_MODEL_BASE_PATH",
    r"E:\model\AI_Project\AI_Project\Intent_Classification_Trainer\models\macbert",
)
INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.35"))
INTENT_DEVICE = os.getenv("INTENT_DEVICE", "cpu")

# 投诉工单库
COMPLAINT_DB_PATH = os.path.join(MILVUS_PATH, "complaint_tickets.db")

# 咨询类多级问答缓存（Redis exact → BM25 候选 → BGE 语义门槛 → MySQL）
QA_CACHE_ENABLED = os.getenv("QA_CACHE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
# 兼容旧名：现作为「问法语义相似度」门槛（BGE 余弦），建议 0.80~0.88；0.95 过严易过度 miss
QA_BM25_THRESHOLD = float(os.getenv("QA_BM25_THRESHOLD", "0.82"))
QA_SEMANTIC_THRESHOLD = float(os.getenv("QA_SEMANTIC_THRESHOLD", str(QA_BM25_THRESHOLD)))
QA_BM25_CANDIDATE_TOP_K = int(os.getenv("QA_BM25_CANDIDATE_TOP_K", "5"))
# 用户热问答案缓存 TTL；问题索引本身不过期
QA_ANSWER_TTL_SECONDS = int(os.getenv("QA_ANSWER_TTL_SECONDS", str(24 * 3600)))

# Redis / MySQL 会话：见仓库根目录 shared_config/session_store.py + .env.session
