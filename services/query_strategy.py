"""查询策略选择器：仅用于咨询类 RAG 路径。

策略（默认直接检索）：
- direct         直接检索
- rewrite        查询改写检索
- multi_query    子问题检索
- keyword_boost  关键词增强检索

使用与项目相同的 LLM_MODEL_* 与 DEEPSEEK_KEY。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from settings import (
    LLM_MODEL_BASE_URL,
    LLM_MODEL_NAME,
    QUERY_STRATEGY_DEFAULT,
    QUERY_STRATEGY_ENABLED,
    get_aliyun_api_key,
)

logger = logging.getLogger(__name__)


class QueryStrategy(str, Enum):
    DIRECT = "direct"
    REWRITE = "rewrite"
    MULTI_QUERY = "multi_query"
    KEYWORD_BOOST = "keyword_boost"


STRATEGY_NAMES: dict[str, str] = {
    QueryStrategy.DIRECT.value: "直接检索",
    QueryStrategy.REWRITE.value: "查询改写检索",
    QueryStrategy.MULTI_QUERY.value: "子问题检索",
    QueryStrategy.KEYWORD_BOOST.value: "关键词增强检索",
}

_SELECTOR_PROMPT = """你是 RAG 检索策略选择器。只为「已确定要查知识库」的咨询类问题选择策略，不要回答用户问题。

可选策略：
1. direct — 直接检索：问题短、意图单一、表达清晰（默认优先）
2. rewrite — 查询改写检索：口语、指代、语序乱，需要改写成标准检索问句
3. multi_query — 子问题检索：一句话里包含多个并列子问题（以及/还有/分别问规格+物流+售后等）
4. keyword_boost — 关键词增强检索：强依赖专有品类词/政策口令（如运费险、七天无理由、包邮）

规则：
- 多数简单问题选 direct
- 只有必要时才选其他策略
- queries：要实际用于检索的查询列表；direct 时填原问题；rewrite 填 1 条改写问句；multi_query 填 2~3 条子问；keyword_boost 填 1 条「原问+关键词」增强问句
- keywords：keyword_boost 时给出 1~5 个关键词，其他策略可为空数组
- 只输出 JSON，不要 Markdown

输出格式：
{
  "strategy": "direct",
  "queries": ["原问题或改写/子问"],
  "keywords": [],
  "reason": "一句话原因"
}
"""


@dataclass
class QueryStrategyPlan:
    strategy: QueryStrategy
    strategy_name: str
    original_query: str
    queries: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "strategy_name": self.strategy_name,
            "original_query": self.original_query,
            "queries": list(self.queries),
            "keywords": list(self.keywords),
            "reason": self.reason,
        }


def _default_plan(query: str, *, reason: str = "默认直接检索") -> QueryStrategyPlan:
    strategy = QueryStrategy.DIRECT
    default_code = (QUERY_STRATEGY_DEFAULT or "direct").strip().lower()
    if default_code in STRATEGY_NAMES:
        strategy = QueryStrategy(default_code)
    return QueryStrategyPlan(
        strategy=strategy,
        strategy_name=STRATEGY_NAMES[strategy.value],
        original_query=query,
        queries=[query],
        keywords=[],
        reason=reason,
    )


def _parse_json(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = next((p for p in parts if "{" in p and "}" in p), raw)
        raw = raw[raw.find("{") : raw.rfind("}") + 1]
    return json.loads(raw)


def _normalize_plan(query: str, payload: dict) -> QueryStrategyPlan:
    code = str(payload.get("strategy", "direct")).strip().lower()
    if code not in STRATEGY_NAMES:
        code = QueryStrategy.DIRECT.value

    strategy = QueryStrategy(code)
    queries_raw = payload.get("queries") or []
    if isinstance(queries_raw, str):
        queries_raw = [queries_raw]
    queries = [str(q).strip() for q in queries_raw if str(q).strip()]

    keywords_raw = payload.get("keywords") or []
    if isinstance(keywords_raw, str):
        keywords_raw = [keywords_raw]
    keywords = [str(k).strip() for k in keywords_raw if str(k).strip()][:5]

    if strategy == QueryStrategy.DIRECT:
        queries = [query]
    elif strategy == QueryStrategy.REWRITE:
        if not queries:
            queries = [query]
        queries = queries[:1]
    elif strategy == QueryStrategy.MULTI_QUERY:
        if len(queries) < 2:
            # 不可靠的 multi_query 降级为直接检索
            return _default_plan(query, reason="子问题不足，降级为直接检索")
        queries = queries[:3]
    elif strategy == QueryStrategy.KEYWORD_BOOST:
        if keywords:
            boost = f"{query} {' '.join(keywords)}".strip()
            queries = [queries[0] if queries else boost]
            if keywords and keywords[0] not in queries[0]:
                queries = [boost]
        else:
            queries = [query] if not queries else queries[:1]

    if not queries:
        queries = [query]

    return QueryStrategyPlan(
        strategy=strategy,
        strategy_name=STRATEGY_NAMES[strategy.value],
        original_query=query,
        queries=queries,
        keywords=keywords,
        reason=str(payload.get("reason", "")).strip() or STRATEGY_NAMES[strategy.value],
    )


class QueryStrategySelector:
    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = init_chat_model(
                model=LLM_MODEL_NAME,
                model_provider="openai",
                api_key=get_aliyun_api_key(),
                base_url=LLM_MODEL_BASE_URL,
                temperature=0,
            )
        return self._llm

    def select(self, query: str) -> QueryStrategyPlan:
        text = (query or "").strip()
        if not text:
            return _default_plan("", reason="空查询，直接检索")
        if not QUERY_STRATEGY_ENABLED:
            return _default_plan(text, reason="策略选择器关闭，使用默认策略")

        try:
            result = self._get_llm().invoke(
                [
                    SystemMessage(content=_SELECTOR_PROMPT),
                    HumanMessage(content=f"用户问题：\n{text}"),
                ]
            )
            content = getattr(result, "content", result)
            payload = _parse_json(str(content))
            return _normalize_plan(text, payload)
        except Exception:
            logger.exception("查询策略选择失败，回退直接检索")
            return _default_plan(text, reason="选择器异常，回退直接检索")


_selector: QueryStrategySelector | None = None


def get_query_strategy_selector() -> QueryStrategySelector:
    global _selector
    if _selector is None:
        _selector = QueryStrategySelector()
    return _selector


def select_query_strategy(query: str) -> QueryStrategyPlan:
    return get_query_strategy_selector().select(query)
