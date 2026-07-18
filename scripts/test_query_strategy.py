"""查询策略选择器单元测试（不打外部 API）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.query_strategy import QueryStrategy, _default_plan, _normalize_plan


def test_default_direct():
    plan = _default_plan("多久发货")
    assert plan.strategy == QueryStrategy.DIRECT
    assert plan.strategy_name == "直接检索"
    assert plan.queries == ["多久发货"]


def test_normalize_rewrite():
    plan = _normalize_plan(
        "那个包邮不",
        {
            "strategy": "rewrite",
            "queries": ["是否包邮"],
            "keywords": [],
            "reason": "口语指代",
        },
    )
    assert plan.strategy == QueryStrategy.REWRITE
    assert plan.strategy_name == "查询改写检索"
    assert plan.queries == ["是否包邮"]


def test_normalize_multi_query():
    plan = _normalize_plan(
        "有运费险吗，多久能发货",
        {
            "strategy": "multi_query",
            "queries": ["是否有运费险", "多久发货"],
            "keywords": [],
            "reason": "并列子问题",
        },
    )
    assert plan.strategy == QueryStrategy.MULTI_QUERY
    assert plan.strategy_name == "子问题检索"
    assert len(plan.queries) == 2


def test_multi_query_fallback_when_too_few():
    plan = _normalize_plan(
        "运费险和发货",
        {"strategy": "multi_query", "queries": ["运费险"], "keywords": [], "reason": "x"},
    )
    assert plan.strategy == QueryStrategy.DIRECT


def test_keyword_boost():
    plan = _normalize_plan(
        "有没有七天无理由",
        {
            "strategy": "keyword_boost",
            "queries": [],
            "keywords": ["七天无理由"],
            "reason": "政策口令",
        },
    )
    assert plan.strategy == QueryStrategy.KEYWORD_BOOST
    assert plan.strategy_name == "关键词增强检索"
    assert "七天无理由" in plan.queries[0]


if __name__ == "__main__":
    test_default_direct()
    test_normalize_rewrite()
    test_normalize_multi_query()
    test_multi_query_fallback_when_too_few()
    test_keyword_boost()
    print("query_strategy unit tests OK")
