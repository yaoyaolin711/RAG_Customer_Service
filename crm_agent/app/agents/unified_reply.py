"""统一回复 Agent：意图识别 → 分流 → 咨询类 RAG + 历史 → LLM → 置信评估。"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

from app.agents.talent_base import TalentBaseAgent
from app.agents.tools.rag_tools import SEARCH_KNOWLEDGE_BASE_TOOL, search_knowledge_base
from app.agents.tools.registry import ToolRegistry
from app.llm import llm
from app.services.chat_history import get_export_messages
from settings import ANSWER_CONFIDENCE_THRESHOLD
from services.answer_confidence import evaluate_answer_confidence
from services.casual_handler import CasualHandler
from services.complaint_handler import ComplaintHandler
from services.handoff_handler import ManualHandoffHandler
from services.intent_classifier import classify_intent
from services.intent_router import resolve_route
from services.models import RouteType
from services.rag_retriever import sanitize_user_reply
from services.transaction_handler import TransactionHandler

logger = logging.getLogger(__name__)

_RAG_HIT_APPENDIX = """

【内部参考（勿照搬原文，用自己的口语转述）】
{rag_context}
"""

_RAG_MISS_APPENDIX = """

【内部参考】
没查到足够相关的内容。别编商品价格/库存/物流政策；像客服一样说你会去确认，语气随意简短即可。
"""


class UnifiedReplyAgent(TalentBaseAgent):
    """
    统一 Agent 入口：
    1. BERT 意图识别分流
    2. 咨询类：RAG 检索 + 历史对话 + LLM + 答案置信评估
    3. 其他类：对应 handler 处理
    """

    def __init__(self):
        super().__init__("unified_reply")
        self._ensure_tools_registered()
        self._transaction = TransactionHandler()
        self._complaint = ComplaintHandler()
        self._casual = CasualHandler()
        self._handoff = ManualHandoffHandler()

    @staticmethod
    def _ensure_tools_registered():
        from app.agents.tools.rag_tools import (
            EVALUATE_CONFIDENCE_TOOL,
            PRODUCT_CATALOG_TOOL,
            QA_CACHE_LOOKUP_TOOL,
            RAG_TOOL_SCHEMAS,
            REQUEST_HANDOFF_TOOL,
            SEARCH_KNOWLEDGE_BASE_TOOL,
        )
        for name, instance in [
            ("search_knowledge_base", SEARCH_KNOWLEDGE_BASE_TOOL),
            ("evaluate_answer_confidence", EVALUATE_CONFIDENCE_TOOL),
            ("qa_cache_lookup", QA_CACHE_LOOKUP_TOOL),
            ("product_catalog", PRODUCT_CATALOG_TOOL),
            ("request_human_handoff", REQUEST_HANDOFF_TOOL),
        ]:
            if ToolRegistry.get(name) is None:
                schema = ToolRegistry.get_schema(name) or RAG_TOOL_SCHEMAS.get(name)
                if schema and instance:
                    ToolRegistry.register(instance, schema)

    def _load_history(self, context: dict) -> list[dict]:
        if context.get("recent_history"):
            return context["recent_history"]
        contact = (
            context.get("session_key")
            or context.get("contact_username", "")
        )
        limit = context.get("history_limit", 50)
        return get_export_messages(contact, limit)

    def _build_rag_appendix(self, rag_result: dict[str, Any]) -> tuple[str, str]:
        if rag_result.get("hit"):
            return _RAG_HIT_APPENDIX.format(rag_context=rag_result.get("context") or ""), "rag"
        return _RAG_MISS_APPENDIX, "no_hit"

    def build_messages(
        self,
        task: str = "",
        context: Optional[dict] = None,
        *,
        rag_appendix: str = "",
    ) -> list:
        context = context or {}
        system_content = self.system_prompt + rag_appendix

        if context:
            parts = []
            profile = context.get("buyer_profile") or context.get("talent_profile")
            if profile:
                parts.append(f"【买家信息】{profile}")

            prior_summaries = context.get("prior_summaries", [])
            if prior_summaries:
                summary_parts = ["【历史摘要】"]
                l2 = context.get("l2_summary")
                if l2:
                    summary_parts.append(f"- {l2.get('summary', '')}")
                    newer = [s for s in prior_summaries
                             if s.get("start_msg_id", 0) > l2.get("end_msg_id", 0)]
                    for s in newer:
                        summary_parts.append(f"- {s.get('summary', '')}")
                else:
                    for s in prior_summaries[:3]:
                        summary_parts.append(f"- {s.get('summary', '')}")
                parts.append("\n".join(summary_parts))

            _recent_limit = context.get("recent_history_limit", 10)
            history = self._load_history(context)
            if history:
                history_lines = []
                for msg in history[-_recent_limit:]:
                    role = "我方" if msg.get("role") == "assistant" else "对方"
                    history_lines.append(f"{role}：{msg.get('content', '')}")
                parts.append("【历史对话】\n" + "\n".join(history_lines))

            if parts:
                system_content += "\n\n" + "\n\n".join(parts)

        messages = [{"role": "system", "content": system_content}]

        _recent_limit = context.get("recent_history_limit", 10)
        for msg in self._load_history(context)[-_recent_limit:]:
            if msg.get("role") != "tool":
                messages.append(msg)

        user_content = context.get("message") or task
        if user_content:
            if not user_content.startswith("用户新消息") and not user_content.startswith("达人的新消息"):
                user_content = f"买家刚发来：{user_content}\n请直接回复这条消息（口语、简短，像真人客服打字，尽快回）："
            messages.append({"role": "user", "content": user_content})
        return messages

    def _handler_response_to_output(
        self,
        handler_response,
        *,
        tools_used: list | None = None,
        history_count: int = 0,
        rag_hit: bool = False,
    ) -> dict:
        return {
            "result": handler_response.answer,
            "reply_mode": handler_response.reply_mode.value,
            "sources": [
                {
                    "content": s.content,
                    "source": s.source,
                    "score": s.score,
                    "section": s.section,
                    "chunk_type": s.chunk_type,
                    "question": s.question,
                }
                for s in handler_response.sources
            ],
            "tools_used": tools_used or [],
            "history_count": history_count,
            "rag_hit": rag_hit,
            "intent": handler_response.intent,
            "intent_confidence": handler_response.intent_confidence,
            "action": handler_response.action,
            "ticket_id": handler_response.ticket_id,
            "route": handler_response.route.value,
            "answer_confidence": getattr(handler_response, "answer_confidence", 0.0),
            "answer_supported": getattr(handler_response, "answer_supported", False),
            "needs_handoff": getattr(handler_response, "needs_handoff", False),
            "confidence_reason": getattr(handler_response, "confidence_reason", ""),
        }

    def invoke(self, input_data: dict) -> dict:
        task = input_data.get("task", "")
        context = dict(input_data.get("context") or {})
        message = (context.get("message") or task).strip()
        if message.startswith("达人的新消息："):
            message = message.replace("达人的新消息：", "", 1).strip()
        elif message.startswith("用户新消息："):
            message = message.replace("用户新消息：", "", 1).strip()
        context["message"] = message
        user_id = context.get("session_id", "wx_demo_user_001").replace("session_", "")

        if context.get("tool_loop"):
            return self._invoke_flexible(context, message, user_id)

        if not message:
            return {
                "agent": self.name,
                "output": {"result": "", "reply_mode": "no_hit", "sources": []},
                "success": False,
                "error": "消息不能为空",
            }

        intent = classify_intent(message)
        decision = resolve_route(intent)
        history = self._load_history(context)
        tools_used: list[dict] = [{
            "tool": "intent_classifier",
            "input": {"message": message},
            "success": True,
            "result": {
                "intent": intent.category.value,
                "confidence": intent.confidence,
                "action": intent.action,
                "route": decision.route.value,
            },
        }]

        route = decision.route
        if route == RouteType.TRANSACTION:
            resp = self._transaction.handle(user_id, message, intent)
            return {
                "agent": self.name,
                "output": self._handler_response_to_output(
                    resp, tools_used=tools_used, history_count=len(history)
                ),
                "success": True,
                "error": None,
            }

        if route == RouteType.COMPLAINT_HANDOFF:
            resp = self._complaint.handle(user_id, message, intent)
            return {
                "agent": self.name,
                "output": self._handler_response_to_output(
                    resp, tools_used=tools_used, history_count=len(history)
                ),
                "success": True,
                "error": None,
            }

        if route in (RouteType.CASUAL_CHAT, RouteType.FALLBACK):
            resp = self._casual.handle(
                user_id, message, intent,
                history=history,
                recent_history_limit=context.get("recent_history_limit", 10),
                fallback=(route == RouteType.FALLBACK),
            )
            return {
                "agent": self.name,
                "output": self._handler_response_to_output(
                    resp, tools_used=tools_used, history_count=len(history)
                ),
                "success": True,
                "error": None,
            }

        # 咨询类 → 缓存问答（exact → BM25 → MySQL）→ miss 再 RAG
        try:
            from services.qa_cache import hit_to_response, lookup_qa_cache

            cache_hit = lookup_qa_cache(message)
        except Exception:
            logger.exception("QA 缓存查询失败，回退 RAG")
            cache_hit = None

        if cache_hit is not None:
            resp = hit_to_response(user_id, message, cache_hit, intent=intent)
            tools_used.append({
                "tool": "qa_cache",
                "input": {"query": message},
                "success": True,
                "result": {
                    "hit": True,
                    "match_type": cache_hit.match_type,
                    "faq_id": cache_hit.faq_id,
                    "score": cache_hit.score,
                },
            })
            return {
                "agent": self.name,
                "output": self._handler_response_to_output(
                    resp,
                    tools_used=tools_used,
                    history_count=len(history),
                    rag_hit=True,
                ),
                "success": True,
                "error": None,
            }

        rag_result = search_knowledge_base(message)
        tools_used.append({
            "tool": "search_knowledge_base",
            "input": {"query": message},
            "success": True,
            "result": {
                "hit": rag_result["hit"],
                "count": rag_result["count"],
                "threshold": rag_result["threshold"],
                "query_strategy": rag_result.get("query_strategy", "direct"),
                "query_strategy_name": rag_result.get("query_strategy_name", "直接检索"),
            },
        })

        rag_appendix, reply_mode = self._build_rag_appendix(rag_result)
        messages = self.build_messages(task, context, rag_appendix=rag_appendix)

        try:
            response = llm.invoke(
                messages,
                session_id=context.get("session_id"),
                agent_name=self.name,
            )
        except Exception as e:
            logger.exception("统一 Agent 生成失败")
            return {
                "agent": self.name,
                "output": {},
                "success": False,
                "error": str(e),
            }

        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                answer = response
                response = {}

        if isinstance(response, dict) and response.get("choices"):
            choice = response["choices"][0]
            answer = choice.get("message", {}).get("content", "") or ""
        else:
            answer = str(response) if not isinstance(response, dict) else ""

        answer = sanitize_user_reply(answer.strip())
        sources = rag_result.get("sources", [])
        chunks = rag_result.get("chunks") or []

        judge = evaluate_answer_confidence(
            question=message,
            chunks=chunks,
            answer=answer,
        )
        tools_used.append({
            "tool": "answer_confidence",
            "input": {"question": message, "chunk_count": len(chunks)},
            "success": True,
            "result": {
                "confidence": judge.confidence,
                "supported": judge.supported,
                "needs_handoff": judge.needs_handoff,
                "reason": judge.reason,
            },
        })

        if judge.needs_handoff or judge.confidence < ANSWER_CONFIDENCE_THRESHOLD:
            handoff = self._handoff.handle(
                user_id=user_id,
                message=message,
                intent=intent,
                answer_confidence=judge.confidence,
                confidence_reason=judge.reason,
            )
            output = self._handler_response_to_output(
                handoff,
                tools_used=tools_used,
                history_count=len(history),
                rag_hit=rag_result.get("hit", False),
            )
            output["sources"] = sources
            output["query_strategy"] = rag_result.get("query_strategy", "direct")
            output["query_strategy_name"] = rag_result.get("query_strategy_name", "直接检索")
            output["query_strategy_reason"] = rag_result.get("query_strategy_reason", "")
            return {
                "agent": self.name,
                "output": output,
                "success": True,
                "error": None,
            }

        return {
            "agent": self.name,
            "output": {
                "result": answer,
                "reply_mode": reply_mode,
                "sources": sources,
                "tools_used": tools_used,
                "history_count": len(history),
                "rag_hit": rag_result.get("hit", False),
                "intent": intent.category.value,
                "intent_confidence": intent.confidence,
                "action": intent.action,
                "route": RouteType.RAG_AGENT.value,
                "ticket_id": None,
                "answer_confidence": judge.confidence,
                "answer_supported": judge.supported,
                "needs_handoff": False,
                "confidence_reason": judge.reason,
                "query_strategy": rag_result.get("query_strategy", "direct"),
                "query_strategy_name": rag_result.get("query_strategy_name", "直接检索"),
                "query_strategy_reason": rag_result.get("query_strategy_reason", ""),
            },
            "success": True,
            "error": None,
        }

    def _invoke_flexible(self, context: dict, message: str, user_id: str,
                         event_callback: Callable[[str, dict], None] | None = None) -> dict:
        """迭代式工具调用模式：LLM 自主选择工具，循环执行。"""
        history = self._load_history(context)
        tools_used: list[dict] = []

        # 意图预处理：投诉/交易类直接走 handler，不跑 LLM 循环
        intent = classify_intent(message)
        decision = resolve_route(intent)
        tools_used.append({
            "tool": "intent_classifier",
            "input": {"message": message},
            "success": True,
            "result": {
                "intent": intent.category.value,
                "confidence": intent.confidence,
                "action": intent.action,
                "route": decision.route.value,
            },
        })

        if decision.route == RouteType.COMPLAINT_HANDOFF:
            resp = self._complaint.handle(user_id, message, intent)
            return {
                "agent": self.name,
                "output": self._handler_response_to_output(
                    resp, tools_used=tools_used, history_count=len(history)
                ),
                "success": True,
            }

        messages = self.build_messages(f"用户新消息：{message}", context)
        messages[0]["content"] += (
            f"\n\n【意图识别】用户问题分类：{intent.category.value}（置信度 {intent.confidence:.2f}）"
        )
        tools = self.get_tools()

        from app.agents.talent_base import TalentAgentState
        state = TalentAgentState()
        state.task = f"用户新消息：{message}"
        state.messages = messages
        state._last_summary_iteration = 0
        state._progress_summary_index = None

        answer = ""
        sources = []
        chunks = []
        _handoff_requested = False

        for _ in range(self.max_iterations):
            try:
                response = llm.invoke(
                    state.messages,
                    tools=tools,
                    session_id=context.get("session_id"),
                    agent_name=self.name,
                )
            except Exception as e:
                return {"agent": self.name, "output": {}, "success": False, "error": str(e)}

            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except json.JSONDecodeError:
                    answer = response
                    break

            choice = response.get("choices", [{}])[0]
            msg = choice.get("message", {})
            tool_calls = msg.get("tool_calls", [])

            if not tool_calls:
                logger.info(f"[FLEXIBLE] LLM 直接生成答案，未调工具: {message}")
                answer = msg.get("content", "")
                break

            _tool_detail = [{'name': c.get('function',{}).get('name',''),
                             'args': c.get('function',{}).get('arguments','{}')}
                            for c in tool_calls]
            logger.info(f"[FLEXIBLE] LLM 调了 {len(tool_calls)} 个工具: "
                        f"{json.dumps(_tool_detail, ensure_ascii=False)}")
            if event_callback:
                for td in _tool_detail:
                    try:
                        args = json.loads(td['args']) if isinstance(td['args'], str) else td['args']
                    except json.JSONDecodeError:
                        args = {}
                    event_callback("tool_call", {"name": td['name'], "args": args})
            state.messages.append(msg)

            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _run_tool(call):
                func = call.get("function", {})
                tool_name = func.get("name", "")
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                result = self.execute_tool(tool_name, args)
                return call, tool_name, args, result

            with ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
                futures = {pool.submit(_run_tool, call): call for call in tool_calls}
                for future in as_completed(futures):
                    call, tool_name, args, tool_result = future.result()
                    state.iterations += 1

                    tools_used.append({
                        "tool": tool_name,
                        "input": args,
                        "success": tool_result.success,
                        "result": tool_result.result if tool_result.success else tool_result.error,
                    })

                    if tool_name == "search_knowledge_base" and tool_result.success:
                        data = tool_result.result or {}
                        if isinstance(data, dict):
                            sources = data.get("sources", [])
                            chunks = data.get("chunks", [])

                    if tool_name == "request_human_handoff" and tool_result.success:
                        _handoff_requested = True

                    result_content = (
                        json.dumps(tool_result.result, ensure_ascii=False, indent=2)
                        if isinstance(tool_result.result, dict)
                        else str(tool_result.result or "")
                    )
                    if event_callback:
                        event_callback("tool_result", {"name": tool_name, "summary": result_content[:200]})
                    logger.info(f"[FLEXIBLE] 工具 '{tool_name}' 返回: {result_content}")
                    state.messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": result_content if tool_result.success else f"错误: {tool_result.error}",
                    })

                    if not tool_result.success:
                        state.messages.append({
                            "role": "system",
                            "content": f"工具 '{tool_name}' 执行失败，请修正后重试。",
                        })

        answer = answer or ""
        tools_used.append({
            "tool": "flexible_agent",
            "input": {"message": message},
            "success": True,
            "result": {"iterations": state.iterations},
        })

        if _handoff_requested:
            handoff = self._handoff.handle(
                user_id=user_id, message=message, intent=None,
                answer_confidence=0.0,
                confidence_reason="LLM 通过 request_human_handoff 工具主动请求转人工",
            )
            return {
                "agent": self.name,
                "output": self._handler_response_to_output(
                    handoff, tools_used=tools_used, history_count=len(history)
                ),
                "success": True,
            }

        return {
            "agent": self.name,
            "output": {
                "result": answer,
                "reply_mode": "rag" if chunks else "no_hit",
                "sources": sources,
                "tools_used": tools_used,
                "history_count": len(history),
                "rag_hit": bool(chunks),
                "intent": "",
                "intent_confidence": 0.0,
                "action": "flexible_agent",
                "ticket_id": None,
                "route": RouteType.RAG_AGENT.value,
                "answer_confidence": 0.0,
                "answer_supported": False,
                "needs_handoff": False,
                "confidence_reason": "",
            },
            "success": True,
            "error": None,
        }

    def invoke_stream(self, input_data: dict,
                      event_callback: Callable[[str, dict], None] | None = None) -> dict:
        context = dict(input_data.get("context") or {})
        message = (context.get("message") or input_data.get("task", "")).strip()
        if message.startswith("达人的新消息："):
            message = message.replace("达人的新消息：", "", 1).strip()
        elif message.startswith("用户新消息："):
            message = message.replace("用户新消息：", "", 1).strip()
        context["message"] = message
        user_id = context.get("session_id", "wx_demo_user_001").replace("session_", "")
        return self._invoke_flexible(context, message, user_id, event_callback=event_callback)
