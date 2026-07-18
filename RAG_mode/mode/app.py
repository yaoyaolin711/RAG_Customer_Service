import html
import os
import sys
from pathlib import Path
from datetime import datetime

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import streamlit as st

from settings import (
    DEEPSEEK_KEY_ENV,
    DEEPSEEK_KEY_FILE,
    INTENT_MODEL_ADAPTER_PATH,
    KB_DOC_DISPLAY_NAME,
    LLM_MODEL_NAME,
    MILVUS_URI,
    RAG_COLLECTION_NAME,
    RAG_RELEVANCE_THRESHOLD,
    get_deepseek_key,
)
from services.models import ReplyMode, RouteType
from services.wechat_handler import WeChatMessageHandler
from vectorstore import check_milvus_connection
from services.models import SessionSnapshot, SessionStatus
from services.session_status import detect_end_trigger
from services.session_store_mysql import get_mysql_session_archive
from services.session_store_redis import get_redis_session_store

st.set_page_config(
    page_title="抖音店铺智能客服 · 模拟台",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💬",
)

CUSTOM_CSS = """
<style>
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1rem; padding-bottom: 1.5rem; max-width: 1180px; }

    .jc-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #2563eb 100%);
        border-radius: 14px; padding: 18px 24px; color: #f8fafc;
        margin-bottom: 1rem; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
    }
    .jc-hero h1 { font-size: 1.45rem; font-weight: 700; margin: 0 0 6px 0; }
    .jc-hero p { margin: 0; color: #cbd5e1; font-size: 0.9rem; }
    .jc-badge {
        display: inline-block; background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.18); border-radius: 999px;
        padding: 3px 10px; font-size: 0.72rem; margin-right: 6px; margin-top: 10px;
    }

    .jc-tag {
        display: inline-flex; align-items: center; gap: 4px;
        border-radius: 8px; padding: 4px 10px; font-size: 0.78rem; font-weight: 700;
    }
    .jc-tag-a { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
    .jc-tag-b { background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }
    .jc-tag-c { background: #e0e7ff; color: #3730a3; border: 1px solid #a5b4fc; }

    .jc-pipeline {
        background: #fff; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 14px 16px; margin-bottom: 12px;
    }
    .jc-pipeline-title {
        font-size: 0.88rem; font-weight: 700; color: #334155;
        margin-bottom: 10px; display: flex; align-items: center; gap: 6px;
    }
    .jc-step {
        display: flex; gap: 10px; align-items: flex-start;
        padding: 8px 0; border-bottom: 1px dashed #e2e8f0;
        font-size: 0.82rem; color: #475569;
    }
    .jc-step:last-child { border-bottom: none; }
    .jc-step-num {
        width: 22px; height: 22px; border-radius: 50%; flex-shrink: 0;
        background: #f1f5f9; color: #64748b; font-size: 0.72rem; font-weight: 700;
        display: flex; align-items: center; justify-content: center;
    }
    .jc-step-active .jc-step-num { background: #2563eb; color: #fff; }
    .jc-step-warn .jc-step-num { background: #f59e0b; color: #fff; }
    .jc-step-success .jc-step-num { background: #10b981; color: #fff; }
    .jc-step-upgrade .jc-step-num { background: #8b5cf6; color: #fff; }

    .jc-mode-pill {
        display: inline-block; border-radius: 999px; padding: 2px 10px;
        font-size: 0.75rem; font-weight: 600;
    }
    .jc-mode-rag { background: #dcfce7; color: #166534; }
    .jc-mode-casual { background: #ffedd5; color: #9a3412; }
    .jc-mode-transaction { background: #dbeafe; color: #1e40af; }
    .jc-mode-handoff { background: #fee2e2; color: #991b1b; }
    .jc-intent-pill {
        display: inline-block; border-radius: 999px; padding: 2px 10px;
        font-size: 0.75rem; font-weight: 600; background: #ede9fe; color: #5b21b6;
    }

    .jc-upgrade-banner {
        background: linear-gradient(90deg, #f5f3ff, #ede9fe);
        border: 1px solid #c4b5fd; border-radius: 10px;
        padding: 10px 12px; margin: 8px 0; font-size: 0.82rem; color: #5b21b6;
    }

    .jc-chat-header {
        background: #f1f5f9; border-bottom: 1px solid #e2e8f0;
        padding: 12px 16px; margin: -16px -16px 14px -16px;
        font-weight: 600; color: #334155; font-size: 0.95rem;
    }
    .jc-msg-user, .jc-msg-bot {
        display: flex; gap: 10px; margin-bottom: 14px; align-items: flex-start;
    }
    .jc-msg-user { flex-direction: row-reverse; }
    .jc-avatar {
        width: 36px; height: 36px; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem; flex-shrink: 0;
    }
    .jc-avatar-user { background: #dbeafe; }
    .jc-avatar-bot { background: linear-gradient(135deg, #fbbf24, #f59e0b); }
    .jc-bubble {
        max-width: 75%; padding: 11px 14px; border-radius: 14px;
        line-height: 1.6; font-size: 0.93rem; white-space: pre-wrap; word-break: break-word;
    }
    .jc-bubble-user { background: #2563eb; color: white; border-bottom-right-radius: 4px; }
    .jc-bubble-bot {
        background: white; color: #1e293b; border: 1px solid #e2e8f0;
        border-bottom-left-radius: 4px; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .jc-stream-cursor {
        display: inline-block; width: 2px; height: 1em;
        background: #2563eb; margin-left: 2px; vertical-align: text-bottom;
        animation: jc-blink 0.75s step-end infinite;
    }
    @keyframes jc-blink { 50% { opacity: 0; } }
    .jc-source-card {
        background: #fff; border: 1px solid #e2e8f0; border-left: 4px solid #2563eb;
        border-radius: 10px; padding: 10px 12px; margin-top: 8px;
        font-size: 0.82rem; color: #475569;
    }
    .jc-source-low { border-left-color: #f59e0b; opacity: 0.85; }
    .jc-source-title { font-weight: 600; color: #1e293b; margin-bottom: 4px; font-size: 0.85rem; }

    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%); }
    .jc-tip {
        background: white; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 10px 12px; margin-bottom: 8px; font-size: 0.86rem; color: #475569;
    }
    .jc-kw { display: inline-block; background: #f1f5f9; border-radius: 6px;
        padding: 2px 7px; margin: 2px; font-size: 0.72rem; color: #64748b; }
    .jc-kw-cat { font-size: 0.75rem; font-weight: 700; color: #475569; margin: 8px 0 4px 0; }
    .jc-kw-cat:first-child { margin-top: 0; }
    .jc-kb-card {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 12px 14px; margin-bottom: 12px; font-size: 0.82rem; color: #475569;
    }
    .jc-kb-title { font-weight: 700; color: #1e293b; margin-bottom: 8px; font-size: 0.88rem; }
    .jc-chunk-tag {
        display: inline-block; border-radius: 4px; padding: 1px 6px;
        font-size: 0.68rem; font-weight: 600; margin-right: 4px;
    }
    .jc-chunk-faq { background: #dcfce7; color: #166534; }
    .jc-chunk-upgrade { background: #fef3c7; color: #92400e; }
    .jc-chunk-reject { background: #fee2e2; color: #991b1b; }
    .jc-chunk-section { background: #e0e7ff; color: #3730a3; }
    .jc-flow-legend {
        display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px;
        font-size: 0.78rem; color: #cbd5e1;
    }
    .jc-flow-item { display: flex; align-items: center; gap: 4px; }
</style>
"""

QUICK_BY_SCENARIO = {
    "📗 咨询类 · RAG": [
        ("这款有什么规格？适合什么人群？", "consult"),
        ("有运费险吗？支持七天无理由吗？", "consult"),
        ("大概多久发货？包邮吗？", "consult"),
    ],
    "📦 交易类": [
        ("我的订单什么时候发货？", "transaction"),
        ("怎么退款？", "transaction"),
    ],
    "⚠️ 投诉类": [
        ("客服态度太差了我要投诉", "complaint"),
        ("这个东西质量有问题", "complaint"),
    ],
    "💬 其他类 · 闲聊": [
        ("今天天气怎么样？", "other"),
        ("你吃饭了吗？", "other"),
    ],
}

WELCOME_MESSAGE = (
    "亲你好，我是店铺智能客服，商品、发货、售后有问题直接问我就行，"
    "我会尽快回复你哈。"
)

CHUNK_TYPE_LABELS = {
    "faq_qa": ("FAQ 问答", "jc-chunk-faq"),
    "product_card": ("产品资料", "jc-chunk-faq"),
    "script_faq": ("商品话术", "jc-chunk-upgrade"),
    "policy_norm": ("客服规范", "jc-chunk-section"),
    "upgrade_keyword": ("关键词片段", "jc-chunk-upgrade"),
    "reject_reason": ("说明片段", "jc-chunk-reject"),
    "section_header": ("章节", "jc-chunk-section"),
}

TAG_LABELS = {}


def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def mode_badge(mode: ReplyMode | None, route: RouteType | None = None) -> str:
    if route == RouteType.TRANSACTION or mode == ReplyMode.TRANSACTION:
        return '<span class="jc-mode-pill jc-mode-transaction">交易查询</span>'
    if route == RouteType.MANUAL_HANDOFF:
        return '<span class="jc-mode-pill jc-mode-handoff">低置信转人工</span>'
    if route == RouteType.COMPLAINT_HANDOFF or mode == ReplyMode.HANDOFF:
        return '<span class="jc-mode-pill jc-mode-handoff">投诉转人工</span>'
    if mode == ReplyMode.CACHE:
        return '<span class="jc-mode-pill jc-mode-rag">缓存问答命中</span>'
    if mode == ReplyMode.RAG:
        return '<span class="jc-mode-pill jc-mode-rag">RAG 知识库回答</span>'
    if mode == ReplyMode.CASUAL:
        return '<span class="jc-mode-pill jc-mode-casual">闲聊回复</span>'
    return ""


def intent_badge(intent: str, confidence: float = 0.0) -> str:
    if not intent:
        return ""
    conf = f" {confidence:.0%}" if confidence else ""
    return f'<span class="jc-intent-pill">{html.escape(intent)}{conf}</span>'


def init_session_state():
    defaults = {
        "chat_history": [],
        "processing_question": None,
        "user_id": "buyer_demo_001",
        "last_response": None,
        "handler_ready": False,
        "session_id": None,
        "session_snapshot": None,
        "chat_input_error": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_session_snapshot() -> SessionSnapshot | None:
    """确保 Redis/MySQL 会话快照已准备好。"""
    if st.session_state.session_snapshot is not None:
        return st.session_state.session_snapshot
    try:
        store = get_redis_session_store()
        snap = store.get_or_create(st.session_state.session_id, st.session_state.user_id, channel="streamlit")
        st.session_state.session_id = snap.session_id
        st.session_state.session_snapshot = snap
        # 首次也写一份到 MySQL（若可用）
        try:
            get_mysql_session_archive().upsert_session(snap)
        except Exception:
            pass
        return snap
    except Exception:
        # Redis 不可用时，仍允许模拟台运行
        return None


def persist_session_snapshot(snap: SessionSnapshot) -> None:
    """短期写 Redis + 长期 upsert MySQL。"""
    try:
        get_redis_session_store().update(snap)
    except Exception:
        pass
    try:
        get_mysql_session_archive().upsert_session(snap)
    except Exception:
        pass


def chunk_type_badge(chunk_type: str) -> str:
    label, css = CHUNK_TYPE_LABELS.get(chunk_type, (chunk_type or "文本", "jc-chunk-section"))
    return f'<span class="jc-chunk-tag {css}">{html.escape(label)}</span>'


def get_kb_stats() -> dict:
    try:
        from vectorstore import get_collection_chunk_types, get_collection_count
        from settings import RAG_COLLECTION_NAME
        count = get_collection_count(RAG_COLLECTION_NAME)
        types = get_collection_chunk_types(RAG_COLLECTION_NAME)
        return {"count": count, "types": types, "ok": True}
    except Exception:
        return {"count": 0, "types": {}, "ok": False}


def render_hero():
    kb = get_kb_stats()
    kb_line = f"{KB_DOC_DISPLAY_NAME} · {kb['count']} chunks" if kb["ok"] else f"{KB_DOC_DISPLAY_NAME} · 未入库"
    st.markdown(
        f"""
        <div class="jc-hero">
            <h1>抖音店铺智能客服 · 模拟台</h1>
            <p>BERT 意图识别 → 分流 → RAG / 交易 / 投诉 / 闲聊 · 目标尽快回复（约 15 秒内）</p>
            <span class="jc-badge">知识库 {kb_line}</span>
            <span class="jc-badge">BGE-M3 Dense</span>
            <span class="jc-badge">{RAG_COLLECTION_NAME}</span>
            <span class="jc-badge">MacBERT 意图模型</span>
            <div class="jc-flow-legend">
                <span class="jc-flow-item">📗 咨询类 → RAG</span>
                <span class="jc-flow-item">📦 交易类 → 查单</span>
                <span class="jc-flow-item">⚠️ 投诉类 → 转人工</span>
                <span class="jc-flow-item">💬 其他类 → 闲聊</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_message(role: str, content: str):
    safe = html.escape(content).replace("\n", "<br>")
    is_user = role == "user"
    cls = "jc-msg-user" if is_user else "jc-msg-bot"
    av_cls = "jc-avatar-user" if is_user else "jc-avatar-bot"
    bub_cls = "jc-bubble-user" if is_user else "jc-bubble-bot"
    icon = "🧑" if is_user else "✨"
    st.markdown(
        f'<div class="{cls}"><div class="jc-avatar {av_cls}">{icon}</div>'
        f'<div class="jc-bubble {bub_cls}">{safe}</div></div>',
        unsafe_allow_html=True,
    )


def render_streaming_bot(placeholder, content: str, *, show_cursor: bool = True):
    safe = html.escape(content).replace("\n", "<br>")
    cursor = '<span class="jc-stream-cursor"></span>' if show_cursor else ""
    placeholder.markdown(
        f'<div class="jc-msg-bot"><div class="jc-avatar jc-avatar-bot">✨</div>'
        f'<div class="jc-bubble jc-bubble-bot">{safe}{cursor}</div></div>',
        unsafe_allow_html=True,
    )


def _build_final_summary(user_text: str, answer_text: str) -> str:
    return (user_text[:80] + " / " + answer_text[:120]).strip(" /")


def _apply_session_end(snap: SessionSnapshot, decision, user_text: str, answer_text: str) -> None:
    snap.status = decision.status
    snap.end_reason = decision.reason
    if decision.status in (SessionStatus.RESOLVED, SessionStatus.CLOSED):
        snap.ended_at = _now_iso()
        snap.final_summary = _build_final_summary(user_text, answer_text)
        try:
            get_mysql_session_archive().finalize_session(snap)
        except Exception:
            pass
    elif decision.status == SessionStatus.HANDOFF_PENDING:
        # 交易/投诉等转人工仅更新状态，不写 final_summary，允许后续继续追加 turns
        pass


def append_assistant_message(response):
    meta = {
        "route": response.route,
        "reply_mode": response.reply_mode,
        "sources_count": len(response.sources),
        "intent": response.intent,
        "intent_confidence": response.intent_confidence,
        "action": response.action,
        "ticket_id": response.ticket_id,
        "answer_confidence": response.answer_confidence,
        "answer_supported": response.answer_supported,
        "needs_handoff": response.needs_handoff,
        "confidence_reason": response.confidence_reason,
    }
    st.session_state.chat_history.append(
        {"role": "assistant", "content": response.answer, "meta": meta}
    )
    st.session_state.last_response = response

    # 更新会话快照（若可用）
    snap = ensure_session_snapshot()
    if snap is None:
        return
    user_text = st.session_state.get("processing_question_last", "")
    snap.append_turn(
        user_text,
        response.answer,
        intent=response.intent or "",
        intent_confidence=float(response.intent_confidence or 0.0),
        route=response.route.value if response.route else "",
        reply_mode=response.reply_mode.value if response.reply_mode else "",
        answer_confidence=float(getattr(response, "answer_confidence", 0.0) or 0.0),
        needs_handoff=bool(getattr(response, "needs_handoff", False)),
    )
    snap.last_intent = response.intent or snap.last_intent
    snap.last_route = response.route.value if response.route else snap.last_route
    snap.last_reply_mode = response.reply_mode.value if response.reply_mode else snap.last_reply_mode
    snap.last_answer_confidence = float(getattr(response, "answer_confidence", 0.0) or 0.0)
    snap.last_needs_handoff = bool(getattr(response, "needs_handoff", False))
    snap.touch()

    decision = detect_end_trigger(user_text=user_text, response=response)
    if decision is not None:
        _apply_session_end(snap, decision, user_text, response.answer)

    persist_session_snapshot(snap)


def is_blank_message(text: str | None) -> bool:
    """空字符串、纯空白（含空格/换行/全角空格）均视为无效消息。"""
    if text is None:
        return True
    return not str(text).strip()


def queue_user_message(question: str):
    q = (question or "").strip()
    if is_blank_message(q):
        st.session_state.chat_input_error = "消息不能为空，请输入内容后再发送。"
        return
    st.session_state.chat_input_error = ""
    st.session_state["_pending_clear_chat_input"] = True
    # 首次发消息时才创建会话
    snap = ensure_session_snapshot()
    if snap is not None:
        snap.touch()
        persist_session_snapshot(snap)
    st.session_state.chat_history.append({"role": "user", "content": q})
    st.session_state.processing_question = q
    st.session_state.processing_question_last = q
    st.rerun()


def process_pending_message(question: str):
    """用户消息已在 chat_history 中，此处流式生成助手回复。"""
    st.session_state.processing_question = None
    if is_blank_message(question):
        return
    try:
        session = st.session_state.wechat_handler.prepare_message_stream(
            user_id=st.session_state.user_id,
            message=question,
        )
        if session.instant:
            append_assistant_message(session.instant)
            st.rerun()
            return

        placeholder = st.empty()
        parts: list[str] = []
        for chunk in session.text_stream:
            parts.append(chunk)
            render_streaming_bot(placeholder, "".join(parts))

        response = session.finalize("".join(parts))
        # 若质检判定转人工，立即用转接话术覆盖流式草稿，避免用户看到答非所问全文
        if response.needs_handoff or response.reply_mode.value == "handoff":
            render_streaming_bot(placeholder, response.answer, show_cursor=False)
        append_assistant_message(response)
        st.rerun()
    except Exception as e:
        err_msg = f"抱歉，处理出错了：{e}"
        st.session_state.chat_history.append(
            {"role": "assistant", "content": err_msg, "meta": {}}
        )
        snap = ensure_session_snapshot()
        if snap is not None:
            user_text = st.session_state.get("processing_question_last", question)
            snap.append_turn(user_text, err_msg)
            snap.touch()
            persist_session_snapshot(snap)
        st.rerun()


def render_retrieved_chunks(chunks: list, show_filtered: bool = False):
    if not chunks:
        st.info("未召回任何文档片段（score 均低于阈值或未检索到）")
        return
    for i, c in enumerate(chunks, 1):
        low = c.score < RAG_RELEVANCE_THRESHOLD
        card_cls = "jc-source-card jc-source-low" if low and show_filtered else "jc-source-card"
        body = html.escape(c.content).replace("\n", "<br>")
        type_badge = chunk_type_badge(c.chunk_type) if c.chunk_type else ""
        section_line = f" · {html.escape(c.section)}" if c.section else ""
        st.markdown(
            f"""
            <div class="{card_cls}">
                <div class="jc-source-title">
                    {type_badge}
                    📄 片段 {i} · {html.escape(c.source)}{section_line}
                    · score={c.score:.2f}
                    {" · ⚠️ 低于阈值" if low else " · ✅ 有效命中"}
                </div>
                {body}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_kb_overview():
    kb = get_kb_stats()
    type_lines = ""
    if kb["types"]:
        for t, n in kb["types"].items():
            label, css = CHUNK_TYPE_LABELS.get(t, (t, "jc-chunk-section"))
            type_lines += f'<span class="jc-chunk-tag {css}">{html.escape(label)} ×{n}</span> '
    st.markdown(
        f"""
        <div class="jc-kb-card">
            <div class="jc-kb-title">📚 当前知识库</div>
            文档：<b>{html.escape(KB_DOC_DISPLAY_NAME)}</b><br>
            向量库：<b>{RAG_COLLECTION_NAME}</b> · 共 <b>{kb["count"]}</b> 条 chunk<br>
            切分类型：{type_lines or "暂无数据，请运行知识库入库脚本"}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_panel(response):
    """右侧：展示本轮完整处理链路。"""
    if response is None:
        render_kb_overview()
        st.markdown(
            """
            <div class="jc-pipeline">
                <div class="jc-pipeline-title">⚙️ 处理链路</div>
                <div style="color:#94a3b8;font-size:0.82rem">
                    发送消息后，此处展示：BERT 意图识别 → 路由分流 → 回复生成
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    route_label = {
        RouteType.RAG_AGENT: "咨询类 → RAG Agent",
        RouteType.TRANSACTION: "交易类 → 订单查询",
        RouteType.COMPLAINT_HANDOFF: "投诉类 → 建工单转人工",
        RouteType.MANUAL_HANDOFF: "答案低置信 → 转人工",
        RouteType.CASUAL_CHAT: "其他类 → LLM 闲聊",
        RouteType.FALLBACK: "低置信 → 兜底闲聊",
        RouteType.UNSUPPORTED: "未支持",
    }.get(response.route, str(response.route))

    intent_html = intent_badge(response.intent, response.intent_confidence)
    prob_lines = ""
    if response.intent_probabilities:
        prob_lines = " · ".join(
            f"{html.escape(k)}:{v:.0%}" for k, v in response.intent_probabilities.items()
        )

    step2_cls = "jc-step-success" if response.sources else "jc-step-warn"
    if response.route == RouteType.RAG_AGENT:
        strategy_name = getattr(response, "query_strategy_name", "") or "直接检索"
        strategy_reason = getattr(response, "query_strategy_reason", "") or ""
        step2_text = (
            f"策略={strategy_name} · 召回 {len(response.sources)} 条有效片段"
            if response.sources
            else f"策略={strategy_name} · 无有效命中 → 切换闲聊模式"
        )
        if strategy_reason:
            step2_text += f"<br><span style=\"font-size:0.75rem;color:#64748b\">{html.escape(strategy_reason)}</span>"
        step2_block = f"""
            <div class="jc-step {step2_cls}">
                <div class="jc-step-num">2</div>
                <div><b>RAG 向量检索</b><br>{step2_text}</div>
            </div>
        """
    else:
        step2_block = f"""
            <div class="jc-step jc-step-active">
                <div class="jc-step-num">2</div>
                <div><b>分流处理</b><br>{html.escape(route_label)}</div>
            </div>
        """

    step3_text = {
        ReplyMode.RAG: "基于知识库内容生成回答",
        ReplyMode.CACHE: "多级缓存命中（Redis/BM25/MySQL）",
        ReplyMode.CASUAL: "LLM 闲聊回复",
        ReplyMode.TRANSACTION: "交易类 Mock 查询回复",
        ReplyMode.HANDOFF: "投诉登记 + 转人工话术",
    }.get(response.reply_mode, "")
    if response.route == RouteType.MANUAL_HANDOFF:
        step3_text = "答案置信度低于阈值，转人工兜底"

    judge_status = "证据支持" if response.answer_supported else "待人工确认"
    if response.route == RouteType.MANUAL_HANDOFF or response.needs_handoff:
        judge_status = "低置信已转人工"
    judge_reason = response.confidence_reason or "未提供评估原因"

    ticket_html = ""
    if response.ticket_id:
        ticket_html = f"""
        <div class="jc-step jc-step-warn">
            <div class="jc-step-num">🎫</div>
            <div><b>投诉工单</b><br>工单号 #{response.ticket_id}（待对接抖音客服）</div>
        </div>
        """

    st.markdown(
        f"""
        <div class="jc-pipeline">
            <div class="jc-pipeline-title">⚙️ 本轮处理链路</div>
            <div class="jc-step jc-step-active">
                <div class="jc-step-num">1</div>
                <div><b>意图识别</b><br>{intent_html} · action={html.escape(response.action or "")}<br>
                <span style="font-size:0.75rem;color:#64748b">{prob_lines}</span></div>
            </div>
            {step2_block}
            <div class="jc-step jc-step-active">
                <div class="jc-step-num">3</div>
                <div><b>回复模式</b><br>{mode_badge(response.reply_mode, response.route)} · {html.escape(step3_text)}</div>
            </div>
            <div class="jc-step {'jc-step-warn' if response.needs_handoff else 'jc-step-active'}">
                <div class="jc-step-num">4</div>
                <div><b>答案评估</b><br>置信度 {response.answer_confidence:.0%} · {html.escape(judge_status)}<br>
                <span style="font-size:0.75rem;color:#64748b">{html.escape(judge_reason)}</span></div>
            </div>
            {ticket_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 会话状态展示
    snap = st.session_state.get("session_snapshot")
    if snap is not None:
        st.caption(
            f"Session: `{snap.session_id}` · status={snap.status.value}"
            + (f" · end_reason={snap.end_reason}" if snap.end_reason else "")
            + (f" · turns={len(snap.turns)}" if snap.turns else "")
        )

    if response.sources:
        with st.expander(f"📚 有效召回片段（{len(response.sources)} 条）", expanded=True):
            render_retrieved_chunks(response.sources)
    elif response.route == RouteType.RAG_AGENT:
        st.warning("知识库未命中：已进入闲聊模式，不会编造产品/政策细节。")


def render_sidebar():
    with st.sidebar:
        st.markdown("### 👤 模拟店铺买家")
        st.text_input("用户 ID", key="user_id")

        st.markdown("---")
        st.markdown("### 💡 快捷提问")
        st.caption("按意图类型测试：咨询 / 交易 / 投诉 / 其他")
        for scenario, questions in QUICK_BY_SCENARIO.items():
            st.markdown(f"**{scenario}**")
            for q, kind in questions:
                if st.button(q, use_container_width=True, key=f"quick_{kind}_{q}"):
                    queue_user_message(q)

        st.markdown("---")
        st.markdown("### ⚙️ 系统信息")
        api_ok = "已配置" if (os.getenv(DEEPSEEK_KEY_ENV) or Path(DEEPSEEK_KEY_FILE).is_file()) else "未检测到"
        st.markdown(
            f"""
            <div class="jc-tip">🔑 API Key<br><b>{api_ok}</b></div>
            <div class="jc-tip">🤖 大模型<br><b>{LLM_MODEL_NAME}</b></div>
            <div class="jc-tip">🧠 Embedding<br><b>BGE-M3 本地</b></div>
            <div class="jc-tip">🎯 意图模型<br><b>MacBERT+LoRA</b></div>
            <div class="jc-tip">🗄️ 向量库<br><b>{RAG_COLLECTION_NAME}</b><br>Milvus · {MILVUS_URI}</div>
            <div class="jc-tip">📄 知识库<br><b>{KB_DOC_DISPLAY_NAME}</b></div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🗑️ 清空对话", use_container_width=True):
            # 结束会话并归档
            snap = ensure_session_snapshot()
            if snap is not None:
                snap.status = SessionStatus.CLOSED
                snap.end_reason = "manual_clear"
                snap.ended_at = _now_iso()
                if snap.turns:
                    last = snap.turns[-1]
                    snap.final_summary = _build_final_summary(
                        last.get("user_message", ""),
                        last.get("assistant_message", ""),
                    )
                try:
                    get_redis_session_store().mark_closed(snap, status=SessionStatus.CLOSED, reason="manual_clear")
                except Exception:
                    pass
                try:
                    get_mysql_session_archive().finalize_session(snap)
                except Exception:
                    pass
            # 清空 UI 对话（新会话会在下次发送时创建）
            st.session_state.chat_history = []
            st.session_state.last_response = None
            st.session_state.processing_question = None
            st.session_state.session_id = None
            st.session_state.session_snapshot = None
            st.rerun()


def render_chat_column():
    """消息区与输入区分开，避免 form 被 CSS overflow 遮挡。"""
    if st.session_state.pop("_pending_clear_chat_input", False):
        st.session_state.chat_question_input = ""

    with st.container(border=True):
        st.markdown(
            f'<div class="jc-chat-header">💬 对话 · '
            f'{html.escape(st.session_state.user_id)}</div>',
            unsafe_allow_html=True,
        )

        if not st.session_state.chat_history:
            render_message("assistant", WELCOME_MESSAGE)
        else:
            for msg in st.session_state.chat_history:
                render_message(msg["role"], msg["content"])
                meta = msg.get("meta")
                if meta and msg["role"] == "assistant":
                    badge = mode_badge(meta.get("reply_mode"), meta.get("route"))
                    intent_line = intent_badge(
                        meta.get("intent", ""),
                        meta.get("intent_confidence", 0.0),
                    )
                    conf = meta.get("answer_confidence", 0.0)
                    judge_line = ""
                    if conf:
                        judge_label = "已转人工" if meta.get("needs_handoff") else "答案置信"
                        judge_line = (
                            f'<span class="jc-intent-pill">{html.escape(judge_label)} '
                            f'{conf:.0%}</span>'
                        )
                    lines = [b for b in (intent_line, badge) if b]
                    if judge_line:
                        lines.append(judge_line)
                    if lines:
                        st.markdown(
                            f'<div style="margin:-8px 0 12px 46px">{" ".join(lines)}</div>',
                            unsafe_allow_html=True,
                        )

        if st.session_state.processing_question:
            process_pending_message(st.session_state.processing_question)

    disabled = bool(st.session_state.processing_question)
    # 空提交时不要清空输入，避免看起来像“已经发出”
    with st.form("chat_form", clear_on_submit=False):
        c1, c2 = st.columns([6, 1])
        with c1:
            question = st.text_input(
                "chat_question",
                placeholder="模拟买家消息，如：这款多久发货？有运费险吗？",
                label_visibility="collapsed",
                disabled=disabled,
                key="chat_question_input",
            )
        with c2:
            submitted = st.form_submit_button(
                "发送",
                type="primary",
                use_container_width=True,
                disabled=disabled,
            )
    if st.session_state.get("chat_input_error"):
        st.warning(st.session_state.chat_input_error)
    return submitted, question


def main():
    inject_css()
    init_session_state()
    render_sidebar()

    try:
        if not st.session_state.handler_ready:
            with st.spinner("正在连接 Milvus 并初始化意图路由..."):
                check_milvus_connection()
                get_deepseek_key()
                st.session_state.wechat_handler = WeChatMessageHandler()
                st.session_state.handler_ready = True
    except Exception as e:
        render_hero()
        err = str(e)
        hints = []
        if "DEEPSEEK_KEY" in err or ("环境变量" in err and isinstance(e, ValueError)):
            hints.append(f'请设置 $env:DEEPSEEK_KEY="你的Key"，或写入 {DEEPSEEK_KEY_FILE}')
        if "Milvus" in err or "milvus" in err.lower():
            hints.append(f"请确认 Milvus 路径存在：{MILVUS_URI}")
            hints.append("请从仓库根目录运行：python scripts/ingest_bd_docx.py")
        if not hints:
            hints = [f"设置 {DEEPSEEK_KEY_ENV}", f"确认 {MILVUS_URI}", "运行 ingest_bd_docx.py"]
        st.error(f"初始化失败：{err}\n\n" + "\n".join(f"- {h}" for h in hints))
        st.stop()

    render_hero()

    col_chat, col_pipeline = st.columns([3, 2])
    with col_chat:
        submitted, question = render_chat_column()
    with col_pipeline:
        st.markdown("#### 📊 实时链路监控")
        render_pipeline_panel(st.session_state.last_response)

    if submitted:
        if is_blank_message(question):
            st.session_state.chat_input_error = "消息不能为空，空格或空内容不能发送。"
            st.warning(st.session_state.chat_input_error)
        else:
            queue_user_message(question)


main()

if __name__ == "__main__":
    import subprocess

    from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

    from settings import STREAMLIT_PORT

    if get_script_run_ctx() is None:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                __file__,
                "--server.port",
                str(STREAMLIT_PORT),
                "--server.headless",
                "true",
            ]
        )
