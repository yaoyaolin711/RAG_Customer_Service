"""统一 API 请求/响应模型（OpenAPI 文档用）。"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ChatMode(str, Enum):
    """保留兼容；当前所有取值均走 UnifiedReplyAgent，不触发分流。"""

    RAG = "rag"
    TALENT = "talent"
    AUTO = "auto"


class ApiState(BaseModel):
    code: int = Field(0, description="0=成功，非0=失败")
    message: str = Field("ok", description="状态描述")


class SourceItem(BaseModel):
    content: str = Field(..., description="召回片段内容")
    source: str = Field(..., description="来源文档")
    score: float = Field(..., description="相关度分数 0~1")
    section: str = Field("", description="所属章节")
    chunk_type: str = Field("", description="片段类型：faq_qa 等")
    question: str = Field("", description="FAQ 对应问题（如有）")


class ToolUsedItem(BaseModel):
    tool: str = Field(..., description="工具名称")
    input: dict[str, Any] = Field(default_factory=dict, description="工具入参")
    success: bool = Field(True, description="是否调用成功")
    result: dict[str, Any] = Field(default_factory=dict, description="工具返回摘要")


class MessageEnvelope(BaseModel):
    direction: str = Field(..., description="receive 或 send")
    message: str = Field(..., description="消息正文")
    intent: str | None = Field(None, description="simulate 接口专用")


class TagUpgrade(BaseModel):
    """已废弃：店铺买家不再升标签；保留模型以免旧客户端解析失败。"""

    from_tag: str = Field(..., alias="from", description="原标签（废弃）")
    to_tag: str = Field(..., alias="to", description="新标签（废弃）")
    keywords: list[str] = Field(default_factory=list, description="触发关键词（废弃）")
    applied: bool = Field(False, description="是否已写入数据库（废弃）")

    model_config = {"populate_by_name": True}


class ChatData(BaseModel):
    count: int = Field(0, description="sources 条数")
    mode: str = Field("agent", description="固定 agent")
    route: str = Field("unified_agent", description="路由类型")
    answer: str = Field(..., description="生成的回复文本")
    reply_mode: str = Field("", description="rag / no_hit / casual / transaction / handoff")
    rag_hit: bool = Field(False, description="知识库是否有效命中")
    user_id: str = Field("", description="用户/会话 ID")
    buyer_name: str | None = Field(None, description="买家展示名")
    talent_id: str | None = Field(None, description="兼容旧字段，等同 buyer_name")
    session_key: str | None = Field(None, description="会话键（读历史）")
    history_count: int = Field(0, description="读到的历史对话条数")
    sources: list[SourceItem] = Field(default_factory=list, description="RAG 召回片段")
    tools_used: list[ToolUsedItem] = Field(default_factory=list, description="工具调用记录")
    intent: str = Field("", description="意图分类：咨询类/交易类/投诉类/其他类")
    intent_confidence: float = Field(0.0, description="意图置信度 0~1")
    action: str = Field("", description="路由动作")
    ticket_id: int | None = Field(None, description="投诉工单 ID（如有）")
    tag_upgrade: TagUpgrade | None = Field(
        None, description="已废弃，店铺买家固定为 null"
    )
    answer_confidence: float = Field(0.0, description="答案置信度 0~1")
    answer_supported: bool = Field(False, description="答案是否被召回资料支持")
    needs_handoff: bool = Field(False, description="是否建议转人工")
    confidence_reason: str = Field("", description="置信评估原因")
    query_strategy: str = Field("", description="查询策略代码 direct/rewrite/multi_query/keyword_boost")
    query_strategy_name: str = Field("", description="查询策略中文名")
    query_strategy_reason: str = Field("", description="策略选择原因")
    session_id: str = Field("", description="Redis 会话 ID")
    session_status: str = Field("", description="会话状态 open/resolved/handoff_pending/closed")
    session_end_reason: str = Field("", description="会话结束/转人工原因")
    received: MessageEnvelope = Field(..., description="收到的用户消息")
    reply: MessageEnvelope = Field(..., description="发送的回复")
    success: bool = Field(True, description="Agent 是否成功")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="买家消息内容")
    mode: ChatMode = Field(
        ChatMode.AUTO,
        description="保留兼容，当前均走 UnifiedReplyAgent，传 rag/talent/auto 效果相同",
    )
    user_id: str = Field("buyer_demo_001", description="买家/会话 ID")
    user_tag: str = Field(
        "B",
        description="已废弃，保留兼容，不再参与路由（店铺买家一视同仁）",
    )
    buyer_name: str = Field("", description="买家展示名，写入用户画像")
    talent_id: str = Field("", description="兼容旧字段，等同 buyer_name")
    session_key: str = Field("", description="会话键，用于读取历史对话")
    contact_username: str = Field("", description="兼容旧字段，等同 session_key")
    tool_loop: bool = Field(False, description="启用灵活工具调用模式（迭代式 LLM 选工具）")

    @model_validator(mode="after")
    def _merge_compat_aliases(self):
        if not self.buyer_name and self.talent_id:
            self.buyer_name = self.talent_id
        elif self.buyer_name and not self.talent_id:
            self.talent_id = self.buyer_name
        if not self.session_key and self.contact_username:
            self.session_key = self.contact_username
        elif self.session_key and not self.contact_username:
            self.contact_username = self.session_key
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "这款多久发货？有运费险吗？",
                    "user_id": "buyer_demo_001",
                    "session_key": "dy_session_001",
                },
                {
                    "message": "你好，想问下尺码怎么选",
                    "buyer_name": "买家小李",
                },
            ]
        }
    }


class ChatResponse(BaseModel):
    data: ChatData
    state: ApiState


class HealthData(BaseModel):
    count: int = 0
    status: str = Field("ok", description="ok / degraded")
    milvus: str = Field("unknown", description="connected 或 error: ...")
    redis: str = Field("unknown", description="connected 或 error: ...")
    llm: str = "configured"
    agent: str = Field("unified_reply", description="当前 Agent 名称")


class HealthResponse(BaseModel):
    data: HealthData
    state: ApiState


class MetaData(BaseModel):
    count: int = 0
    version: str = "3.1.0"
    agent: str = "unified_reply"
    llm_model: str = ""
    collection: str = ""
    intent_model: str = ""
    relevance_threshold: float = 0.45
    answer_confidence_threshold: float = 0.75
    rerank_enabled: bool = True
    kb_doc: str = ""
    modes: list[str] = Field(default_factory=lambda: ["agent", "rag", "talent", "auto"])
    routes: list[str] = Field(
        default_factory=lambda: [
            "rag_agent",
            "transaction",
            "complaint_handoff",
            "manual_handoff",
            "casual_chat",
            "fallback",
        ]
    )
    intents: list[str] = Field(
        default_factory=lambda: ["咨询类", "交易类", "投诉类", "其他类"]
    )
    reply_modes: list[str] = Field(
        default_factory=lambda: ["rag", "no_hit", "casual", "transaction", "handoff"]
    )
    tools: list[str] = Field(
        default_factory=lambda: [
            "intent_classifier",
            "search_knowledge_base",
            "answer_confidence",
        ]
    )
    flow: str = "intent_classify → route → rag/handler/llm → confidence/handoff → session"


class MetaResponse(BaseModel):
    data: MetaData
    state: ApiState


class TalentSimulateRequest(BaseModel):
    """兼容旧路径请求体；语义为买家消息模拟。"""

    message: str = Field(..., min_length=1, description="买家消息")
    session_key: str = Field("", description="会话键")
    contact_username: str = Field("", description="兼容旧字段，等同 session_key")

    @model_validator(mode="after")
    def _merge_session_key(self):
        if not self.session_key and self.contact_username:
            self.session_key = self.contact_username
        elif self.session_key and not self.contact_username:
            self.contact_username = self.session_key
        return self


class TalentSimulateData(BaseModel):
    count: int = 1
    mode: str = "agent"
    received: MessageEnvelope
    reply: MessageEnvelope
    sources: list[SourceItem] = Field(default_factory=list)
    reply_mode: str = ""
    history_count: int = 0


class TalentSimulateResponse(BaseModel):
    data: TalentSimulateData
    state: ApiState


class UnifiedResponse(BaseModel):
    data: dict[str, Any]
    state: ApiState


# ============================================================
# 历史对话读写
# ============================================================


class HistoryMessageItem(BaseModel):
    sender_username: str = Field(..., description="发送方用户名")
    content: str = Field(..., description="消息内容")
    datetime: str = Field(..., description="消息时间")


class HistoryReadData(BaseModel):
    count: int = Field(0, description="消息条数")
    contact_username: str = Field("", description="联系人/会话键")
    messages: list[HistoryMessageItem] = Field(default_factory=list, description="历史消息列表")


class HistoryReadResponse(BaseModel):
    data: HistoryReadData
    state: ApiState


class HistoryWriteRequest(BaseModel):
    contact_username: str = Field(..., min_length=1, description="联系人/会话键")
    self_username: str = Field(..., min_length=1, description="我方用户名")
    incoming_message: str = Field("", description="对方发送的消息")
    outgoing_message: str = Field("", description="我方的回复")


class HistoryWriteData(BaseModel):
    count: int = 1
    success: bool = Field(False, description="是否写入成功")
    message_id: int | None = Field(None, description="最后写入的消息 ID")


class HistoryWriteResponse(BaseModel):
    data: HistoryWriteData
    state: ApiState
