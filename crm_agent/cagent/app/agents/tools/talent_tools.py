from typing import Optional, Dict, Any
from app.agents.tools.base import BaseTool, ToolOutput, tool
from pydantic import BaseModel, Field


class IntentRecognitionInput(BaseModel):
    message: str = Field(..., description="达人回复的消息内容")
    context: Optional[str] = Field(None, description="建联上下文信息，如达人名称、历史对话等")


@tool(name="talent_intent_recognition", description="识别达人回复的意图，用于自动分流处理")
class TalentIntentRecognitionTool(BaseTool):
    name = "talent_intent_recognition"
    description = "识别达人回复的意图（感兴趣/拒绝/询价/已合作等），返回意图分类和后续建议"
    input_model = IntentRecognitionInput

    INTENT_ACTIONS = {
        "interested": {"label": "感兴趣", "action": "推进合作，发送详细合作方案"},
        "rejected": {"label": "已拒绝", "action": "标记为拒绝，移入公海或结束跟进"},
        "asking_price": {"label": "询问价格", "action": "发送报价和合作模式说明"},
        "already_cooperating": {"label": "已合作", "action": "检查合作状态，更新为合作中"},
        "not_interested": {"label": "不感兴趣", "action": "礼貌结束，感谢回复"},
        "asking_details": {"label": "询问详情", "action": "提供更多信息，解答疑问"},
        "pending": {"label": "待确认", "action": "标记待跟进，需要人工介入"},
        "unclear": {"label": "意图不明", "action": "请求澄清或转人工处理"},
    }

    def execute(self, input_data: Dict) -> ToolOutput:
        message = input_data.get("message", "")
        context = input_data.get("context", "")

        if not message:
            return ToolOutput(success=False, error="消息内容不能为空")

        from app.llm import llm

        prompt = f"""分析以下达人回复消息的意图，只返回JSON。

回复消息：{message}
上下文：{context if context else "无"}

意图类别（选一个）：
- interested: 感兴趣，愿意合作
- rejected: 明确拒绝
- asking_price: 询问价格或合作费用
- already_cooperating: 已在与其他品牌合作
- not_interested: 不感兴趣
- asking_details: 询问更多详情
- pending: 需要进一步确认
- unclear: 意图不明确

JSON格式：
{{"intent": "类别", "confidence": 0.0~1.0, "reasoning": "判断理由", "suggestion": "建议操作"}}
"""

        try:
            response = llm.invoke([
                {"role": "system", "content": "你是一个专业的达人营销专家，擅长分析达人回复意图。"},
                {"role": "user", "content": prompt}
            ])

            content = llm.extract_content(response)

            import re, json
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            result = json.loads(content)

            intent = result.get("intent", "unclear")
            intent_info = self.INTENT_ACTIONS.get(intent, self.INTENT_ACTIONS["unclear"])

            return ToolOutput(success=True, result={
                "intent": intent,
                "label": intent_info["label"],
                "confidence": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", ""),
                "suggestion": result.get("suggestion", ""),
                "action": intent_info["action"],
            })
        except Exception as e:
            return ToolOutput(success=False, error=f"意图识别失败: {e}")


class MessageGeneratorInput(BaseModel):
    talent_profile: str = Field(..., description="达人画像/信息")
    goal: str = Field("寻求合作带货", description="建联目标")


@tool(name="talent_message_generator", description="根据达人画像生成个性化建联消息")
class TalentMessageGeneratorTool(BaseTool):
    name = "talent_message_generator"
    description = "根据达人画像和建联目标，AI生成个性化建联话术"
    input_model = MessageGeneratorInput

    def execute(self, input_data: Dict) -> ToolOutput:
        profile = input_data.get("talent_profile", "")
        goal = input_data.get("goal", "寻求合作带货")

        from app.llm import llm

        prompt = f"""为以下达人生成个性化建联消息。

达人信息：{profile}
建联目标：{goal}

要求：
- 不超过150字
- 突出合作价值和利益点
- 语气友好专业
- 提及对方近期内容或数据
- 明确行动号召

输出格式：
消息内容（直接可发送）
---
消息亮点说明"""

        try:
            response = llm.invoke([
                {"role": "system", "content": "你是一个专业的达人营销文案专家。"},
                {"role": "user", "content": prompt}
            ])
            content = llm.extract_content(response)
            return ToolOutput(success=True, result={"message": content, "generated_at": "2026-06-04"})
        except Exception as e:
            return ToolOutput(success=False, error=f"消息生成失败: {e}")


class FollowupInput(BaseModel):
    situation: str = Field(..., description="当前情况，如无回复/价格异议/时间不合适等")
    context: Optional[str] = Field(None, description="背景信息")


@tool(name="talent_followup_generator", description="生成多轮跟进话术")
class TalentFollowupGeneratorTool(BaseTool):
    name = "talent_followup_generator"
    description = "根据达人回复或无回复情况，生成多轮针对性跟进话术"
    input_model = FollowupInput

    def execute(self, input_data: Dict) -> ToolOutput:
        situation = input_data.get("situation", "")
        context = input_data.get("context", "")

        from app.llm import llm

        prompt = f"""为以下情况生成3轮跟进话术。

情况：{situation}
背景：{context if context else "首次建联后无回复"}

每轮间隔2-3天，语气递进：
第1轮：温和提醒，重申价值
第2轮：增加紧迫感，引导决策
第3轮：最终确认，礼貌收尾

格式：
### 第1轮
消息内容...

### 第2轮
消息内容...

### 第3轮
消息内容..."""

        try:
            response = llm.invoke([
                {"role": "system", "content": "你是一个专业的达人营销专员。"},
                {"role": "user", "content": prompt}
            ])
            content = llm.extract_content(response)
            return ToolOutput(success=True, result={
                "followups": content,
                "round_count": 3,
                "generated_at": "2026-06-04",
            })
        except Exception as e:
            return ToolOutput(success=False, error=f"跟进消息生成失败: {e}")


class ProfileAnalyzerInput(BaseModel):
    profile: str = Field(..., description="达人画像信息")


@tool(name="talent_profile_analyzer", description="分析达人画像评估合作价值")
class TalentProfileAnalyzerTool(BaseTool):
    name = "talent_profile_analyzer"
    description = "分析达人信息，评估合作价值和最佳触达策略"
    input_model = ProfileAnalyzerInput

    def execute(self, input_data: Dict) -> ToolOutput:
        profile = input_data.get("profile", "")

        from app.llm import llm

        prompt = f"""分析以下达人画像，输出结构化合作评估。

达人信息：{profile}

请输出JSON：
{{"positioning": "达人定位", "value_assessment": "合作价值评估", "suggested_model": "建议合作模式", "key_points": ["话术要点"], "expected_roi": "预期ROI范围", "risks": ["风险提示"]}}"""

        try:
            response = llm.invoke([
                {"role": "system", "content": "你是一个专业的达人营销分析师。"},
                {"role": "user", "content": prompt}
            ])
            content = llm.extract_content(response)

            import re, json
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                result = {"analysis": content}

            return ToolOutput(success=True, result=result)
        except Exception as e:
            return ToolOutput(success=False, error=f"画像分析失败: {e}")


class AutoReplyInput(BaseModel):
    talent_message: str = Field(..., description="达人发送的消息内容")
    talent_name: Optional[str] = Field(None, description="达人昵称")
    product_info: Optional[str] = Field(None, description="我们产品的信息，如卖点、价格、合作模式等")
    context: Optional[str] = Field(None, description="历史对话或额外上下文")


@tool(name="talent_auto_reply", description="自动回复达人消息，根据达人回复生成自然友好的回复文本（不含分析过程，直接输出可发送的消息）")
class TalentAutoReplyTool(BaseTool):
    name = "talent_auto_reply"
    description = "根据达人回复的消息内容，自动生成自然友好的回复文本，像真人一样直接回复，不展示分析过程"
    input_model = AutoReplyInput

    def execute(self, input_data: Dict) -> ToolOutput:
        talent_message = input_data.get("talent_message", "")
        talent_name = input_data.get("talent_name", "")
        product_info = input_data.get("product_info", "我们的产品")
        context = input_data.get("context", "")

        if not talent_message:
            return ToolOutput(success=False, error="消息内容不能为空")

        from app.llm import llm

        prompt = f"""你是品牌方的达人运营，正在和抖音达人私下聊天沟通合作。
达人给你发了消息，请像真人一样自然地回复对方。

达人昵称：{talent_name if talent_name else "未知"}
产品信息：{product_info}
历史对话：{context if context else "首次沟通"}

达人消息：{talent_message}

要求：
1. 语气轻松友好，像朋友聊天一样自然
2. 不要用太正式/官方的语言，接地气一点
3. 回复不要太长，2-3句话搞定
4. 直接输出回复文本，不要加任何前缀或说明
5. 不要说"亲爱的~"等过度亲密的称呼
6. 根据达人消息的内容判断对方意图，自然地回应即可"""

        try:
            response = llm.invoke([
                {"role": "system", "content": "你是一个性格开朗、说话自然的抖音品牌方的运营人员，正在和达人们私信沟通合作。你回复的风格很轻松，像朋友聊天一样。"},
                {"role": "user", "content": prompt}
            ])

            content = llm.extract_content(response)
            return ToolOutput(success=True, result={"reply": content})
        except Exception as e:
            return ToolOutput(success=False, error=f"自动回复生成失败: {e}")


TOOL_SCHEMAS = {}

for cls in [TalentIntentRecognitionTool, TalentMessageGeneratorTool,
            TalentFollowupGeneratorTool, TalentProfileAnalyzerTool,
            TalentAutoReplyTool]:
    tool_instance = cls()
    schema = {
        "type": "function",
        "function": {
            "name": tool_instance.name,
            "description": tool_instance.description,
            "parameters": tool_instance.input_model.schema() if hasattr(tool_instance.input_model, 'schema') else {},
        }
    }
    TOOL_SCHEMAS[tool_instance.name] = schema


TALENT_INTENT_RECOGNITION_TOOL = TalentIntentRecognitionTool()
TALENT_MESSAGE_GENERATOR_TOOL = TalentMessageGeneratorTool()
TALENT_FOLLOWUP_GENERATOR_TOOL = TalentFollowupGeneratorTool()
TALENT_PROFILE_ANALYZER_TOOL = TalentProfileAnalyzerTool()
TALENT_AUTO_REPLY_TOOL = TalentAutoReplyTool()
