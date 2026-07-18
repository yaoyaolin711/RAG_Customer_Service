"""Legacy：原达人回复 Agent，非店铺买家主链路，仅兼容保留。"""

from app.agents.talent_base import TalentBaseAgent


class TalentReplyAgent(TalentBaseAgent):
    def __init__(self):
        super().__init__("talent_reply")
        self.tools = []
