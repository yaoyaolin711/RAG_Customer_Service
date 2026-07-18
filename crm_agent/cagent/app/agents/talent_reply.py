from app.agents.talent_base import TalentBaseAgent


class TalentReplyAgent(TalentBaseAgent):
    def __init__(self):
        super().__init__("talent_reply")
        self.tools = []
