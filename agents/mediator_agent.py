from .base_agent import BaseAgent
from typing import List
from openai import OpenAI
from utils.types import CodebookUpdate

class MediatorAgent(BaseAgent):
    """An agent that mediates codebook update discussions."""
    def __init__(self, client: OpenAI, model: str, prompt_template: str):
        super().__init__(client, model, prompt_template)

    def mediate(self, proposals: List[CodebookUpdate]) -> str:
        """Summarizes proposals and asks for agreement."""
        proposals_formatted = "\n\n".join(
            [f"Agent {i+1}'s proposal:\n{proposal.new_codebook if proposal.new_codebook else proposal.reasoning}" 
                for i, proposal in enumerate(proposals)]
        )
        self.reset_context()
        mediate_prompt = f"Here are the codebook update proposals from other social scientists:\n\n{proposals_formatted}"
        self.add_user_message(mediate_prompt)

        summary = self._generate_answer()
        self.add_assistant_message(summary)

        self.reset_context()
        return summary