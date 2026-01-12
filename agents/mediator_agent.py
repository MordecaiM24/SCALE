from .base_agent import BaseAgent
from typing import List
from openai import OpenAI
from utils.types import CodebookUpdate

class MediatorAgent(BaseAgent):
    """
    An agent that mediates codebook update discussions.
    
    Enhanced to consider disagreement context when synthesizing proposals,
    ensuring the mediated codebook addresses actual coding challenges.
    """
    def __init__(self, client: OpenAI, model: str, prompt_template: str):
        super().__init__(client, model, prompt_template)

    def mediate(
        self, 
        proposals: List[CodebookUpdate],
        disagreement_context: str = "",
        evolution_history: str = ""
    ) -> str:
        """
        Synthesizes codebook update proposals into a unified codebook.
        
        Enhanced to include:
        - Context about disagreements that prompted the proposals
        - History of codebook evolution to maintain consistency
        
        Args:
            proposals: List of codebook update proposals from agents
            disagreement_context: Summary of disagreements from recent coding
            evolution_history: History of previous codebook changes
        """
        proposals_formatted = "\n\n".join(
            [f"Agent {i+1}'s proposal:\nReasoning: {proposal.reasoning}\n"
             f"Proposed Changes: {proposal.new_codebook if proposal.new_codebook else 'No changes'}" 
             for i, proposal in enumerate(proposals)]
        )
        
        self.reset_context()
        
        mediate_prompt_parts = []
        
        # Include disagreement context (so mediator understands WHAT problems to solve)
        if disagreement_context:
            mediate_prompt_parts.append(
                f"DISAGREEMENTS TO ADDRESS:\n{disagreement_context}\n\n"
                f"The codebook updates should help resolve these coding challenges."
            )
        
        # Include evolution history (so mediator maintains consistency)
        if evolution_history:
            mediate_prompt_parts.append(
                f"PREVIOUS CODEBOOK CHANGES:\n{evolution_history}\n\n"
                f"Maintain consistency with previous decisions."
            )
        
        mediate_prompt_parts.append(
            f"PROPOSALS FROM CODERS:\n\n{proposals_formatted}\n\n"
            f"Synthesize these proposals into a unified codebook that:\n"
            f"1. Addresses the disagreements encountered\n"
            f"2. Incorporates valid points from each proposal\n"
            f"3. Maintains clarity and consistency\n"
            f"4. Avoids contradicting previous evolution decisions"
        )
        
        mediate_prompt = "\n\n".join(mediate_prompt_parts)
        self.add_user_message(mediate_prompt)

        summary = self._generate_answer()
        self.add_assistant_message(summary)

        self.reset_context()
        return summary
    
    def extract_change_rationale(self, original: str, updated: str, proposals: List[CodebookUpdate]) -> str:
        """
        Extracts a summary rationale for why the codebook was changed.
        This is stored in evolution history for future reference.
        """
        if original == updated:
            return "No changes made - codebook deemed adequate for current examples."
        
        # Summarize the key reasons from proposals
        reasons = [p.reasoning for p in proposals if p.need_update]
        if reasons:
            return f"Changed to address: {'; '.join(reasons[:3])}"
        return "Updated based on team discussion to improve clarity."