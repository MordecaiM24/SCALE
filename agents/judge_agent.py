from typing import List, Any
from utils.types import CodingResponse, CodebookUpdate

class JudgeAgent:
    """A judge that judges whether other agents are in agreement."""
    def check_agreement(self, agent_responses: List[CodingResponse]) -> bool:
        """
        Compares agent responses and judges if they are the same.
        Returns True for agreement, False for disagreement.
        """
        if len(agent_responses) == 0:
            return False
        
        if len(agent_responses) == 1:
            return True
        
        codes = [response.code for response in agent_responses]
        
        return len(set(codes)) == 1

    def check_codebook_agreement(self, agent_responses: List[CodebookUpdate]) -> bool:
        """
        Checks if all agents agree with the mediated codebook.
        Returns True if all agents have need_update=False (all agree).
        """
        if len(agent_responses) == 0:
            return False
        
        return all(not response.need_update for response in agent_responses)
