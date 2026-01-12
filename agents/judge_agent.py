from typing import List, Any, Set, Union
from utils.types import CodingResponse, CodebookUpdate


def _to_set(value: Union[Set, List, int, Any]) -> Set:
    """Convert a value to a frozenset for comparison."""
    if isinstance(value, (set, frozenset)):
        return frozenset(value)
    if isinstance(value, (list, tuple)):
        return frozenset(value)
    # Single value (int, etc.)
    return frozenset([value])


class JudgeModule:
    """A judge that judges whether other agents are in agreement."""
    
    def __init__(self, task_type: str = "class"):
        """
        Initialize the judge.
        
        Args:
            task_type: "class" for multi-class (direct equality), 
                       "label" for multi-label (set-based comparison)
        """
        self.task_type = task_type
    
    def check_agreement(self, agent_responses: List[CodingResponse]) -> bool:
        """
        Compares agent responses and judges if they are the same.
        Returns True for agreement, False for disagreement.
        
        For multi-class (task_type="class"): uses direct equality
        For multi-label (task_type="label"): uses set-based comparison
        """
        if len(agent_responses) == 0:
            return False
        
        if len(agent_responses) == 1:
            return True
        
        codes = [response.code for response in agent_responses]
        
        if self.task_type == "label":
            # Multi-label: compare as sets (order doesn't matter)
            code_sets = [_to_set(code) for code in codes]
            return len(set(code_sets)) == 1
        else:
            # Multi-class: direct equality (original behavior)
            # Need to convert lists to tuples for hashability if any slip through
            hashable_codes = [tuple(c) if isinstance(c, list) else c for c in codes]
            return len(set(hashable_codes)) == 1

    def check_codebook_agreement(self, agent_responses: List[CodebookUpdate]) -> bool:
        """
        Checks if all agents agree with the mediated codebook.
        Returns True if all agents have need_update=False (all agree).
        """
        if len(agent_responses) == 0:
            return False
        
        return all(not response.need_update for response in agent_responses)
