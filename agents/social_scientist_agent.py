from .base_agent import BaseAgent
from typing import List, Optional
from openai import OpenAI
from utils.types import CodingResponse, CodebookUpdate


class SocialScientistAgent(BaseAgent):
    """
    Represents an LLM agent emulating a social scientist coder.
    
    Enhanced to support:
    - Session memory for cross-phase learning
    - Disagreement context injection during discussions
    - Codebook evolution history awareness
    """
    def __init__(self, client: OpenAI, model: str, persona: str, codebook: str):
        self.persona = persona
        self.codebook = codebook
        self.agent_id: Optional[int] = None  # Set by simulation for identification
        system_prompt = f"Persona:\n{persona}\n\nCODEBOOK:\n{codebook}"
        super().__init__(client, model, system_prompt)

    def code_text(self, text: str, discussion_context: str = "") -> CodingResponse:
        """
        Codes a single piece of text based on the codebook and persona.
        
        Args:
            text: The text to code
            discussion_context: Optional context from past discussions/disagreements
                              to inform coding decisions (mimics human learning)
        """
        coding_prompt = f"TEXT:\n{text}"
        
        # Inject past discussion context if available (human coders remember past decisions)
        if discussion_context:
            coding_prompt = f"{discussion_context}\n\n{coding_prompt}"
        
        self.add_user_message(coding_prompt)
        response = self._generate_answer(response_format=CodingResponse)
        self.add_assistant_message(response)
        return response

    def discuss(
        self, 
        text: str, 
        your_answer: CodingResponse, 
        other_answers: List[CodingResponse],
        disagreement_context: str = "",
        round_num: int = 1
    ) -> CodingResponse:
        """
        Participates in a discussion to resolve coding disagreements.
        
        Enhanced to include:
        - Full reasoning from other coders (not just their codes)
        - Context from previous disagreements on similar texts
        - Round-aware prompting for multi-round discussions
        
        Args:
            text: The text being discussed
            your_answer: This agent's previous answer
            other_answers: Other agents' responses with full reasoning
            disagreement_context: Context from past disagreements (patterns, resolutions)
            round_num: Current discussion round number
        """
        # Format other answers with emphasis on their reasoning
        other_answers_formatted = "\n\n".join(
            [f"Coder {i+1}'s response:\nCode: {ans.code}\nReasoning: {ans.reasoning}" 
             for i, ans in enumerate(other_answers)]
        )
        
        # Build discussion prompt with richer context
        discussion_prompt_parts = []
        
        # Add disagreement context if available (helps coders learn from patterns)
        if disagreement_context:
            discussion_prompt_parts.append(
                f"RELEVANT CONTEXT FROM PREVIOUS DISCUSSIONS:\n{disagreement_context}"
            )
        
        # Add round-specific guidance (in real CA, later rounds focus on narrowing differences)
        if round_num > 1:
            discussion_prompt_parts.append(
                f"[Discussion Round {round_num}] Focus on the specific points of disagreement "
                f"and try to reach consensus. Consider whether the differences stem from "
                f"codebook interpretation or the text itself."
            )
        
        discussion_prompt_parts.extend([
            f"TEXT:\n{text}",
            f"YOUR PREVIOUS ANSWER:\nCode: {your_answer.code}\nReasoning: {your_answer.reasoning}",
            f"OTHER CODERS' RESPONSES:\n{other_answers_formatted}",
            "Please carefully consider the reasoning provided by other coders. "
            "If you change your answer, explain why their reasoning convinced you. "
            "If you maintain your answer, explain why your interpretation is more appropriate."
        ])
        
        discussion_prompt = "\n\n".join(discussion_prompt_parts)
        self.add_user_message(discussion_prompt)
        response = self._generate_answer(response_format=CodingResponse)
        self.add_assistant_message(response)
        return response

    def inject_intervention(self, intervention_prompt: str) -> None:
        """
        Injects human intervention guidance into the agent's context.
        
        The intervention is freeform text that will influence the agent's next response.
        After injection, call the appropriate phase method (discuss, review_mediated_codebook, etc.)
        to get a properly typed response.
        """
        self.add_user_message(intervention_prompt)

    def propose_codebook_update(
        self, 
        original_codebook: str,
        disagreement_context: str = "",
        evolution_history: str = ""
    ) -> CodebookUpdate:
        """
        Proposes changes to the codebook based on recent analysis.
        
        Enhanced to include:
        - Context about disagreements that prompted the evolution
        - History of previous codebook changes and their rationale
        
        Args:
            original_codebook: Current codebook text
            disagreement_context: Summary of disagreements from recent coding
            evolution_history: History of how the codebook has evolved
        """
        update_prompt_parts = []
        
        # Include disagreement context (the WHY behind potential changes)
        if disagreement_context:
            update_prompt_parts.append(
                f"DISAGREEMENTS ENCOUNTERED DURING CODING:\n{disagreement_context}\n\n"
                f"Consider whether these disagreements indicate ambiguity or gaps in the codebook "
                f"that should be addressed."
            )
        
        # Include evolution history (avoid re-introducing removed rules, build on improvements)
        if evolution_history:
            update_prompt_parts.append(
                f"CODEBOOK EVOLUTION HISTORY:\n{evolution_history}\n\n"
                f"Consider this history when proposing changes to maintain consistency "
                f"and avoid reverting previously resolved issues."
            )
        
        update_prompt_parts.append(f"ORIGINAL CODEBOOK:\n{original_codebook}")
        
        update_prompt = "\n\n".join(update_prompt_parts)
        self.add_user_message(update_prompt)
        response = self._generate_answer(response_format=CodebookUpdate)
        self.add_assistant_message(response)
        return response

    def review_mediated_codebook(
        self, 
        mediator_summary: str,
        disagreement_context: str = ""
    ) -> CodebookUpdate:
        """
        Reviews the summary from the Mediator and provides a final opinion.
        
        Enhanced to consider whether the mediated codebook addresses 
        the disagreements that prompted the evolution.
        
        Returns CodebookUpdate with need_update=False if agreeing, 
        or need_update=True with new_codebook if proposing changes.
        """
        review_prompt_parts = []
        
        if disagreement_context:
            review_prompt_parts.append(
                f"CONTEXT - Disagreements that prompted this codebook review:\n{disagreement_context}"
            )
        
        review_prompt_parts.append(mediator_summary)
        review_prompt_parts.append(
            "Evaluate whether this proposed codebook adequately addresses the coding "
            "challenges encountered. Accept if it improves clarity, reject with specific "
            "suggestions if issues remain."
        )
        
        review_prompt = "\n\n".join(review_prompt_parts)
        self.add_user_message(review_prompt)
        response = self._generate_answer(response_format=CodebookUpdate)
        self.add_assistant_message(response)
        return response
        
    def update_codebook(self, new_codebook: str, preserve_memory: bool = True):
        """
        Updates the agent's internal codebook and resets the context.
        
        Args:
            new_codebook: The new codebook text
            preserve_memory: If True, session memory (past learnings) is preserved
        """
        self.codebook = new_codebook
        self.base_system_prompt = f"Persona:\n{self.persona}\n\nCODEBOOK:\n{new_codebook}"
        # Reset context but optionally preserve session memory (learnings from past discussions)
        self.reset_context(preserve_memory=preserve_memory)
    
    def extract_key_learning(self, text: str, resolution: CodingResponse, was_changed: bool) -> str:
        """
        Extracts a key learning point from a resolved disagreement.
        This can be stored in session memory to inform future coding.
        
        Args:
            text: The text that was discussed
            resolution: The final coding decision
            was_changed: Whether this agent changed their answer
        """
        if was_changed:
            return f"Changed to Code {resolution.code} because: {resolution.reasoning[:100]}"
        else:
            return f"Maintained Code {resolution.code}: {resolution.reasoning[:100]}"