from .base_agent import BaseAgent
from typing import List
from openai import OpenAI
from utils.types import CodingResponse, CodebookUpdate


class SocialScientistAgent(BaseAgent):
    """Represents an LLM agent emulating a social scientist."""
    def __init__(self, client: OpenAI, model: str, persona: str, codebook: str):
        self.persona = persona
        self.codebook = codebook
        # system_prompt = f"Persona:\n{persona}\n\nCODEBOOK:\n{codebook}\n\nINSTRUCTION:\n{instruction}"
        system_prompt = f"Persona:\n{persona}\n\nCODEBOOK:\n{codebook}"
        super().__init__(client, model, system_prompt)

    def code_text(self, text: str) -> CodingResponse:
        """Codes a single piece of text based on the codebook and persona."""
        coding_prompt = f"TEXT:\n{text}"
        self.add_user_message(coding_prompt)
        response = self._generate_answer(response_format=CodingResponse)
        self.add_assistant_message(response)
        return response

    def discuss(self, text: str, your_answer: CodingResponse, other_answers: List[CodingResponse]) -> CodingResponse:
        """Participates in a discussion to resolve coding disagreements."""
        other_answers_formatted = "\n\n".join(
            [f"Another agent's response:\n{ans}" for ans in other_answers]
        )
        discussion_prompt = (
            f"TEXT:\n{text}\n\n"
            f"YOUR PREVIOUS ANSWER:\n{your_answer}\n\n"
            f"RESPONSES FROM OTHERS:\n{other_answers_formatted}"
        )
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

    def propose_codebook_update(self, orginal_codebook) -> CodebookUpdate:
        """Proposes changes to the codebook based on recent analysis."""
        update_prompt = (
            f"ORIGINAL CODEBOOK:\n{orginal_codebook}"
        )
        self.add_user_message(update_prompt)
        response = self._generate_answer(response_format=CodebookUpdate)
        self.add_assistant_message(response)
        return response

    def review_mediated_codebook(self, mediator_summary: str) -> CodebookUpdate:
        """Reviews the summary from the Mediator and provides a final opinion.
        
        Returns CodebookUpdate with need_update=False if agreeing, 
        or need_update=True with new_codebook if proposing changes.
        """
        self.add_user_message(mediator_summary)
        response = self._generate_answer(response_format=CodebookUpdate)
        self.add_assistant_message(response)
        return response
        
    def update_codebook(self, new_codebook: str):
        """Updates the agent's internal codebook and resets the context for the next round."""
        self.codebook = new_codebook
        self.system_prompt = f"Persona:\n{self.persona}\n\nCODEBOOK:\n{new_codebook}"
        # Resets the context to start fresh with the new system prompt
        self.reset_context()