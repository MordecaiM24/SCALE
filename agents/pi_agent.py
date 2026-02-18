from typing import List

from openai import OpenAI

from .base_agent import BaseAgent
from utils.types import CalibrationResult, CodebookUpdate, CodingResponse


class PIAgent(BaseAgent):
    """Principal Investigator agent for adjudication and strategic oversight."""

    def __init__(self, client: OpenAI, model: str, persona: str, codebook: str):
        system_prompt = f"Persona:\n{persona}\n\nCODEBOOK:\n{codebook}"
        super().__init__(client, model, system_prompt)
        self.persona = persona
        self.codebook = codebook

    def adjudicate(
        self,
        text: str,
        agent_responses: List[CodingResponse],
        discussion_history: List[List[CodingResponse]],
    ) -> CodingResponse:
        """Final call on unresolved disagreements after max discussion rounds."""
        final_responses_formatted = "\n\n".join(
            [
                (
                    f"Agent {i+1} Final Response:\n"
                    f"Code: {response.code}\n"
                    f"Reasoning: {response.reasoning}\n"
                    f"Confidence: {response.confidence}"
                )
                for i, response in enumerate(agent_responses)
            ]
        )

        rounds_summary = []
        for round_index, round_responses in enumerate(discussion_history):
            round_name = "Initial coding" if round_index == 0 else f"Discussion round {round_index}"
            round_summary = "\n".join(
                [
                    (
                        f"- Agent {i+1}: code={response.code}, "
                        f"confidence={response.confidence}, reasoning={response.reasoning}"
                    )
                    for i, response in enumerate(round_responses)
                ]
            )
            rounds_summary.append(f"{round_name}:\n{round_summary}")

        rounds_summary_text = "\n\n".join(rounds_summary)

        prompt = (
            "The coding team did not reach agreement after all discussion rounds. "
            "Please make a final adjudication decision.\n\n"
            f"TEXT:\n{text}\n\n"
            f"FINAL AGENT RESPONSES:\n{final_responses_formatted}\n\n"
            "POSITION EVOLUTION ACROSS ROUNDS:\n"
            f"{rounds_summary_text}"
        )

        self.reset_context()
        self.add_user_message(prompt)
        response = self._generate_answer(response_format=CodingResponse)
        self.add_assistant_message(response)
        self.reset_context()
        return response

    def review_calibration(self, calibration_results: List[CalibrationResult]) -> str:
        """Review pilot coding results and provide coaching guidance."""
        results_formatted = "\n\n".join(
            [
                (
                    f"Text ID: {result.text_id}\n"
                    f"Agent codes: {result.agent_codes}\n"
                    f"Ground truth: {result.ground_truth}\n"
                    f"Agreement with ground truth: {result.agreement}\n"
                    f"Notes: {result.notes}"
                )
                for result in calibration_results
            ]
        )

        prompt = (
            "Please review the calibration results below. Identify recurring error patterns, "
            "explain likely causes, and provide actionable coaching guidance for the coding team. "
            "Highlight codebook categories that need additional attention.\n\n"
            f"CALIBRATION RESULTS:\n{results_formatted}"
        )

        self.reset_context()
        self.add_user_message(prompt)
        response = self._generate_answer()
        self.add_assistant_message(response)
        self.reset_context()
        return response

    def guide_codebook_evolution(
        self,
        current_codebook: str,
        proposals: List[CodebookUpdate],
        memo_summary: str = "",
    ) -> str:
        """Strategic direction for codebook changes."""
        proposals_formatted = "\n\n".join(
            [
                (
                    f"Agent {i+1} Proposal:\n"
                    f"Need update: {proposal.need_update}\n"
                    f"Reasoning: {proposal.reasoning}\n"
                    f"Proposed codebook: {proposal.new_codebook if proposal.new_codebook else '[No new codebook text provided]'}"
                )
                for i, proposal in enumerate(proposals)
            ]
        )

        prompt = (
            "Provide strategic PI guidance on whether and how the codebook should evolve. "
            "Indicate which proposed changes are warranted, which should be rejected, and why.\n\n"
            f"CURRENT CODEBOOK:\n{current_codebook}\n\n"
            f"CODEBOOK UPDATE PROPOSALS:\n{proposals_formatted}\n\n"
            f"MEMO SUMMARY:\n{memo_summary if memo_summary else '[No memo summary available]'}"
        )

        self.reset_context()
        self.add_user_message(prompt)
        response = self._generate_answer()
        self.add_assistant_message(response)
        self.reset_context()
        return response

    def update_codebook(self, new_codebook: str):
        self.codebook = new_codebook
        self.system_prompt = f"Persona:\n{self.persona}\n\nCODEBOOK:\n{new_codebook}"
        self.reset_context()
