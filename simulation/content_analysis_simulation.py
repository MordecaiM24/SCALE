import pandas as pd
import os
from typing import List, Dict, Any, Optional

from agents.social_scientist_agent import SocialScientistAgent
from agents.judge_agent import JudgeAgent
from agents.mediator_agent import MediatorAgent
from agents.human_expert import HumanExpert
from utils.logger import Logger
from utils.config_loader import load_codebook
from openai import OpenAI

from utils.types import CodingResponse
from evaluator import Evaluator, load_ground_truth

class ContentAnalysisSimulation:
    def __init__(self, config: Dict[str, Any], logger: Logger, run_id: int = 0):
        self.config = config
        self.logger = logger
        self.run_id = run_id
        
        # Simulation parameters
        self.num_agents = config['settings']['agents']
        self.discussion_rounds = config['settings']['rounds']
        self.chunk_size = config['settings']['chunk_size']
        self.model = config['settings']['model']

        # Load data
        data_file = os.path.join(config['paths']['data_path'], config['dataset_name'], 'data.xlsx')
        df = pd.read_excel(data_file)
        self.text_chunks = [df['Text'][i:i + self.chunk_size] for i in range(0, len(df), self.chunk_size)]

        # Coder Simulation
        self.logger.log("********** Bot Annotation **********\n")

        client_kwargs = {"api_key": config['api_key']}
        if 'base_url' in config.get('settings', {}):
            client_kwargs["base_url"] = config['settings']['base_url']
        self.client = OpenAI(**client_kwargs)

        self.codebook = load_codebook(config['dataset_name'], config['paths']['data_path'])
        
        self.scientists = self._create_scientists()
        self.judge = JudgeAgent()
        self.mediator = MediatorAgent(self.client, self.model, config['prompt']['mediator'])
        self.logger.log(f"Initialized {self.num_agents} Social Scientist Agents For {self.config['dataset_name']} Task.\n")

        # Intervention settings
        self.intervention_settings = config['settings'].get('intervention', {})
        self.intervention_enabled = self.intervention_settings.get('enabled', False)
        if self.intervention_enabled:
            self.intervention_scope = self.intervention_settings.get('scope', 'targeted')
            self.intervention_authority = self.intervention_settings.get('authority', 'collaborative')
            self.human_expert = HumanExpert(system_prompt=config['prompt'][self.intervention_authority])
        else:
            self.human_expert = None
        
        # Initialize evaluator with ground truth
        self._init_evaluator(df)
        

    def _create_scientists(self) -> List[SocialScientistAgent]:
        """Initializes the SocialScientistAgent instances."""
        scientists = []
        personas = list(self.config['persona'].values())
        for i in range(self.num_agents):
            agent = SocialScientistAgent(
                client=self.client,
                model=self.model,
                persona=personas[i],
                codebook=self.codebook,
            )
            scientists.append(agent)
        return scientists
    
    def _init_evaluator(self, df):
        """Initialize the evaluator with ground truth from the dataset."""
        self.ground_truth = load_ground_truth(df)
        self.evaluator = Evaluator(self.ground_truth)


    def _human_intervention(self, phase: str) -> bool:
        """Inject human intervention guidance into all agents' context.
        
        The intervention is freeform text. After injection, the calling code should
        re-run the appropriate phase method to get properly typed responses.
        
        Returns True if intervention was provided, False otherwise.
        """
        intervention_prompt = self.human_expert.intervene()
        if intervention_prompt:
            self.logger.log(f"!!! {self.intervention_authority.upper()} Intervention on {phase} Activated !!!\n")
            self.logger.log(f"Intervention Prompt:\n{intervention_prompt}\n")
            for agent in self.scientists:
                agent.inject_intervention(intervention_prompt)
            return True
        else:
            self.logger.log(f"No intervention provided for {phase}. Continuing without changes.\n")
            return False


    def run(self) -> Dict[str, Any]:
        """Runs the entire content analysis simulation loop."""
        full_log = []
        all_coding_results = {}
        all_final_answers = {}
        all_coding_agreements = {}
        all_final_agreements = {}
        
        for i, chunk in enumerate(self.text_chunks):
            self.logger.log(f"===== Processing Chunk {i+1}/{len(self.text_chunks)} =====\n")
            
            # Bot Annotation
            coding_results, coding_agreements = self._run_coding_phase(chunk)
            all_coding_results.update(coding_results)
            all_coding_agreements.update(coding_agreements)
            
            # Agent Discussion
            discussion_results, final_answers, final_agreements = self._run_discussion_phase(chunk, coding_results, coding_agreements)
            all_final_answers.update(final_answers)
            all_final_agreements.update(final_agreements)
            
            # Codebook Evolution
            self._run_codebook_evolution_phase()
            
            # Log results for this chunk
            chunk_log = {
                "chunk_id": i,
                "coding_phase": {"results": coding_results, "agreements": coding_agreements},
                "discussion_phase": {"history": discussion_results, "results": final_answers, "agreements": final_agreements},
                "final_codebook": self.codebook
            }
            full_log.append(chunk_log)
            self.logger.save_json(chunk_log, f'chunk_{i}_results.json')
        
        # Evaluate results
        self.logger.log("\n" + "=" * 50)
        self.logger.log("EVALUATION RESULTS")
        self.logger.log("=" * 50)
        merged_post_discussion = {}
        merged_post_discussion_agreements = {}
        for text_id, coding_responses in all_coding_results.items():
            if text_id in all_final_answers:
                merged_post_discussion[text_id] = all_final_answers[text_id]
                merged_post_discussion_agreements[text_id] = all_final_agreements.get(text_id, False)
            else:
                merged_post_discussion[text_id] = coding_responses
                merged_post_discussion_agreements[text_id] = all_coding_agreements.get(text_id, False)
        
        eval_result = self.evaluator.evaluate_run(
            all_coding_results,
            merged_post_discussion if merged_post_discussion else None,
            coding_agreements=all_coding_agreements if all_coding_agreements else None,
            discussion_agreements=merged_post_discussion_agreements if merged_post_discussion_agreements else None,
            log_fn=self.logger.log
        )
        
        self.logger.save_json(full_log, 'full_simulation_log.json')
        self.logger.save_json(eval_result, 'evaluation_results.json')
        self.logger.log("\n===== Simulation Complete =====\n")
        
        return eval_result

    def _run_coding_phase(self, chunk: pd.Series):
        self.logger.log("********** Bot Annotation **********\n")

    
        coding_results: Dict[str, List[CodingResponse]] = {}
        coding_agreements: Dict[str, bool] = {}

        self.logger.log("--- Agents Coding Texts ---\n")
        for i, text in enumerate(chunk):
            text_id = f"Text-{chunk.index[i]+1}"
            self.logger.log(f"--- Coding {text_id} ---\n{text}\n")
            
            for agent in self.scientists:
                agent.reset_context()
                agent.add_user_message(self.config['prompt']['coding'])
            responses = [agent.code_text(text) for agent in self.scientists]
            for j, response in enumerate(responses):
                self.logger.log(f"Agent {j+1}: {response}\n")

            coding_results[text_id] = responses
            agreement = self.judge.check_agreement(responses)
            coding_agreements[text_id] = agreement
            self.logger.log(f"Judge's Verdict: {'Agreement' if agreement else 'Disagreement'}\n")

        return coding_results, coding_agreements

    def _run_discussion_phase(self, chunk: pd.Series, coding_results: Dict, coding_agreements: Dict):
        self.logger.log("********** Agent Discussion **********\n")

        discussion_results: Dict[str, List[List[CodingResponse]]] = {}
        final_answers: Dict[str, List[CodingResponse]] = {}
        final_agreements: Dict[str, bool] = {}

        self.logger.log("\n--- Agents Discussing Disagreements ---\n")
        for i, text in enumerate(chunk):
            text_id = f"Text-{chunk.index[i]+1}"
            if not coding_agreements[text_id]:
                self.logger.log(f"\n--- Discussing {text_id} ---\n")
                for agent in self.scientists:
                    agent.reset_context()
                    agent.add_user_message(self.config['prompt']['discussion'])
                
                discussion_history: List[List[CodingResponse]] = [coding_results[text_id]]

                agreement = False
                for round_num in range(self.discussion_rounds):
                    self.logger.log(f"<Discussion Round {round_num + 1}>\n")
                    current_answers = discussion_history[-1]

                    next_round_answers = [agent.discuss(text, current_answers[j], current_answers[:j] + current_answers[j+1:])
                                          for j, agent in enumerate(self.scientists)]
                    for j, answer in enumerate(next_round_answers):
                        self.logger.log(f"Agent {j+1}: {answer}\n")
                    
                    # *** HUMAN INTERVENTION POINT (DISCUSSION) ***
                    if self.intervention_enabled:
                        if self._human_intervention(phase='discussion'):
                            # Re-run discuss with intervention context injected
                            next_round_answers = [agent.discuss(text, current_answers[j], current_answers[:j] + current_answers[j+1:])
                                                  for j, agent in enumerate(self.scientists)]
                            for j, answer in enumerate(next_round_answers):
                                self.logger.log(f"Agent {j+1} (Post-Intervention): {answer}\n")
                    # *** END INTERVENTION ***

                    discussion_history.append(next_round_answers)
                    agreement = self.judge.check_agreement(next_round_answers)
                    self.logger.log(f"Judge's Verdict: {'Agreement' if agreement else 'Disagreement'}\n")
                    if agreement:
                        self.logger.log(f"--- Consensus Reached for {text_id} ---\n")
                        break
                
                final_agreements[text_id] = agreement
                discussion_results[text_id] = discussion_history
                final_answers[text_id] = discussion_history[-1]

        return discussion_results, final_answers, final_agreements

    def _run_codebook_evolution_phase(self):
        self.logger.log("********** Codebook Evolution **********\n")
        
        # Initial proposal step
        update_prompt = (
            f"{self.config['prompt']['update']}\n\n"
            f"Here is an example of updating CODEBOOK:\n"
            f"Example ORIGINAL CODEBOOK:\n{self.config['codebook_example']['original']}\n\n"
            f"Example UPDATED CODEBOOK:\n{self.config['codebook_example']['updated']}"
        )
        for agent in self.scientists:
            agent.reset_context()
            agent.add_user_message(update_prompt)
        
        self.logger.log("--- Agents Proposing Initial Codebook Updates ---\n")
        proposals = [agent.propose_codebook_update(self.codebook) for agent in self.scientists]
        for i, proposal in enumerate(proposals):
            self.logger.log(f"Agent {i+1}'s Proposal: {proposal}\n")

        if all(not p.need_update for p in proposals):
            self.logger.log("--- No codebook changes proposed. Keeping current codebook. ---\n")
            return
            
        # *** HUMAN INTERVENTION POINT (CODEBOOK PROPOSAL) ***
        if self.intervention_enabled and self.intervention_scope == 'extensive':
            if self._human_intervention(phase='codebook proposal'):
                # Re-run proposal with intervention context injected
                proposals = [agent.propose_codebook_update(self.codebook) for agent in self.scientists]
                for i, proposal in enumerate(proposals):
                    self.logger.log(f"Agent {i+1}'s Proposal (Post-Intervention): {proposal}\n")
        
        # Multi-round mediation and review loop
        current_proposals = proposals
        final_codebook = ""
        for round_num in range(self.discussion_rounds):
            self.logger.log(f"<Codebook Discussion Round {round_num + 1}>\n")
            
            self.logger.log("--- Mediator Summarizing Proposals ---\n")
            mediator_summary = self.mediator.mediate(current_proposals)
            mediator_message = f"{mediator_summary}\n\nDo you all agree with the unified CODEBOOK?"
            self.logger.log(f"Mediator's Summary & Proposal:\n{mediator_message}\n")
            
            self.logger.log("--- Agents Reviewing Mediated Codebook ---\n")
            opinions = [agent.review_mediated_codebook(mediator_message) for agent in self.scientists]
            for i, opinion in enumerate(opinions):
                self.logger.log(f"Agent {i+1}'s Final Opinion: {opinion}\n")
            
            final_codebook = mediator_summary # Store the latest mediated version

            # *** HUMAN INTERVENTION POINT (CODEBOOK REVIEW) ***
            if self.intervention_enabled and self.intervention_scope == 'extensive':
                if self._human_intervention(phase=f'codebook review round {round_num+1}'):
                    # Re-run review with intervention context injected
                    opinions = [agent.review_mediated_codebook(mediator_message) for agent in self.scientists]
                    for i, opinion in enumerate(opinions):
                        self.logger.log(f"Agent {i+1}'s Opinion (Post-Intervention): {opinion}\n")

            agreement = self.judge.check_codebook_agreement(opinions)
            self.logger.log(f"Codebook Agreement Verdict: {'Yes' if agreement else 'No'}\n")
            
            if agreement:
                self.logger.log("--- Consensus on Codebook Reached! ---\n")
                break
            
            current_proposals = opinions # Use latest opinions for the next round
            if round_num == self.discussion_rounds - 1:
                self.logger.log("--- Max rounds reached. Adopting last mediated codebook. ---\n")

        # Update codebook in all scientist agents for the next chunk
        self.codebook = final_codebook
        for agent in self.scientists:
            agent.update_codebook(self.codebook)
        self.logger.log("--- Final Codebook Adopted and Updated for all Agents. ---\n")
    
    def get_evaluator(self) -> Evaluator:
        """Return the evaluator instance for multi-run aggregation."""
        return self.evaluator