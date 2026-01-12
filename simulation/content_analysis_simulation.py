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

from utils.types import CodingResponse, SessionMemory, DisagreementRecord, CodebookEvolutionRecord
from evaluator import Evaluator, load_ground_truth

class ContentAnalysisSimulation:
    """
    Simulates a content analysis workflow with multiple AI coders.
    
    Enhanced for realistic context management:
    - SessionMemory tracks disagreements, resolutions, and learnings across the simulation
    - Disagreement reasoning is injected into discussion phases
    - Codebook evolution considers the full history of disagreements and changes
    - Agents maintain session memory to learn from past coding decisions
    """
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
        self.original_codebook = self.codebook  # Keep original for evolution tracking
        
        self.scientists = self._create_scientists()
        self.judge = JudgeAgent()
        self.mediator = MediatorAgent(self.client, self.model, config['prompt']['mediator'])
        self.logger.log(f"Initialized {self.num_agents} Social Scientist Agents For {self.config['dataset_name']} Task.\n")

        # Initialize session memory for realistic cross-phase context
        self.session_memory = SessionMemory()
        self.logger.log("Initialized SessionMemory for cross-phase learning.\n")

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
        """Initializes the SocialScientistAgent instances with unique IDs."""
        scientists = []
        personas = list(self.config['persona'].values())
        for i in range(self.num_agents):
            agent = SocialScientistAgent(
                client=self.client,
                model=self.model,
                persona=personas[i],
                codebook=self.codebook,
            )
            agent.agent_id = i + 1  # Assign ID for tracking in session memory
            scientists.append(agent)
        return scientists
    
    def _init_evaluator(self, df):
        """Initialize the evaluator with ground truth from the dataset."""
        self.ground_truth = load_ground_truth(df)
        self.evaluator = Evaluator(self.ground_truth)

    def _sync_session_memory_to_agents(self):
        """
        Updates all agents with the current session memory context.
        This allows agents to learn from past discussions and disagreements,
        mimicking how human coders calibrate over time.
        """
        discussion_context = self.session_memory.get_discussion_context()
        if discussion_context:
            self.logger.log(f"Syncing session memory to agents ({len(self.session_memory.disagreements)} disagreements tracked)\n")
            for agent in self.scientists:
                agent.update_session_memory(discussion_context)

    def _extract_key_learnings(self, text_id: str, initial_codes: List[CodingResponse], final_codes: List[CodingResponse]) -> List[str]:
        """
        Extracts key learnings from a resolved disagreement.
        These learnings are stored in session memory for future reference.
        """
        learnings = []
        initial_code_set = set(r.code for r in initial_codes)
        final_code_set = set(r.code for r in final_codes)
        
        # If consensus was reached from disagreement
        if len(initial_code_set) > 1 and len(final_code_set) == 1:
            final_code = list(final_code_set)[0]
            # Find agents who changed their minds
            for i, (init, final) in enumerate(zip(initial_codes, final_codes)):
                if init.code != final.code:
                    learnings.append(
                        f"Agent {i+1} changed from Code {init.code} to {final.code}: {final.reasoning[:100]}"
                    )
        
        # If disagreement persisted, note the sticking points
        if len(final_code_set) > 1:
            learnings.append(f"Unresolved: codes remained {list(final_code_set)}")
        
        return learnings

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
            
            # Update agents with current session memory before coding
            # (mimics how human coders learn from past discussions)
            self._sync_session_memory_to_agents()
            
            # Bot Annotation
            coding_results, coding_agreements = self._run_coding_phase(chunk)
            all_coding_results.update(coding_results)
            all_coding_agreements.update(coding_agreements)
            
            # Agent Discussion (now receives disagreement context)
            discussion_results, final_answers, final_agreements = self._run_discussion_phase(
                chunk, coding_results, coding_agreements
            )
            all_final_answers.update(final_answers)
            all_final_agreements.update(final_agreements)
            
            # Codebook Evolution (now receives full disagreement context)
            self._run_codebook_evolution_phase(chunk_id=i)
            
            # Log results for this chunk including session memory state
            chunk_log = {
                "chunk_id": i,
                "coding_phase": {"results": coding_results, "agreements": coding_agreements},
                "discussion_phase": {"history": discussion_results, "results": final_answers, "agreements": final_agreements},
                "final_codebook": self.codebook,
                "session_memory": {
                    "total_disagreements": len(self.session_memory.disagreements),
                    "resolved_disagreements": sum(1 for d in self.session_memory.disagreements if d.resolved),
                    "codebook_evolutions": len(self.session_memory.codebook_history),
                    "calibration_notes": self.session_memory.calibration_notes[-5:]  # Last 5 notes
                }
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
        
        # Log session memory summary
        self.logger.log("\n" + "=" * 50)
        self.logger.log("SESSION MEMORY SUMMARY")
        self.logger.log("=" * 50)
        self.logger.log(f"Total disagreements tracked: {len(self.session_memory.disagreements)}")
        self.logger.log(f"Resolved disagreements: {sum(1 for d in self.session_memory.disagreements if d.resolved)}")
        self.logger.log(f"Codebook evolutions: {len(self.session_memory.codebook_history)}")
        self.logger.log(f"Calibration notes: {len(self.session_memory.calibration_notes)}")
        
        if self.session_memory.codebook_history:
            self.logger.log("\nCodebook Evolution History:")
            for record in self.session_memory.codebook_history:
                self.logger.log(f"  Chunk {record.chunk_id}: {record.change_rationale}")
        
        self.logger.save_json(full_log, 'full_simulation_log.json')
        self.logger.save_json(eval_result, 'evaluation_results.json')
        
        # Save session memory for analysis
        session_memory_log = {
            "disagreements": [
                {
                    "text_id": d.text_id,
                    "text_content": d.text_content[:200] + "..." if len(d.text_content) > 200 else d.text_content,
                    "initial_codes": [r.code for r in d.agent_responses],
                    "resolved": d.resolved,
                    "resolution_codes": [r.code for r in d.resolution] if d.resolution else None,
                    "key_points": d.key_points
                }
                for d in self.session_memory.disagreements
            ],
            "codebook_evolutions": [
                {
                    "chunk_id": e.chunk_id,
                    "triggering_disagreements": e.triggering_disagreements,
                    "change_rationale": e.change_rationale,
                    "codebook_changed": e.original_codebook != e.updated_codebook
                }
                for e in self.session_memory.codebook_history
            ],
            "calibration_notes": self.session_memory.calibration_notes
        }
        self.logger.save_json(session_memory_log, 'session_memory.json')
        
        self.logger.log("\n===== Simulation Complete =====\n")
        
        return eval_result

    def _run_coding_phase(self, chunk: pd.Series):
        """
        Runs the coding phase where agents independently code each text.
        
        Enhanced to:
        - Record disagreements in session memory with full reasoning
        - Use past discussion context when coding (agents learn from previous texts)
        """
        self.logger.log("********** Bot Annotation **********\n")
    
        coding_results: Dict[str, List[CodingResponse]] = {}
        coding_agreements: Dict[str, bool] = {}
        
        # Get context from past discussions to inform coding
        discussion_context = self.session_memory.get_discussion_context()

        self.logger.log("--- Agents Coding Texts ---\n")
        for i, text in enumerate(chunk):
            text_id = f"Text-{chunk.index[i]+1}"
            self.logger.log(f"--- Coding {text_id} ---\n{text}\n")
            
            # Reset context but preserve session memory (learnings from past discussions)
            for agent in self.scientists:
                agent.reset_context(preserve_memory=True)
                agent.add_user_message(self.config['prompt']['coding'])
            
            # Code with discussion context (mimics human learning from past calibration)
            responses = [agent.code_text(text, discussion_context=discussion_context) 
                        for agent in self.scientists]
            
            for j, response in enumerate(responses):
                self.logger.log(f"Agent {j+1}: {response}\n")

            coding_results[text_id] = responses
            agreement = self.judge.check_agreement(responses)
            coding_agreements[text_id] = agreement
            self.logger.log(f"Judge's Verdict: {'Agreement' if agreement else 'Disagreement'}\n")
            
            # Record disagreements in session memory for later discussion/evolution
            if not agreement:
                disagreement_record = DisagreementRecord(
                    text_id=text_id,
                    text_content=str(text),
                    agent_responses=responses,
                    resolved=False
                )
                self.session_memory.add_disagreement(disagreement_record)
                self.logger.log(f"[SessionMemory] Recorded disagreement for {text_id}\n")

        return coding_results, coding_agreements

    def _run_discussion_phase(self, chunk: pd.Series, coding_results: Dict, coding_agreements: Dict):
        """
        Runs the discussion phase where agents deliberate on disagreements.
        
        Enhanced to:
        - Inject full reasoning from other coders into discussions
        - Use past disagreement patterns as context
        - Track discussion resolutions and key learnings
        - Update session memory with resolution outcomes
        """
        self.logger.log("********** Agent Discussion **********\n")

        discussion_results: Dict[str, List[List[CodingResponse]]] = {}
        final_answers: Dict[str, List[CodingResponse]] = {}
        final_agreements: Dict[str, bool] = {}
        
        # Get context from past resolved disagreements to help current discussions
        past_disagreement_context = self.session_memory.get_discussion_context()

        self.logger.log("\n--- Agents Discussing Disagreements ---\n")
        for i, text in enumerate(chunk):
            text_id = f"Text-{chunk.index[i]+1}"
            if not coding_agreements[text_id]:
                self.logger.log(f"\n--- Discussing {text_id} ---\n")
                
                # Reset context but preserve session memory
                for agent in self.scientists:
                    agent.reset_context(preserve_memory=True)
                    agent.add_user_message(self.config['prompt']['discussion'])
                
                discussion_history: List[List[CodingResponse]] = [coding_results[text_id]]
                initial_responses = coding_results[text_id]
                
                # Build context about THIS specific disagreement
                current_disagreement_context = self._build_disagreement_context(
                    text_id, initial_responses, past_disagreement_context
                )

                agreement = False
                for round_num in range(self.discussion_rounds):
                    self.logger.log(f"<Discussion Round {round_num + 1}>\n")
                    current_answers = discussion_history[-1]

                    # Enhanced discuss() with disagreement context and round number
                    next_round_answers = [
                        agent.discuss(
                            text=str(text), 
                            your_answer=current_answers[j], 
                            other_answers=current_answers[:j] + current_answers[j+1:],
                            disagreement_context=current_disagreement_context,
                            round_num=round_num + 1
                        )
                        for j, agent in enumerate(self.scientists)
                    ]
                    
                    for j, answer in enumerate(next_round_answers):
                        self.logger.log(f"Agent {j+1}: {answer}\n")
                    
                    # *** HUMAN INTERVENTION POINT (DISCUSSION) ***
                    if self.intervention_enabled:
                        if self._human_intervention(phase='discussion'):
                            # Re-run discuss with intervention context injected
                            next_round_answers = [
                                agent.discuss(
                                    text=str(text), 
                                    your_answer=current_answers[j], 
                                    other_answers=current_answers[:j] + current_answers[j+1:],
                                    disagreement_context=current_disagreement_context,
                                    round_num=round_num + 1
                                )
                                for j, agent in enumerate(self.scientists)
                            ]
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
                
                # Update session memory with resolution and key learnings
                key_learnings = self._extract_key_learnings(
                    text_id, initial_responses, discussion_history[-1]
                )
                self.session_memory.resolve_disagreement(
                    text_id=text_id,
                    resolution=discussion_history[-1],
                    key_points=key_learnings
                )
                
                # Add calibration note if consensus was reached
                if agreement:
                    final_code = discussion_history[-1][0].code
                    self.session_memory.add_calibration_note(
                        f"{text_id}: Resolved to Code {final_code} after {len(discussion_history)-1} rounds"
                    )
                else:
                    self.session_memory.add_calibration_note(
                        f"{text_id}: Unresolved after {self.discussion_rounds} rounds - "
                        f"may indicate codebook ambiguity"
                    )
                
                self.logger.log(f"[SessionMemory] Updated resolution for {text_id}: "
                              f"{'Resolved' if agreement else 'Unresolved'}\n")

        return discussion_results, final_answers, final_agreements
    
    def _build_disagreement_context(
        self, 
        text_id: str, 
        responses: List[CodingResponse],
        past_context: str
    ) -> str:
        """
        Builds context about the current disagreement to help agents understand
        why they disagree and what patterns might be relevant from past discussions.
        """
        lines = []
        
        # Add past context if available
        if past_context:
            lines.append(past_context)
            lines.append("")
        
        # Analyze the current disagreement
        codes = [r.code for r in responses]
        unique_codes = list(set(codes))
        
        lines.append(f"CURRENT DISAGREEMENT ANALYSIS for {text_id}:")
        lines.append(f"Codes given: {codes} (unique: {unique_codes})")
        
        # Group reasoning by code
        for code in unique_codes:
            matching = [r for r in responses if r.code == code]
            lines.append(f"\nReasoning for Code {code}:")
            for i, r in enumerate(matching):
                lines.append(f"  - {r.reasoning[:200]}...")
        
        return "\n".join(lines)

    def _run_codebook_evolution_phase(self, chunk_id: int = 0):
        """
        Runs the codebook evolution phase where agents propose and discuss updates.
        
        Enhanced to:
        - Include full disagreement context in proposals (agents know WHY changes are needed)
        - Track evolution history with rationale for changes
        - Store evolution records in session memory for future reference
        - Ensure mediator considers both disagreements and past evolutions
        """
        self.logger.log("********** Codebook Evolution **********\n")
        
        # Get disagreement context to inform codebook updates
        disagreement_context = self.session_memory.get_codebook_evolution_context()
        evolution_history = self.session_memory.get_codebook_evolution_summary()
        
        # Log context being used
        recent_disagreements = self.session_memory.get_recent_disagreements(5)
        triggering_text_ids = [d.text_id for d in recent_disagreements if not d.resolved or 
                              (d.resolved and len(set(r.code for r in d.agent_responses)) > 1)]
        self.logger.log(f"Disagreements informing evolution: {triggering_text_ids}\n")
        
        # Initial proposal step with rich context
        update_prompt = (
            f"{self.config['prompt']['update']}\n\n"
            f"Here is an example of updating CODEBOOK:\n"
            f"Example ORIGINAL CODEBOOK:\n{self.config['codebook_example']['original']}\n\n"
            f"Example UPDATED CODEBOOK:\n{self.config['codebook_example']['updated']}"
        )
        for agent in self.scientists:
            agent.reset_context(preserve_memory=True)
            agent.add_user_message(update_prompt)
        
        self.logger.log("--- Agents Proposing Initial Codebook Updates ---\n")
        # Enhanced proposals with disagreement and evolution context
        proposals = [
            agent.propose_codebook_update(
                original_codebook=self.codebook,
                disagreement_context=disagreement_context,
                evolution_history=evolution_history
            ) 
            for agent in self.scientists
        ]
        for i, proposal in enumerate(proposals):
            self.logger.log(f"Agent {i+1}'s Proposal: {proposal}\n")

        if all(not p.need_update for p in proposals):
            self.logger.log("--- No codebook changes proposed. Keeping current codebook. ---\n")
            # Record this as a non-evolution event
            evolution_record = CodebookEvolutionRecord(
                chunk_id=chunk_id,
                original_codebook=self.codebook,
                updated_codebook=self.codebook,
                triggering_disagreements=triggering_text_ids,
                agent_proposals=[p.reasoning for p in proposals],
                change_rationale="No changes needed - codebook deemed adequate"
            )
            self.session_memory.add_codebook_evolution(evolution_record)
            return
            
        # *** HUMAN INTERVENTION POINT (CODEBOOK PROPOSAL) ***
        if self.intervention_enabled and self.intervention_scope == 'extensive':
            if self._human_intervention(phase='codebook proposal'):
                # Re-run proposal with intervention context injected
                proposals = [
                    agent.propose_codebook_update(
                        original_codebook=self.codebook,
                        disagreement_context=disagreement_context,
                        evolution_history=evolution_history
                    ) 
                    for agent in self.scientists
                ]
                for i, proposal in enumerate(proposals):
                    self.logger.log(f"Agent {i+1}'s Proposal (Post-Intervention): {proposal}\n")
        
        # Multi-round mediation and review loop
        current_proposals = proposals
        final_codebook = self.codebook  # Default to current if no consensus
        original_codebook = self.codebook
        
        for round_num in range(self.discussion_rounds):
            self.logger.log(f"<Codebook Discussion Round {round_num + 1}>\n")
            
            self.logger.log("--- Mediator Summarizing Proposals ---\n")
            # Enhanced mediation with disagreement and evolution context
            mediator_summary = self.mediator.mediate(
                proposals=current_proposals,
                disagreement_context=disagreement_context,
                evolution_history=evolution_history
            )
            mediator_message = f"{mediator_summary}\n\nDo you all agree with the unified CODEBOOK?"
            self.logger.log(f"Mediator's Summary & Proposal:\n{mediator_message}\n")
            
            self.logger.log("--- Agents Reviewing Mediated Codebook ---\n")
            # Enhanced review with disagreement context
            opinions = [
                agent.review_mediated_codebook(
                    mediator_summary=mediator_message,
                    disagreement_context=disagreement_context
                ) 
                for agent in self.scientists
            ]
            for i, opinion in enumerate(opinions):
                self.logger.log(f"Agent {i+1}'s Final Opinion: {opinion}\n")
            
            final_codebook = mediator_summary  # Store the latest mediated version

            # *** HUMAN INTERVENTION POINT (CODEBOOK REVIEW) ***
            if self.intervention_enabled and self.intervention_scope == 'extensive':
                if self._human_intervention(phase=f'codebook review round {round_num+1}'):
                    # Re-run review with intervention context injected
                    opinions = [
                        agent.review_mediated_codebook(
                            mediator_summary=mediator_message,
                            disagreement_context=disagreement_context
                        ) 
                        for agent in self.scientists
                    ]
                    for i, opinion in enumerate(opinions):
                        self.logger.log(f"Agent {i+1}'s Opinion (Post-Intervention): {opinion}\n")

            agreement = self.judge.check_codebook_agreement(opinions)
            self.logger.log(f"Codebook Agreement Verdict: {'Yes' if agreement else 'No'}\n")
            
            if agreement:
                self.logger.log("--- Consensus on Codebook Reached! ---\n")
                break
            
            current_proposals = opinions  # Use latest opinions for the next round
            if round_num == self.discussion_rounds - 1:
                self.logger.log("--- Max rounds reached. Adopting last mediated codebook. ---\n")

        # Record the evolution in session memory
        change_rationale = self.mediator.extract_change_rationale(
            original_codebook, final_codebook, proposals
        )
        evolution_record = CodebookEvolutionRecord(
            chunk_id=chunk_id,
            original_codebook=original_codebook,
            updated_codebook=final_codebook,
            triggering_disagreements=triggering_text_ids,
            agent_proposals=[p.reasoning for p in proposals],
            change_rationale=change_rationale
        )
        self.session_memory.add_codebook_evolution(evolution_record)
        self.logger.log(f"[SessionMemory] Recorded codebook evolution: {change_rationale}\n")

        # Update codebook in all scientist agents for the next chunk
        self.codebook = final_codebook
        for agent in self.scientists:
            agent.update_codebook(self.codebook, preserve_memory=True)
        self.logger.log("--- Final Codebook Adopted and Updated for all Agents. ---\n")
    
    def get_evaluator(self) -> Evaluator:
        """Return the evaluator instance for multi-run aggregation."""
        return self.evaluator