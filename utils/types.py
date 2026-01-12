from pydantic import BaseModel
from typing import List, Dict, Optional
from dataclasses import dataclass, field


class CodingResponse(BaseModel):
    code: int
    reasoning: str


class CodebookUpdate(BaseModel):
    """Used for both proposing codebook updates and reviewing mediated codebooks.
    
    For proposals: need_update=True with new_codebook containing the proposed update.
    For reviews: need_update=False means agreement, need_update=True means disagreement with new_codebook.
    """
    need_update: bool
    reasoning: str
    new_codebook: str | None


@dataclass
class DisagreementRecord:
    """Records a disagreement instance with full context for learning."""
    text_id: str
    text_content: str
    agent_responses: List[CodingResponse]
    resolution: Optional[List[CodingResponse]] = None  # Final answers after discussion
    resolved: bool = False
    key_points: List[str] = field(default_factory=list)  # Learnings from this disagreement
    
    def get_reasoning_summary(self) -> str:
        """Summarizes the different reasoning approaches from agents."""
        lines = []
        for i, resp in enumerate(self.agent_responses):
            lines.append(f"Agent {i+1} (Code {resp.code}): {resp.reasoning}")
        return "\n".join(lines)
    
    def get_resolution_summary(self) -> str:
        """Summarizes how the disagreement was resolved."""
        if not self.resolution:
            return "Unresolved"
        codes = [r.code for r in self.resolution]
        if len(set(codes)) == 1:
            return f"Consensus reached on Code {codes[0]}"
        return f"Remained split: {codes}"


@dataclass  
class CodebookEvolutionRecord:
    """Tracks a single codebook evolution with context."""
    chunk_id: int
    original_codebook: str
    updated_codebook: str
    triggering_disagreements: List[str]  # text_ids that prompted evolution
    agent_proposals: List[str]  # Summary of each agent's proposal reasoning
    change_rationale: str  # Why the change was made
    
    
@dataclass
class SessionMemory:
    """
    Maintains long-term memory across the simulation to mirror how human coders
    learn and calibrate over time.
    
    This tracks:
    - Disagreement history with full reasoning
    - Discussion outcomes and key learnings  
    - Codebook evolution history with rationale
    - Calibration notes (edge cases, ambiguous texts)
    """
    disagreements: List[DisagreementRecord] = field(default_factory=list)
    codebook_history: List[CodebookEvolutionRecord] = field(default_factory=list)
    calibration_notes: List[str] = field(default_factory=list)
    
    def add_disagreement(self, record: DisagreementRecord):
        """Records a new disagreement."""
        self.disagreements.append(record)
    
    def resolve_disagreement(self, text_id: str, resolution: List[CodingResponse], key_points: List[str] = None):
        """Updates a disagreement record with its resolution."""
        for d in self.disagreements:
            if d.text_id == text_id:
                d.resolution = resolution
                d.resolved = True
                if key_points:
                    d.key_points = key_points
                break
    
    def add_codebook_evolution(self, record: CodebookEvolutionRecord):
        """Records a codebook evolution event."""
        self.codebook_history.append(record)
    
    def add_calibration_note(self, note: str):
        """Adds a calibration note for future reference."""
        self.calibration_notes.append(note)
    
    def get_recent_disagreements(self, n: int = 5) -> List[DisagreementRecord]:
        """Returns the n most recent disagreements for context."""
        return self.disagreements[-n:] if self.disagreements else []
    
    def get_unresolved_patterns(self) -> str:
        """Identifies patterns in unresolved disagreements."""
        unresolved = [d for d in self.disagreements if not d.resolved]
        if not unresolved:
            return "No unresolved disagreements."
        
        lines = ["Recent unresolved disagreements:"]
        for d in unresolved[-3:]:  # Last 3 unresolved
            lines.append(f"- {d.text_id}: {d.get_reasoning_summary()[:200]}...")
        return "\n".join(lines)
    
    def get_codebook_evolution_summary(self) -> str:
        """Summarizes how the codebook has evolved and why."""
        if not self.codebook_history:
            return "No codebook changes yet."
        
        lines = ["Codebook Evolution History:"]
        for i, record in enumerate(self.codebook_history):
            lines.append(f"\n[Evolution {i+1} after Chunk {record.chunk_id}]")
            lines.append(f"Triggered by: {', '.join(record.triggering_disagreements)}")
            lines.append(f"Rationale: {record.change_rationale}")
        return "\n".join(lines)
    
    def get_discussion_context(self, text_id: str = None) -> str:
        """
        Generates context from past disagreements to inform current discussions.
        This helps agents learn from previous coding challenges.
        """
        if not self.disagreements:
            return ""
        
        lines = ["CONTEXT FROM PREVIOUS DISCUSSIONS:"]
        
        # Include resolved disagreements as learning examples
        resolved = [d for d in self.disagreements if d.resolved]
        if resolved:
            lines.append("\nResolved disagreements and learnings:")
            for d in resolved[-3:]:  # Last 3 resolved
                lines.append(f"\n• Text {d.text_id}: {d.get_resolution_summary()}")
                if d.key_points:
                    lines.append(f"  Key learnings: {'; '.join(d.key_points)}")
        
        # Include calibration notes
        if self.calibration_notes:
            lines.append("\nCalibration notes from team discussions:")
            for note in self.calibration_notes[-5:]:
                lines.append(f"• {note}")
        
        return "\n".join(lines) if len(lines) > 1 else ""
    
    def get_codebook_evolution_context(self) -> str:
        """
        Generates context about disagreements and discussions to inform codebook evolution.
        This ensures codebook updates address actual coding challenges.
        """
        lines = ["DISAGREEMENT CONTEXT FOR CODEBOOK EVOLUTION:"]
        
        # Summarize all disagreements from recent chunk
        recent = self.get_recent_disagreements(10)
        if recent:
            lines.append(f"\nTotal disagreements encountered: {len(self.disagreements)}")
            lines.append(f"Resolved: {sum(1 for d in self.disagreements if d.resolved)}")
            
            lines.append("\nKey disagreement patterns:")
            for d in recent:
                lines.append(f"\n[{d.text_id}] '{d.text_content[:100]}...'")
                lines.append(f"  Initial codes: {[r.code for r in d.agent_responses]}")
                lines.append(f"  Reasoning differences:")
                for i, r in enumerate(d.agent_responses):
                    lines.append(f"    Agent {i+1}: {r.reasoning[:150]}...")
                if d.resolved:
                    lines.append(f"  Resolution: {d.get_resolution_summary()}")
                    if d.key_points:
                        lines.append(f"  Learnings: {'; '.join(d.key_points)}")
        
        return "\n".join(lines)