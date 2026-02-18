from pydantic import BaseModel
from typing import Dict, List


class CodingResponse(BaseModel):
    code: int
    reasoning: str
    confidence: float = 1.0
    memo: str | None = None


class CalibrationResult(BaseModel):
    text_id: str
    agent_codes: List[int]
    ground_truth: int
    agreement: bool
    notes: str


class AgreementResult(BaseModel):
    agreed: bool
    codes: List[int]
    confidence_weighted: bool = False
    disagreement_categories: List[int] | None = None


class IRRMetrics(BaseModel):
    cohens_kappa: Dict[str, float] | None = None
    krippendorffs_alpha: float | None = None
    fleiss_kappa: float | None = None


class CodebookUpdate(BaseModel):
    """Used for both proposing codebook updates and reviewing mediated codebooks.
    
    For proposals: need_update=True with new_codebook containing the proposed update.
    For reviews: need_update=False means agreement, need_update=True means disagreement with new_codebook.
    """
    need_update: bool
    reasoning: str
    new_codebook: str | None