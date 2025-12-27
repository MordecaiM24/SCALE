from pydantic import BaseModel


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