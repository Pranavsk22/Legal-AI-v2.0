# backend/nlp_modules/schema.py
from pydantic import BaseModel, Field
from typing import List, Optional, Union

class DocumentRecord(BaseModel):
    doc_id: str = Field(..., min_length=1, description="Unique identifier for the document")
    doc_type: str = Field(..., min_length=1, description="Type of the document, e.g. Agreement, NDA")
    risk_flags: List[str] = Field(default_factory=list, description="List of detected risk labels")
    parties: Union[List[str], str, None] = Field(default=None, description="Parties involved in the contract")
    effective_date: Optional[str] = Field(default=None, description="Effective date of the document in YYYY-MM-DD or readable text format")
    governing_law: Optional[str] = Field(default=None, description="Governing law/jurisdiction")
    source_format: str = Field(..., min_length=1, description="Source format of the document, e.g. PDF, DOCX, TXT")
    clause_index: int = Field(..., ge=0, description="Index of the clause/chunk in the document")
