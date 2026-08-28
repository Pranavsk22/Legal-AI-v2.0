# backend/api/models.py

from pydantic import BaseModel
from typing import List, Optional


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    risks: list[str] | None = None
    summary: str


class AskRequest(BaseModel):
    query: str
    source: Optional[str] = None  # Optional filter for specific source file


class Citation(BaseModel):
    clause: str
    source: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]


class DocumentDraftResponse(BaseModel):
    doc_id: str
    doc_type: str
    risk_flags: List[str]
    parties: Optional[List[str]] = None
    effective_date: Optional[str] = None
    governing_law: Optional[str] = None
    source_format: str
    chunks: List[str]


class DocumentConfirmRequest(BaseModel):
    doc_id: str
    doc_type: str
    risk_flags: List[str]
    parties: Optional[List[str]] = None
    effective_date: Optional[str] = None
    governing_law: Optional[str] = None
    source_format: str
    chunks: List[str]

