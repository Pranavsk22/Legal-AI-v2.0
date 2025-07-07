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
