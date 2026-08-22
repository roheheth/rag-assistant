"""
Pydantic models for API request/response validation.

Extended for NatWest Banking Compliance:
  - QuestionRequest: carries user_role and user_department for RBAC.
  - DocumentUploadResponse / DocumentInfo: carry clearance_level, department,
    effective_date, expiry_date, and doc_status for temporal policy filtering.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date


# ── RBAC Role Constants ────────────────────────────────────────────────
ROLES = ["Teller", "Manager", "Executive", "Admin"]

# Clearance levels each role is permitted to access
ROLE_CLEARANCE_MAP: dict[str, List[str]] = {
    "Teller":    ["Public"],
    "Manager":   ["Public", "Internal"],
    "Executive": ["Public", "Internal", "Restricted"],
    "Admin":     ["Public", "Internal", "Restricted"],
}


# ── Request Models ─────────────────────────────────────────────────────


class QuestionRequest(BaseModel):
    """User question with optional chat continuation and RBAC context."""
    question: str
    chat_id: Optional[str] = None
    # RBAC — who is asking?
    user_role: str = Field(
        default="Teller",
        description="User's role for access control (Teller, Manager, Executive, Admin)"
    )
    user_department: str = Field(
        default="Retail",
        description="User's department for content scoping (e.g., Retail, Lending, Compliance)"
    )


# ── Response Models ────────────────────────────────────────────────────


class Source(BaseModel):
    """A retrieved document chunk used as evidence, with banking citation fields."""
    text: str
    document_name: str
    page_number: Optional[int] = None
    relevance_score: float
    # Banking-specific citation metadata
    effective_date: Optional[str] = None
    doc_status: Optional[str] = None


class TokenStats(BaseModel):
    """Token usage statistics for a single query."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AnswerResponse(BaseModel):
    """Dual-format answer with metadata."""
    summary_answer: str
    full_answer: str
    sources: List[Source]
    chat_id: str
    turn_number: int
    token_stats: TokenStats
    tokens_saved: int = 0


class DocumentUploadResponse(BaseModel):
    """Response after successful document ingestion."""
    document_id: str
    filename: str
    chunk_count: int = 0
    total_pages: int = 0
    status: str = "processing"
    # Banking compliance metadata
    clearance_level: str = "Internal"
    department: str = "Retail"
    effective_date: str = ""
    expiry_date: str = ""
    doc_status: str = "Active"


class DocumentInfo(BaseModel):
    """Document metadata for listing."""
    document_id: str
    filename: str
    chunk_count: int = 0
    total_pages: int = 0
    status: str = "processed"
    uploaded_at: str
    # Banking compliance metadata
    clearance_level: str = "Internal"
    department: str = "Retail"
    effective_date: str = ""
    expiry_date: str = ""
    doc_status: str = "Active"


class ChatInfo(BaseModel):
    """Chat metadata for listing."""
    chat_id: str
    turn_count: int
    created_at: str
    updated_at: str
    total_tokens_saved: int


class StatsResponse(BaseModel):
    """Application-wide statistics."""
    total_documents: int
    total_chunks: int
    total_chats: int
    total_tokens_saved: int
