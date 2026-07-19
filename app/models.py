"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel
from typing import Optional, List


# ── Request Models ─────────────────────────────────────────────────


class QuestionRequest(BaseModel):
    """User question with optional chat continuation."""
    question: str
    chat_id: Optional[str] = None


# ── Response Models ────────────────────────────────────────────────


class Source(BaseModel):
    """A retrieved document chunk used as evidence."""
    text: str
    document_name: str
    page_number: Optional[int] = None
    relevance_score: float


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


class DocumentInfo(BaseModel):
    """Document metadata for listing."""
    document_id: str
    filename: str
    chunk_count: int = 0
    total_pages: int = 0
    status: str = "processed"  # Default to 'processed' so legacy docs show up as ready
    uploaded_at: str


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
