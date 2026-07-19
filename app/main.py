"""
FastAPI application — routes, lifecycle, and static file serving.

Endpoints:
  POST /api/upload          Upload a PDF document to the knowledge base
  POST /api/ask             Ask a question (returns summary + full answer)
  GET  /api/documents       List all uploaded documents
  DELETE /api/documents/{id} Remove a document and its chunks
  GET  /api/chats           List all chats
  DELETE /api/chats/{id} Delete a chat
  GET  /api/stats           Application-wide statistics
"""

import os
import uuid
import shutil
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from app.config import settings
from app.database import connect_to_mongo, close_mongo_connection, get_db
from app.document_processor import document_processor
from app.rag_pipeline import rag_pipeline
from app.qdrant_service import qdrant_service
from app.models import (
    QuestionRequest,
    AnswerResponse,
    DocumentUploadResponse,
    DocumentInfo,
    ChatInfo,
    StatsResponse,
)
from typing import List
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-24s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Application Lifecycle ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect to MongoDB, initialize Qdrant, create upload dir. Shutdown: close connections."""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    await connect_to_mongo()
    # Initialize Qdrant collection (creates it if it doesn't exist)
    qdrant_service.initialize_collection()
    logger.info("🚀 RAG Application started")
    yield
    await close_mongo_connection()
    logger.info("👋 RAG Application stopped")


app = FastAPI(
    title="RAG Application",
    description="Cost-optimized RAG with MongoDB context management",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Profiling Middleware ──────────────────────────────────────────
from fastapi.responses import HTMLResponse

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "RAG API is running. Please use the Streamlit UI on port 8501."}


# ── Document Endpoints ─────────────────────────────────────────────

@app.post("/api/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload and process a document into the knowledge base in the background."""
    if not file.filename or not file.filename.lower().endswith((".pdf", ".xlsx", ".xls", ".docx", ".doc")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF, Excel, and Word files are supported.",
        )

    document_id = str(uuid.uuid4())
    file_path = os.path.join(settings.UPLOAD_DIR, f"{document_id}_{file.filename}")

    try:
        # Save uploaded file temporarily
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        db = get_db()
        
        # Create an initial document metadata entry with "processing" status
        doc_record = {
            "document_id": document_id,
            "filename": file.filename,
            "total_pages": 0,
            "chunk_count": 0,
            "parent_chunk_count": 0,
            "file_hash": "",
            "status": "processing",
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        await db.documents.insert_one(doc_record)

        # Trigger background processing
        background_tasks.add_task(
            document_processor.process_document_async,
            file_path,
            file.filename,
            document_id
        )

        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            chunk_count=0,
            total_pages=0,
            status="processing"
        )

    except Exception as e:
        # Clean up temp file on failure
        if os.path.exists(file_path):
            os.remove(file_path)


@app.get("/api/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all documents in the knowledge base."""
    db = get_db()
    docs = await db.documents.find({}, {"_id": 0}).to_list(length=100)
    return [DocumentInfo(**doc) for doc in docs]


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Remove a document and all its chunks from the knowledge base."""
    db = get_db()

    result = await db.documents.delete_one({"document_id": document_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Document not found")

    chunk_result = await db.chunks.delete_many({"document_id": document_id})
    logger.info(
        f"Deleted document {document_id} and {chunk_result.deleted_count} chunks"
    )

    return {
        "status": "deleted",
        "document_id": document_id,
        "chunks_removed": chunk_result.deleted_count,
    }


from fastapi.responses import StreamingResponse

# ── Question / Answer Endpoints ───────────────────────────────────

@app.post("/api/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question against the knowledge base (Streaming SSE)."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Return the async generator wrapped in a StreamingResponse
        return StreamingResponse(
            rag_pipeline.ask_stream(
                question=request.question.strip(),
                chat_id=request.chat_id,
            ),
            media_type="text/event-stream"
        )

    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Chat Endpoints ─────────────────────────────────────────

@app.get("/api/chats", response_model=List[ChatInfo])
async def list_chats():
    """List all chats with their token savings stats."""
    db = get_db()
    chats = await db.chats.find({}, {"_id": 0}).to_list(length=100)
    return [
        ChatInfo(
            chat_id=c["chat_id"],
            turn_count=c["turn_count"],
            created_at=c["created_at"],
            updated_at=c["updated_at"],
            total_tokens_saved=c.get("total_tokens_saved", 0),
        )
        for c in chats
    ]


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat and its context."""
    db = get_db()
    result = await db.chats.delete_one(
        {"chat_id": chat_id}
    )
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chat not found")

    return {"status": "deleted", "chat_id": chat_id}


# ── Stats Endpoint ─────────────────────────────────────────────────

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get application-wide statistics."""
    db = get_db()

    total_docs = await db.documents.count_documents({})
    total_chunks = await db.chunks.count_documents({})
    total_chats = await db.chats.count_documents({})

    # Sum total tokens saved across all chats
    pipeline = [
        {"$group": {"_id": None, "total": {"$sum": "$total_tokens_saved"}}}
    ]
    agg_result = await db.chats.aggregate(pipeline).to_list(length=1)
    total_saved = agg_result[0]["total"] if agg_result else 0

    return StatsResponse(
        total_documents=total_docs,
        total_chunks=total_chunks,
        total_chats=total_chats,
        total_tokens_saved=total_saved,
    )
