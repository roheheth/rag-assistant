"""
PDF document processing pipeline: extract → chunk → embed → store.

Uses PyPDF2 for text extraction and LangChain's RecursiveCharacterTextSplitter
for intelligent chunking with overlap for context preservation.
"""

import os
import re
import uuid
import asyncio
from datetime import datetime, timezone
from typing import List, Dict
import mammoth
from python_calamine import CalamineWorkbook
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings
from app.embeddings import embedding_service
from app.database import get_db
from app.qdrant_service import qdrant_service
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles the full document ingestion pipeline."""

    def __init__(self):
        # Parent splitter — creates large context blocks stored for the LLM
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.PARENT_CHUNK_SIZE,
            chunk_overlap=settings.PARENT_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        # Child splitter — creates small precise chunks used for vector search
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHILD_CHUNK_SIZE,
            chunk_overlap=settings.CHILD_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def extract_text_from_pdf(self, file_path: str) -> List[Dict]:
        """
        Extract text from a PDF file, page by page.

        Returns:
            List of dicts with 'text' and 'page_number' keys.
        """
        pages = []
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            text = page.get_text()
            if text and text.strip():
                pages.append({
                    "text": text.strip(),
                    "page_number": i + 1,
                })
        logger.info(f"Extracted text from {len(pages)}/{len(doc)} pages")
        return pages

    def extract_text_from_excel(self, file_path: str) -> List[Dict]:
        """
        Extract text from an Excel file sheet by sheet using Calamine (Rust reader).
        """
        pages = []
        try:
            workbook = CalamineWorkbook.from_path(file_path)
            for i, sheet_name in enumerate(workbook.sheet_names):
                sheet = workbook.get_sheet_by_name(sheet_name)
                rows = sheet.to_python()
                
                # Filter out completely empty rows and clean cell formatting
                clean_rows = []
                for row in rows:
                    if any(cell is not None and str(cell).strip() != "" for cell in row):
                        clean_rows.append([str(cell) if cell is not None else "" for cell in row])
                
                if not clean_rows:
                    continue
                
                # Convert tabular data to clean tab-separated lines
                text_lines = []
                for row in clean_rows:
                    text_lines.append("\t".join(row))
                text = "\n".join(text_lines)
                
                pages.append({
                    "text": f"Sheet: {sheet_name}\n" + text.strip(),
                    "page_number": i + 1,
                })
            logger.info(f"Extracted text from {len(pages)}/{len(workbook.sheet_names)} sheets")
        except Exception as e:
            logger.error(f"Error reading Excel file {file_path}: {e}")
        return pages

    def extract_text_from_docx(self, file_path: str) -> List[Dict]:
        """
        Extract text from a Word document (.docx) and convert it to clean HTML.
        This preserves structural layout (tables, lists) for accurate RAG.
        """
        pages = []
        try:
            with open(file_path, "rb") as docx_file:
                # mammoth converts tables to HTML tables, lists to HTML lists
                result = mammoth.convert_to_html(docx_file)
                html = result.value
                
                for warning in result.messages:
                    logger.warning(f"Mammoth warning: {warning.message}")
                
                if html.strip():
                    pages.append({
                        "text": html.strip(),
                        "page_number": 1,
                    })
            logger.info(f"Extracted text from Word document (converted to HTML)")
        except Exception as e:
            logger.error(f"Error reading Word document {file_path}: {e}")
        return pages

    def create_parent_child_chunks(self, pages: List[Dict]) -> Dict:
        """
        Build a two-tier chunk structure from extracted pages.

        Each page is first split into large Parent chunks (for LLM context).
        Each Parent is then split into small Child chunks (for vector search).
        Children carry a `parent_id` pointer so the retriever can fetch the
        full Parent text after finding a matching Child.

        Returns:
            {
                "parents": [{"parent_id", "text", "page_number"}, ...],
                "children": [{"child_id", "parent_id", "text", "page_number"}, ...],
            }
        """
        parents = []
        children = []

        for page in pages:
            parent_texts = self.parent_splitter.split_text(page["text"])

            for parent_text in parent_texts:
                parent_id = str(uuid.uuid4())
                parents.append({
                    "parent_id": parent_id,
                    "text": parent_text,
                    "page_number": page["page_number"],
                })

                # Split parent into children
                child_texts = self.child_splitter.split_text(parent_text)
                for j, child_text in enumerate(child_texts):
                    children.append({
                        "child_id": f"{parent_id}_{j}",
                        "parent_id": parent_id,
                        "text": child_text,
                        "page_number": page["page_number"],
                    })

        logger.info(
            f"Created {len(parents)} parent chunks and {len(children)} child chunks "
            f"from {len(pages)} pages"
        )
        return {"parents": parents, "children": children}

    def _clean_extracted_text(self, text: str) -> str:
        """
        Applies structural cleaning to the text in-memory:
        1. Consolidates multiple spaces and tabs.
        2. Removes blank lines.
        3. Removes page number lines.
        """
        # Replace multiple spaces/tabs with a single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Skip page numbers (e.g., "1", "Page 1", "1 / 10", "Page 1 of 5")
            if re.match(r'^(page\s+)?\d+(\s*(of|/)\s*\d+)?$', stripped, re.IGNORECASE):
                continue
            cleaned_lines.append(stripped)
            
        return "\n".join(cleaned_lines)

    async def process_document(self, file_path: str, filename: str) -> Dict:
        """
        Full ingestion pipeline (Parent-Child) with File Hash Deduplication:
        1. Calculate file MD5 hash and check database
        2. If duplicate, skip processing and return existing doc info
        3. Else, extract text, chunk, embed, and store with file_hash
        """
        import hashlib
        db = get_db()

        # ── Step 0: Calculate MD5 file hash ────────────────────────
        logger.info(f"Calculating hash for file: {filename}")
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        file_hash = hasher.hexdigest()

        # ── Step 1: Check for existing document in MongoDB ─────────
        existing_doc = await db.documents.find_one({"file_hash": file_hash})
        if existing_doc:
            logger.info(
                f"✓ Document '{filename}' already exists in database (hash match: {file_hash}). "
                f"Skipping extraction and embedding."
            )
            # Remove the temporary uploaded file
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {file_path}: {e}")
                
            return {
                "document_id": existing_doc["document_id"],
                "filename": existing_doc["filename"],
                "chunk_count": existing_doc["chunk_count"],
                "total_pages": existing_doc["total_pages"],
                "status": "already_exists",
            }

        document_id = str(uuid.uuid4())

        # ── Step 2: Extract text ───────────────────────────────────
        logger.info(f"Processing document: {filename}")

        if filename.lower().endswith(('.xlsx', '.xls')):
            pages = self.extract_text_from_excel(file_path)
        elif filename.lower().endswith(('.docx', '.doc')):
            pages = self.extract_text_from_docx(file_path)
        else:
            pages = self.extract_text_from_pdf(file_path)

        # ── Step 2b: Apply structural cleaning in-memory ───────────
        logger.info(f"Applying structural cleaning to extracted text from '{filename}'")
        for page in pages:
            page["text"] = self._clean_extracted_text(page["text"])

        total_pages = len(pages)

        if not pages:
            raise ValueError(
                f"No text could be extracted from '{filename}'. "
                "The document may be image-based or empty."
            )

        # ── Step 3: Build Parent-Child structure ───────────────────
        result = self.create_parent_child_chunks(pages)
        parents = result["parents"]
        children = result["children"]

        if not children:
            raise ValueError("No chunks were generated from the document.")

        # ── Step 4: Embed Children only ────────────────────────────
        logger.info(f"Generating embeddings for {len(children)} child chunks...")
        child_texts = [c["text"] for c in children]
        embeddings = await embedding_service.get_embeddings_batch(
            child_texts, is_query=False
        )

        # ── Step 5: Store in MongoDB ───────────────────────────────
        # 5a — Document record
        doc_record = {
            "document_id": document_id,
            "filename": filename,
            "total_pages": total_pages,
            "chunk_count": len(children),
            "parent_chunk_count": len(parents),
            "file_hash": file_hash,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        await db.documents.insert_one(doc_record)

        # 4b — Parent chunks (stored for LLM context retrieval)
        parent_records = [
            {
                "parent_id": parent["parent_id"],
                "document_id": document_id,
                "document_name": filename,
                "text": parent["text"],
                "page_number": parent["page_number"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            for parent in parents
        ]
        await db.parent_chunks.insert_many(parent_records)

        # 4c — Child chunks (stored with embeddings for vector search)
        child_records = [
            {
                "chunk_id": child["child_id"],
                "parent_id": child["parent_id"],
                "document_id": document_id,
                "document_name": filename,
                "text": child["text"],
                # Tokenize and save words on upload to avoid doing it on every search query
                "tokens": re.findall(r'\w+', child["text"].lower()),
                "page_number": child["page_number"],
                "embedding": embedding,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            for child, embedding in zip(children, embeddings)
        ]
        await db.chunks.insert_many(child_records)

        logger.info(
            f"✓ Stored {len(parent_records)} parent chunks and "
            f"{len(child_records)} child chunks for '{filename}' "
            f"(document_id={document_id})"
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "chunk_count": len(children),
            "total_pages": total_pages,
            "status": "processed",
        }

    async def process_document_async(
        self, file_path: str, filename: str, document_id: str
    ):
        """
        Background task execution of the ingestion pipeline with detailed logging.
        Applies cleaning, chunking, and vectorization, then updates MongoDB.
        """
        import hashlib
        import time
        db = get_db()
        
        start_time = time.time()
        logger.info(f"🚀 [Ingestion Start] ID: {document_id} | File: {filename}")

        try:
            # 1. Calculate File Size & MD5 Hash
            file_size_kb = os.path.getsize(file_path) / 1024
            logger.info(f"📂 [File Details] Name: {filename} | Size: {file_size_kb:.2f} KB")

            hash_start = time.time()
            hasher = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            file_hash = hasher.hexdigest()
            logger.info(f"🔑 [MD5 Hash Calculated] Hash: {file_hash} | Time: {(time.time() - hash_start)*1000:.2f} ms")

            # 2. Check for Duplicate Hash
            dup_start = time.time()
            existing_doc = await db.documents.find_one({"file_hash": file_hash})
            if existing_doc:
                logger.info(
                    f"🎯 [Duplicate Match] Document '{filename}' already exists (ID: {existing_doc['document_id']}). "
                    f"Skipping extraction & embedding. Time: {(time.time() - dup_start)*1000:.2f} ms"
                )
                await db.documents.update_one(
                    {"document_id": document_id},
                    {
                        "$set": {
                            "file_hash": file_hash,
                            "total_pages": existing_doc["total_pages"],
                            "chunk_count": existing_doc["chunk_count"],
                            "parent_chunk_count": existing_doc.get("parent_chunk_count", 0),
                            "status": "processed",
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        }
                    }
                )
                return

            # 3. Extract Text
            extract_start = time.time()
            logger.info(f"⚡ [Text Extraction] Starting parser for: {filename}")
            if filename.lower().endswith(('.xlsx', '.xls')):
                pages = self.extract_text_from_excel(file_path)
            elif filename.lower().endswith(('.docx', '.doc')):
                pages = self.extract_text_from_docx(file_path)
            else:
                pages = self.extract_text_from_pdf(file_path)

            total_pages = len(pages)
            raw_char_count = sum(len(p["text"]) for p in pages)
            logger.info(
                f"📝 [Extraction Complete] Pages/Sheets: {total_pages} | "
                f"Raw characters: {raw_char_count} | Time: {time.time() - extract_start:.3f} s"
            )

            if not pages:
                raise ValueError("No text could be extracted from the document.")

            # 4. Clean Text (In-Memory Compression)
            clean_start = time.time()
            for page in pages:
                page["text"] = self._clean_extracted_text(page["text"])
            
            clean_char_count = sum(len(p["text"]) for p in pages)
            char_reduction = ((raw_char_count - clean_char_count) / raw_char_count) * 100 if raw_char_count > 0 else 0
            logger.info(
                f"🧹 [Cleaning Complete] Compressed characters: {clean_char_count} | "
                f"Reduction: {char_reduction:.1f}% | Time: {(time.time() - clean_start)*1000:.2f} ms"
            )

            # 5. Build Parent-Child Chunks
            chunk_start = time.time()
            result = self.create_parent_child_chunks(pages)
            parents = result["parents"]
            children = result["children"]
            logger.info(
                f"🧩 [Chunking Complete] Generated {len(parents)} Parents | "
                f"{len(children)} Children | Time: {(time.time() - chunk_start)*1000:.2f} ms"
            )

            if not children:
                raise ValueError("No chunks were generated from the document.")

            # 6. Embed Children Chunks
            embed_start = time.time()
            logger.info(f"🤖 [Embedding Generation] Requesting embeddings for {len(children)} child chunks...")
            child_texts = [c["text"] for c in children]
            
            embeddings = await embedding_service.get_embeddings_batch(
                child_texts, is_query=False
            )
            logger.info(f"✨ [Embeddings Complete] Generated {len(embeddings)} vectors | Time: {time.time() - embed_start:.3f} s")

            # 7. Store in MongoDB (metadata) + Qdrant (vectors)
            db_start = time.time()

            # 7a — Parent chunks → MongoDB (full text for LLM context)
            parent_records = [
                {
                    "parent_id": parent["parent_id"],
                    "document_id": document_id,
                    "document_name": filename,
                    "text": parent["text"],
                    "page_number": parent["page_number"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                for parent in parents
            ]
            await db.parent_chunks.insert_many(parent_records)

            # 7b — Child chunks → MongoDB (text + tokens for BM25, NO embedding field)
            child_records = [
                {
                    "chunk_id": child["child_id"],
                    "parent_id": child["parent_id"],
                    "document_id": document_id,
                    "document_name": filename,
                    "text": child["text"],
                    "tokens": re.findall(r'\w+', child["text"].lower()),
                    "page_number": child["page_number"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                for child in children
            ]
            await db.chunks.insert_many(child_records)

            # 7c — Vectors → Qdrant (ANN index for fast vector similarity search)
            qdrant_records = [
                {
                    "chunk_id":      child["child_id"],
                    "parent_id":     child["parent_id"],
                    "document_id":   document_id,
                    "document_name": filename,
                    "page_number":   child["page_number"],
                    "embedding":     embedding,
                }
                for child, embedding in zip(children, embeddings)
            ]
            await asyncio.to_thread(qdrant_service.upsert_vectors, qdrant_records)

            # Update Document record
            await db.documents.update_one(
                {"document_id": document_id},
                {
                    "$set": {
                        "file_hash": file_hash,
                        "total_pages": total_pages,
                        "chunk_count": len(children),
                        "parent_chunk_count": len(parents),
                        "status": "processed",
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                }
            )
            logger.info(
                f"💾 [Database Storage Complete] Inserted: {len(parent_records)} parents, "
                f"{len(child_records)} children | Time: {(time.time() - db_start)*1000:.2f} ms"
            )
            
            total_duration = time.time() - start_time
            logger.info(f"🎉 [Ingestion Success] Completed in {total_duration:.2f} s | File: {filename}")

        except Exception as e:
            total_duration = time.time() - start_time
            logger.error(
                f"❌ [Ingestion Failed] ID: {document_id} | File: {filename} | "
                f"Error: {e} | Time elapsed: {total_duration:.2f} s",
                exc_info=True
            )
            await db.documents.update_one(
                {"document_id": document_id},
                {
                    "$set": {
                        "status": "failed",
                        "error_message": str(e),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                }
            )
        finally:
            # Clean up the file
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"Temporary file '{file_path}' removed.")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file '{file_path}': {e}")


document_processor = DocumentProcessor()

