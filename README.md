<div align="center">

# RAG Assistant

**A production-grade, hybrid-search Retrieval-Augmented Generation (RAG) application built with FastAPI, Qdrant, MongoDB, and Streamlit.**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-7.0+-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-1.18+-FF4444?style=for-the-badge&logo=qdrant&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

Upload documents, ask questions, and receive answers grounded in your own files — not hallucinated from thin air.

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| **Multi-format Ingestion** | Upload PDF, Word (`.docx`), and Excel (`.xlsx`) files |
| **Async Background Processing** | UI unlocks instantly on upload; indexing runs silently in the background |
| **Hybrid Search** | BM25 keyword search + Qdrant vector search, fused with Reciprocal Rank Fusion (RRF) |
| **Parent-Child Chunking** | Small child chunks for precision retrieval, large parent chunks for rich LLM context |
| **MD5 Deduplication** | Uploading the same file twice is detected and skipped in under 0.01 seconds |
| **Query Intent Router** | Zero-shot classifier routes questions to either the knowledge base or conversational mode |
| **Streaming Answers** | LLM responses stream token-by-token via SSE for a ChatGPT-like experience |
| **Auto Text Compression** | Structural cleaning removes page numbers, duplicate spaces, and empty lines on ingestion |
| **Detailed Telemetry Logs** | Every pipeline stage (extraction, cleaning, chunking, embedding) is timed and logged |
| **Externalised Config** | All secrets live in `.env`, never in source code |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Streamlit UI (Port 8501)                  │
│   Sidebar: File Upload + Document Status  │  Chat: SSE Stream   │
└───────────────────────┬─────────────────────────────────────────┘
                        │ HTTP / SSE
┌───────────────────────▼─────────────────────────────────────────┐
│                    FastAPI Backend (Port 8000)                   │
│                                                                 │
│  POST /api/upload  →  BackgroundTask: process_document_async()  │
│  POST /api/ask     →  RAGPipeline.ask_stream()                  │
│  GET  /api/documents, DELETE /api/documents/{id}                │
└──────────┬──────────────────────┬───────────────────────────────┘
           │                      │
   ┌───────▼──────┐      ┌────────▼────────┐
   │   MongoDB    │      │     Qdrant       │
   │              │      │  (Vector Store)  │
   │  documents   │      │                 │
   │  parent_chunks│     │  collection:    │
   │  chunks      │      │  "rag_chunks"   │
   │  (text+tokens│      │  (384-dim HNSW) │
   │   no vectors)│      │                 │
   └──────────────┘      └─────────────────┘
```

### Search Flow (Per Question)
```
User Question
    │
    ▼
[1] Query Router (Zero-Shot Classification)
    ├── "casual conversation"  → LLM answers directly (no DB search)
    └── "document question"    →
            │
            ▼
        [2] Embed query via BAAI/bge-small-en-v1.5
            │
            ├──[3a] Qdrant ANN Search  → Top-K vector matches (chunk_ids + scores)
            └──[3b] BM25 Keyword Search → MongoDB tokens → keyword scores
                        │
                        ▼
                [4] Reciprocal Rank Fusion (RRF)
                        │
                        ▼
                [5] Fetch Parent Chunks from MongoDB
                        │
                        ▼
                [6] Stream answer via Qwen 2.5 7B LLM
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- MongoDB running locally on port `27017`
- A Hugging Face API token (free tier works)

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/rag-assistant.git
cd rag-assistant
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Copy the example env file and fill in your values:
```bash
cp .env.example .env
```

Open `.env` and set your Hugging Face API token:
```env
HF_API_TOKEN=hf_your_token_here
```

### 5. Run the Application

Start the FastAPI backend:
```bash
uvicorn app.main:app --reload
```

In a second terminal, start the Streamlit UI:
```bash
streamlit run streamlit_app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 📦 Tech Stack

### Backend
| Library | Purpose |
|---|---|
| **FastAPI** | Async REST API framework |
| **Motor** | Async MongoDB driver |
| **Qdrant Client** | Local vector store with HNSW index |
| **PyMuPDF (fitz)** | High-speed PDF text extraction |
| **Mammoth** | Clean DOCX → HTML conversion |
| **python-calamine** | Rust-powered Excel parsing |
| **rank-bm25** | BM25 keyword search |
| **LangChain Text Splitters** | Recursive parent-child chunking |
| **Hugging Face Hub** | Embedding model + LLM inference API |
| **Pydantic Settings** | Externalised configuration management |

### Frontend
| Library | Purpose |
|---|---|
| **Streamlit** | Interactive chat UI with SSE streaming |

### Models
| Model | Role |
|---|---|
| `BAAI/bge-small-en-v1.5` | Text embedding (384-dim, L2-normalized) |
| `Qwen/Qwen2.5-7B-Instruct` | Answer generation (streaming) |
| HF Zero-Shot Classifier | Query intent routing |

---

## 📁 Project Structure

```
rag-app/
├── app/
│   ├── config.py              # Pydantic settings (reads from .env)
│   ├── database.py            # MongoDB async connection
│   ├── qdrant_service.py      # Qdrant vector store (init, upsert, search)
│   ├── embeddings.py          # HF embedding service with batching
│   ├── document_processor.py  # Parse → Clean → Chunk → Embed → Store
│   ├── retriever.py           # Hybrid search: Qdrant + BM25 + RRF
│   ├── llm_service.py         # LLM streaming, summarisation, intent router
│   ├── rag_pipeline.py        # Orchestrator: question → streamed answer
│   ├── context_manager.py     # Rolling chat context (MongoDB)
│   ├── models.py              # Pydantic request/response models
│   └── main.py                # FastAPI routes and app lifecycle
├── streamlit_app.py           # Streamlit frontend
├── compress_pdf.py            # Optional: offline PDF pre-compressor
├── clear_db.py                # Utility: wipe all MongoDB collections
├── requirements.txt
├── .env.example               # Template — copy to .env and fill in secrets
└── README.md
```

---

## ⚙️ Configuration Reference

All settings are loaded from `.env`. See `.env.example` for the full list.

| Variable | Default | Description |
|---|---|---|
| `HF_API_TOKEN` | _(required)_ | Hugging Face API token |
| `MONGODB_URI` | `mongodb://127.0.0.1:27017` | MongoDB connection string |
| `MONGODB_DB_NAME` | `rag_app` | Database name |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HF embedding model |
| `EMBEDDING_DIMENSION` | `384` | Must match the model's output size |
| `LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | HF LLM model |
| `QDRANT_PATH` | `./qdrant_storage` | Local Qdrant data directory |
| `QDRANT_COLLECTION` | `rag_chunks` | Qdrant collection name |
| `TOP_K` | `5` | Number of parent chunks to return per search |
| `SIMILARITY_THRESHOLD` | `0.5` | Minimum cosine similarity to include a chunk |
| `PARENT_CHUNK_SIZE` | `1000` | Characters per parent chunk |
| `CHILD_CHUNK_SIZE` | `400` | Characters per child chunk |

---

## 🔄 Document Ingestion Pipeline

When you upload a file, the following steps run **asynchronously in the background**:

```
1. File saved temporarily to disk
2. MD5 hash calculated → duplicate check in MongoDB
   └── If duplicate → skip everything, finish in 0.01s
3. Text extraction (PyMuPDF / Mammoth / Calamine)
4. Structural cleaning (strip page numbers, collapse whitespace)
5. Parent-Child chunking (LangChain RecursiveCharacterTextSplitter)
6. Embedding via BAAI/bge-small (concurrent batches of 8)
7. Store:
   ├── MongoDB: document metadata + parent chunks + child text/tokens
   └── Qdrant: child chunk embeddings (384-dim HNSW indexed)
8. Temporary file deleted
```

---

## 🧪 Edge Case Tests

| Test | Expected Behavior |
|---|---|
| Upload duplicate file | Detected by MD5 hash, skipped in `<0.01s` |
| Scanned PDF (no text) | Fails gracefully with `❌ (Failed)` status |
| Ask out-of-scope question | Intent router answers conversationally |
| Force document question on off-topic | Returns "not found in documents" |
| Upload 3 files simultaneously | All processed concurrently via async event loop |

---

<div align="center">
Built with ❤️ using FastAPI, Qdrant, MongoDB, and Streamlit.
</div>
