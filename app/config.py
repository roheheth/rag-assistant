from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings): 
    HF_API_TOKEN: str = ""

    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "rag_app"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    LLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"

    # ── Qdrant Vector Store ───────────────────────────────────────
    QDRANT_PATH: str = "./qdrant_storage"
    QDRANT_COLLECTION: str = "rag_chunks"

    PARENT_CHUNK_SIZE: int = 2000
    PARENT_CHUNK_OVERLAP: int = 200
    CHILD_CHUNK_SIZE: int = 500
    CHILD_CHUNK_OVERLAP: int = 75

    # ── Retrieval ─────────────────────────────────────────────────
    TOP_K: int = 10
    SIMILARITY_THRESHOLD: float = 0.5

    # ── Generation ────────────────────────────────────────────────
    FULL_ANSWER_MAX_TOKENS: int = 512
    SUMMARY_MAX_TOKENS: int = 256
    CONTEXT_SUMMARY_MAX_TOKENS: int = 300

    # ── App ───────────────────────────────────────────────────────
    UPLOAD_DIR: str = "uploads"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
