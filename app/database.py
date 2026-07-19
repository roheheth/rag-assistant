"""
MongoDB connection management using Motor (async driver).
Handles connection lifecycle and index creation for local MongoDB.
"""

from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class Database:
    """Singleton-style database connection holder."""
    client: AsyncIOMotorClient = None
    db = None


db_instance = Database()


async def connect_to_mongo():
    """Initialize MongoDB connection and create indexes."""
    logger.info(f"Connecting to MongoDB at {settings.MONGODB_URI}")
    db_instance.client = AsyncIOMotorClient(settings.MONGODB_URI)
    db_instance.db = db_instance.client[settings.MONGODB_DB_NAME]

    # Create indexes for efficient queries
    await db_instance.db.chunks.create_index("document_id")
    await db_instance.db.chunks.create_index("parent_id")          # parent-child link
    await db_instance.db.documents.create_index("document_id", unique=True)
    await db_instance.db.documents.create_index("file_hash")         # fast duplicate check
    await db_instance.db.conversations.create_index("conversation_id", unique=True)
    await db_instance.db.parent_chunks.create_index("parent_id", unique=True)
    await db_instance.db.parent_chunks.create_index("document_id")

    # Verify connection
    await db_instance.client.admin.command("ping")
    logger.info("Connected to MongoDB successfully")


async def close_mongo_connection():
    """Gracefully close the MongoDB connection."""
    if db_instance.client:
        db_instance.client.close()
        logger.info("Closed MongoDB connection")


def get_db():
    """Get the database instance. Must be called after connect_to_mongo()."""
    if db_instance.db is None:
        raise RuntimeError("Database not initialized. Call connect_to_mongo() first.")
    return db_instance.db
