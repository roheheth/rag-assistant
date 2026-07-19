import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def clear():
    client = AsyncIOMotorClient('mongodb://127.0.0.1:27017')
    db = client['rag_app']
    await db.chunks.drop()
    await db.parent_chunks.drop()
    await db.documents.drop()
    print('All MongoDB collections cleared! Ready for Qdrant migration.')

asyncio.run(clear())
