import asyncio
from datetime import date
from app.database import get_session_factory
from app.services.data_ingestion import DataIngestionService

async def main():
    factory = get_session_factory()
    async with factory() as session:
        service = DataIngestionService(session)
        print("DataIngestionService methods:", dir(service))

asyncio.run(main())
