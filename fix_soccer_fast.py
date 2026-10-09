import asyncio
import logging
from sqlalchemy import select, delete
from app.database import get_session_factory
from app.db_models import Game, SportType, Pick
from app.services.prediction_engine import PredictionEngine
from app.services.data_ingestion import DataIngestionService
from datetime import date, timedelta

logging.basicConfig(level=logging.INFO)

async def fix_soccer():
    factory = get_session_factory()
    async with factory() as session:
        # Re-ingest games first because they might have been rolled back
        ingestion = DataIngestionService(session)
        await ingestion.ingest_soccer_games(date.today())
        await ingestion.ingest_soccer_games(date.today() + timedelta(days=1))
        
        engine = PredictionEngine(session)
        print("Generating predictions for today...")
        await engine.run_predictions_for_date(date.today())
        print("Generating predictions for tomorrow...")
        await engine.run_predictions_for_date(date.today() + timedelta(days=1))
            
        await session.commit()
        print("Done!")

if __name__ == "__main__":
    asyncio.run(fix_soccer())
