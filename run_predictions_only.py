import asyncio
import logging
from app.database import get_session_factory
from app.services.prediction_engine import PredictionEngine

logging.basicConfig(level=logging.INFO)

from datetime import datetime, timedelta

async def main():
    factory = get_session_factory()
    async with factory() as session:
        engine = PredictionEngine(session)
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        await engine.run_predictions_for_date(today)
        await engine.run_predictions_for_date(tomorrow)
    print("Done!")

asyncio.run(main())
