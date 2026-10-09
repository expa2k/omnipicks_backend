import asyncio
import logging
from datetime import date
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_session_factory, init_db
from app.services.data_ingestion import DataIngestionService
from app.services.prediction_engine import PredictionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

async def run_daily_pipeline():
    logger.info("🚀 Iniciando el Pipeline Diario de OmniPicks...")
    from datetime import timedelta
    today = date.today()
    tomorrow = today + timedelta(days=1)
    
    await init_db()

    factory = get_session_factory()
    async with factory() as session:
        # 1. Ingestar Datos
        ingestion = DataIngestionService(session)
        logger.info("--- PASO 1: Descargando Partidos de Hoy y Mañana ---")
        await ingestion.ingest_daily_games(today)
        await ingestion.ingest_daily_games(tomorrow)
        
        logger.info("--- PASO 2: Descargando Líneas de Casino (Odds) ---")
        await ingestion.ingest_odds()

        # 2. Correr Machine Learning
        engine = PredictionEngine(session)
        logger.info("--- PASO 3: Ejecutando Machine Learning y Calculando Expected Value ---")
        logger.info(f"Running predictions for {today} and {tomorrow}")
        await engine.run_predictions_for_date(today)
        await engine.run_predictions_for_date(tomorrow)

    logger.info("✅ Pipeline completado. Los datos ya están listos en el Frontend.")

if __name__ == "__main__":
    asyncio.run(run_daily_pipeline())
