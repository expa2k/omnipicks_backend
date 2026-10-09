import asyncio
import logging
from datetime import date
from sqlalchemy import select

from app.database import init_db, close_db, get_session_factory
from app.db_models import Game, SportType, Prediction
from app.services.bzzoiro_client import BzzoiroClient
from app.services.data_ingestion import DataIngestionService
from app.services.prediction_engine import PredictionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

async def test_bzzoiro_soccer():
    await init_db()
    
    target_date = date.today()
    logging.info(f"--- TESTING BZZOIRO FOR {target_date} ---")
    
    factory = get_session_factory()
    async with factory() as session:
        # Clear existing SOCCER games for today to ensure a clean test
        from sqlalchemy import delete
        await session.execute(delete(Game).where(Game.sport == SportType.SOCCER))
        await session.execute(delete(Prediction).where(Prediction.sport == SportType.SOCCER))
        await session.commit()
        
        # Ingestion
        ingestion = DataIngestionService(session)
        count_today = await ingestion.ingest_soccer_games(target_date)
        logging.info(f"Ingested {count_today} SOCCER games via Bzzoiro for today!")
        
        from datetime import timedelta
        tomorrow = target_date + timedelta(days=1)
        count_tomorrow = await ingestion.ingest_soccer_games(tomorrow)
        logging.info(f"Ingested {count_tomorrow} SOCCER games via Bzzoiro for tomorrow!")
        
        # Verify games in DB
        result = await session.execute(select(Game).where(Game.sport == SportType.SOCCER))
        games = result.scalars().all()
        for g in games:
            logging.info(f"Game DB Entry: {g.home_team_name} vs {g.away_team_name} in {g.league} at {g.game_time}")
            
        # Prediction
        engine = PredictionEngine(session)
        await engine.run_predictions_for_date(target_date)
        await engine.run_predictions_for_date(tomorrow)
        
        # Verify predictions in DB
        pred_result = await session.execute(select(Prediction).where(Prediction.sport == SportType.SOCCER))
        preds = pred_result.scalars().all()
        for p in preds:
            logging.info(f"Prediction DB Entry for Game {p.game_id}: Home {p.home_win_prob} | Away {p.away_win_prob} | xG {p.predicted_home_score}-{p.predicted_away_score}")
            
    await close_db()

if __name__ == "__main__":
    asyncio.run(test_bzzoiro_soccer())
