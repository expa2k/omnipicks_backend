import asyncio
import logging
from sqlalchemy import select
from app.database import get_session_factory
from app.db_models import Game, OddsSnapshot, SportType, MarketType
from app.services.prediction_engine import PredictionEngine

logging.basicConfig(level=logging.INFO)

async def inject():
    async_session = get_session_factory()
    async with async_session() as db:
        result = await db.execute(select(Game).where(Game.sport == SportType.SOCCER))
        games = result.scalars().all()
        
        for g in games:
            print("Injecting odds for", g.home_team_name, "vs", g.away_team_name)
            snapshot = OddsSnapshot(
                game_id=g.id,
                sport=SportType.SOCCER,
                sportsbook="Bet365",
                market=MarketType.MONEYLINE,
                home_odds=110,
                away_odds=250,
                draw_odds=200
            )
            db.add(snapshot)
        await db.commit()
        
        engine = PredictionEngine(db)
        from datetime import date, timedelta
        await engine.run_predictions_for_date(date.today())
        await engine.run_predictions_for_date(date.today() + timedelta(days=1))
        print("Done generating predictions for soccer!")

asyncio.run(inject())
