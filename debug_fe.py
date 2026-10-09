import asyncio
from app.database import get_session_factory
from app.services.feature_engineering import FeatureEngineer
from sqlalchemy import select
from app.db_models import Game

async def main():
    factory = get_session_factory()
    async with factory() as db:
        result = await db.execute(select(Game).where(Game.home_team_name.ilike('%Giants%'), Game.game_date == '2026-06-12').limit(1))
        game = result.scalar_one_or_none()
        if not game:
            print("No game found")
            return
        print(f"Game: {game.away_team_name} @ {game.home_team_name} on {game.game_date}")
        
        fe = FeatureEngineer()
        features = fe.compute_features(game.raw_data, game.sport.value, game.game_date.isoformat())
        print("Features:", features)

asyncio.run(main())
