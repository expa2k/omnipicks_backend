import asyncio
from app.services.bzzoiro_client import BzzoiroClient

from app.services.meta_model import SoccerMetaModel
from app.database import get_engine
from sqlalchemy.ext.asyncio import AsyncSession

async def test():
    client = BzzoiroClient()
    engine = get_engine()
    async with AsyncSession(engine) as session:
        meta = SoccerMetaModel(session)
    events = await client.get_events_by_date("2026-06-14")
    for e in events:
        home = str(e.get("home_team", ""))
        away = str(e.get("away_team", ""))
        if "Germany" in home or "Germany" in away:
            print("--- MATCH FOUND ---")
            print(f"Match: {home} vs {away}")
            pred = await client.get_event_prediction(e["id"])
            
            print("\\n--- RAW BZZOIRO DATA ---")
            print(f"Home: {pred.get('markets',{}).get('match_result',{}).get('prob_home')}%")
            print(f"Away: {pred.get('markets',{}).get('match_result',{}).get('prob_away')}%")
            
            final_pred = await meta.adjust_prediction(pred, home, away)
            print("\\n--- OMNIPICKS META-MODEL RESULTS ---")
            import json
            print(json.dumps(final_pred, indent=2, ensure_ascii=False))

asyncio.run(test())
