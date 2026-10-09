import asyncio
from app.services.bzzoiro_client import BzzoiroClient

async def test():
    client = BzzoiroClient()
    leagues = await client.get_leagues()
    league_map = {l.get("id"): l.get("name") for l in leagues if l.get("id")}
    
    events = await client.get_events_by_date("2026-09-09")
    for e in events:
        league_id = e.get("league_id")
        name = league_map.get(league_id, "Unknown")
        print(f"Game: {e.get('home_team', {}).get('name')} vs {e.get('away_team', {}).get('name')} - League: {name}")

if __name__ == "__main__":
    asyncio.run(test())
