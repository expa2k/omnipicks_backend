import asyncio
from app.services.odds_client import OddsClient

async def main():
    client = OddsClient()
    odds = await client.get_odds("soccer_fifa_world_cup")
    print("WORLD CUP ODDS:", len(odds))
    for game in odds:
        print(game.get("home_team"), "VS", game.get("away_team"))

asyncio.run(main())
