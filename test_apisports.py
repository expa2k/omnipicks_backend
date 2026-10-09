import asyncio
from app.services.apisports_client import APISportsClient

async def main():
    client = APISportsClient()
    games = await client.get_soccer_games("2026-06-12")
    wc_games = [g for g in games if "World Cup" in g.get("league", {}).get("name", "")]
    print("WORLD CUP GAMES:", len(wc_games))
    for game in wc_games:
        teams = game.get("teams", {})
        print(teams.get("home", {}).get("name"), "VS", teams.get("away", {}).get("name"))

asyncio.run(main())
