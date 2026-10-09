import asyncio
from app.database import get_session_factory
from app.api.routes_predictions import get_value_picks

async def test():
    factory = get_session_factory()
    async with factory() as db:
        try:
            res = await get_value_picks(sport=None, min_value=3, min_edge=2.0, game_date=None, db=db)
            print("Success")
        except Exception as e:
            import traceback
            traceback.print_exc()

asyncio.run(test())
