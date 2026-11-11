"""
OmniPicks — NBA v2: actualización diaria a mano (lo mismo que hace el servidor a las 04:30 ET).

Uso:
    python ml/scripts/nba_daily_update.py              # histórico + lesiones
    python ml/scripts/nba_daily_update.py --injuries   # solo lesiones
"""
import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.database import close_db
from app.services.nba_scheduler import job_sync_injuries, update_history_sync

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


async def _injuries():
    await job_sync_injuries()
    await close_db()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--injuries", action="store_true")
    args = ap.parse_args()
    if not args.injuries:
        print(update_history_sync())
    asyncio.run(_injuries())


if __name__ == "__main__":
    main()
