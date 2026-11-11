"""
OmniPicks — NBA v2 (N1): calcula y guarda los ratings de jugador por fecha.

Recorre todo el histórico en orden y guarda, para cada día de partidos, el rating que
tenía cada jugador ANTES de ese día (tabla nba_player_ratings).

Uso:
    python ml/scripts/nba_ratings_build.py                    # todo desde 2023-10-01, más los de hoy
    python ml/scripts/nba_ratings_build.py --from 2026-10-01  # solo desde esa fecha
    python ml/scripts/nba_ratings_build.py --today-only       # solo los ratings vigentes hoy
"""
import argparse
import logging
import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy import create_engine

from app.config import get_settings
from app.services.nba_ratings import build_and_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="first", default="2023-10-01")
    ap.add_argument("--today-only", action="store_true")
    args = ap.parse_args()
    engine = create_engine(get_settings().mysql_url_sync, pool_pre_ping=True)
    first = None if args.today_only else date.fromisoformat(args.first)
    print(build_and_store(engine, first_output=first, today=date.today()))


if __name__ == "__main__":
    main()
