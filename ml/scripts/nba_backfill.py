"""
OmniPicks — NBA v2: carga del histórico (reanudable).

Uso:
    python ml/scripts/nba_backfill.py                       # 2022-23 → temporada actual
    python ml/scripts/nba_backfill.py --seasons 2026-27     # solo actualizar la actual
    python ml/scripts/nba_backfill.py --no-stints           # solo partidos y box scores
    python ml/scripts/nba_backfill.py --rebuild-stints      # rearmar TODOS los stints (usa el crudo en disco si existe)

Se puede interrumpir en cualquier momento: al volver a correr continúa donde se quedó.
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy import bindparam, create_engine, text

from app.config import get_settings
from app.db_models import Base
from app.services.nba_history import current_seasons, load_season, load_stints

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("nba_backfill")

DEFAULT_FIRST_SEASON = 2022


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", default=None)
    ap.add_argument("--no-stints", action="store_true")
    ap.add_argument("--no-games", action="store_true", help="no volver a pedir partidos ni box scores")
    ap.add_argument("--rebuild-stints", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    seasons = args.seasons or current_seasons(DEFAULT_FIRST_SEASON)

    engine = create_engine(get_settings().mysql_url_sync, pool_pre_ping=True, pool_recycle=1800)
    Base.metadata.create_all(engine)

    if not args.no_games:
        for season in seasons:
            load_season(engine, season)
    if args.rebuild_stints:
        with engine.begin() as conn:
            n = conn.execute(text("UPDATE nba_games SET stints_done = 0 WHERE season IN :s")
                             .bindparams(bindparam("s", expanding=True)), {"s": seasons}).rowcount
        logger.info(f"Se rearmarán los stints de {n} partidos")
    if not args.no_stints:
        done, failed = load_stints(engine, seasons, args.workers)
        logger.info(f"✅ Backfill terminado: {done} partidos con stints, {failed} fallidos")


if __name__ == "__main__":
    main()
