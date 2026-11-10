"""
OmniPicks — NBA v2: carga del histórico (reanudable).

Uso:
    python ml/scripts/nba_backfill.py                       # 2022-23 → temporada actual
    python ml/scripts/nba_backfill.py --seasons 2026-27     # solo actualizar la actual
    python ml/scripts/nba_backfill.py --no-stints           # solo partidos y box scores

Se puede interrumpir en cualquier momento: al volver a correr continúa donde se quedó.
"""
import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.dialects.mysql import insert as mysql_insert

from app.config import get_settings
from app.db_models import Base, NBAGame, NBAPlayerGame, NBAStint
from app.services.nba_history import SEASON_TYPES, build_stints, fetch_season, season_of

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("nba_backfill")

DEFAULT_FIRST_SEASON = 2022


def default_seasons():
    last = int(season_of(date.today())[:4])
    return [f"{y}-{(y + 1) % 100:02d}" for y in range(DEFAULT_FIRST_SEASON, last + 1)]


def chunks(rows, n=1000):
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


def load_season(engine, season: str):
    for season_type in SEASON_TYPES:
        games, player_games = fetch_season(season, season_type)
        if not games:
            logger.info(f"{season} {season_type}: sin partidos")
            continue
        with engine.begin() as conn:
            for part in chunks(games):
                stmt = mysql_insert(NBAGame).values(part)
                conn.execute(stmt.on_duplicate_key_update(
                    home_pts=stmt.inserted.home_pts, away_pts=stmt.inserted.away_pts))
            for part in chunks(player_games):
                conn.execute(mysql_insert(NBAPlayerGame).values(part).prefix_with("IGNORE"))
        logger.info(f"{season} {season_type}: {len(games)} partidos, {len(player_games)} filas de jugador")


def load_stints(engine, seasons, workers: int):
    with engine.connect() as conn:
        pending = conn.execute(text(
            "SELECT game_id, home_team_id FROM nba_games "
            "WHERE stints_done = 0 AND home_pts IS NOT NULL AND season IN :s ORDER BY game_date"
        ).bindparams(bindparam("s", expanding=True)), {"s": seasons}).fetchall()

    total = len(pending)
    logger.info(f"Stints pendientes: {total} partidos")
    done = failed = 0
    start = time.time()

    def work(row):
        return row.game_id, build_stints(row.game_id, row.home_team_id)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, r): r for r in pending}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                game_id, stints = fut.result()
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM nba_stints WHERE game_id = :g"), {"g": game_id})
                    if stints:
                        conn.execute(NBAStint.__table__.insert(), stints)
                    conn.execute(text("UPDATE nba_games SET stints_done = 1 WHERE game_id = :g"), {"g": game_id})
                done += 1
            except Exception as e:
                failed += 1
                logger.warning(f"Partido {row.game_id} falló ({type(e).__name__}: {str(e)[:80]}); se reintenta en la próxima corrida")
            n = done + failed
            if n % 50 == 0 or n == total:
                rate = n / max(time.time() - start, 1)
                eta = (total - n) / max(rate, 0.01) / 60
                logger.info(f"Stints {n}/{total} (ok {done}, fallidos {failed}) · faltan ~{eta:.0f} min")
    return done, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", default=None)
    ap.add_argument("--no-stints", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    seasons = args.seasons or default_seasons()

    engine = create_engine(get_settings().mysql_url_sync, pool_pre_ping=True, pool_recycle=1800)
    Base.metadata.create_all(engine)

    for season in seasons:
        load_season(engine, season)
    if not args.no_stints:
        done, failed = load_stints(engine, seasons, args.workers)
        logger.info(f"✅ Backfill terminado: {done} partidos con stints, {failed} fallidos")


if __name__ == "__main__":
    main()
