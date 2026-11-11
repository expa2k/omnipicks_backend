"""
OmniPicks — NBA v2: tareas programadas (todas gratis, sin cuota).

- Lesiones de ESPN: cada 30 minutos; si el reporte cambió, se vuelven a predecir los
  partidos NBA de hoy.
- Histórico: una vez al día de madrugada (partidos, box scores y stints de ayer) y,
  con eso, los ratings de jugador vigentes para hoy.

Se apaga con NBA_SCHEDULER_ENABLED=false en el .env.
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.config import get_settings
from app.database import get_session_factory

logger = logging.getLogger(__name__)

TIMEZONE = "America/New_York"
_scheduler: Optional[AsyncIOScheduler] = None
_LOCK_NAME = "omnipicks_nba_history"


async def rerun_nba_predictions(reason: str):
    """Vuelve a predecir los partidos NBA de hoy (hora ET). Gratis: no usa APIs de pago."""
    from app.db_models import SportType
    from app.services.prediction_engine import PredictionEngine
    today = datetime.now(ZoneInfo(TIMEZONE)).date()
    try:
        async with get_session_factory()() as db:
            await PredictionEngine(db).run_predictions_for_date(today, sports=[SportType.NBA])
        logger.info(f"Predicciones NBA de {today} recalculadas ({reason})")
    except Exception as e:
        logger.warning(f"Predicciones NBA: no se pudieron recalcular ({type(e).__name__}: {e})")


async def job_sync_injuries():
    from app.services.nba_injuries import sync_injuries
    try:
        async with get_session_factory()() as db:
            result = await sync_injuries(db)
    except Exception as e:
        logger.warning(f"Lesiones NBA: no se pudo actualizar ({type(e).__name__}: {e})")
        return
    # Se re-simula solo si cambió algo en el reporte
    if result.get("new") or result.get("changed") or result.get("cleared"):
        await rerun_nba_predictions(f"lesiones: {result}")


def update_history_sync() -> dict:
    """Trae lo nuevo de la temporada en curso. Bloqueante: se corre en un hilo o por script."""
    from sqlalchemy import create_engine, text
    from app.services.nba_history import current_seasons, load_season, load_stints
    from app.services.nba_ratings import build_and_store

    engine = create_engine(get_settings().mysql_url_sync, pool_pre_ping=True, pool_recycle=1800)
    try:
        # Candado en MySQL: evita que dos cargas (p. ej. el backfill y esta) se pisen
        with engine.connect() as lock_conn:
            if not lock_conn.execute(text("SELECT GET_LOCK(:n, 0)"), {"n": _LOCK_NAME}).scalar():
                logger.info("Histórico NBA: ya hay otra carga corriendo; se omite")
                return {"skipped": True}
            try:
                seasons = current_seasons()
                games = sum(load_season(engine, s) for s in seasons)
                done, failed = load_stints(engine, seasons)
                # Ratings de jugador vigentes hoy, ya con los partidos de ayer
                ratings = build_and_store(engine, first_output=None, today=datetime.now(ZoneInfo(TIMEZONE)).date())
            finally:
                lock_conn.execute(text("SELECT RELEASE_LOCK(:n)"), {"n": _LOCK_NAME})
        return {"skipped": False, "games": games, "stints_done": done, "stints_failed": failed, "ratings": ratings}
    finally:
        engine.dispose()


async def job_update_history():
    try:
        result = await asyncio.to_thread(update_history_sync)
        logger.info(f"Histórico NBA actualizado: {result}")
        if not result.get("skipped"):
            await rerun_nba_predictions("ratings del día")
    except Exception as e:
        logger.warning(f"Histórico NBA: falló la actualización ({type(e).__name__}: {e})")


def start_nba_scheduler() -> Optional[AsyncIOScheduler]:
    global _scheduler
    if _scheduler or not get_settings().nba_scheduler_enabled:
        return _scheduler
    _scheduler = AsyncIOScheduler(timezone=TIMEZONE)
    common = dict(max_instances=1, coalesce=True, misfire_grace_time=600)
    _scheduler.add_job(job_sync_injuries, IntervalTrigger(minutes=30), id="nba_injuries", **common)
    _scheduler.add_job(job_update_history, CronTrigger(hour=4, minute=30), id="nba_history", **common)
    _scheduler.start()
    asyncio.get_running_loop().create_task(job_sync_injuries())      # primera lectura al arrancar
    logger.info("🏀 Tareas NBA programadas: lesiones cada 30 min, histórico diario 04:30 ET")
    return _scheduler


def stop_nba_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
