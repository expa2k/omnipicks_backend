"""
OmniPicks — NBA v2: tareas programadas (todas gratis, sin cuota).

- Al arrancar (pensado para prender la PC solo cuando se quiere): se pone al día con lo
  que pasó mientras estuvo apagada (box scores, stints y ratings), carga los partidos NBA
  de hoy y mañana desde ESPN y los predice. Nada de esto gasta cuota de pago.
- Lesiones (reporte oficial de la NBA y ESPN): cada 10 minutos en la hora y media antes de
  algún partido y cada 30 el resto del día; si alguno cambió, se vuelven a predecir los
  partidos NBA de hoy y mañana. En cada vuelta también revisa si los ratings son de un
  día anterior (p. ej. la PC estaba dormida a las 04:30) y, si es así, se pone al día.
- Histórico: una vez al día a las 04:30 ET, lo mismo que al arrancar.

Se apaga con NBA_SCHEDULER_ENABLED=false en el .env.
"""
import asyncio
import logging
from datetime import date, datetime, timedelta
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


def today_et() -> date:
    return datetime.now(ZoneInfo(TIMEZONE)).date()


# Las corridas de predicción (NBA y NFL) escriben en las mismas tablas: una a la vez, para no
# provocar deadlocks en MySQL (un deadlock deshace el borrado previo y deja duplicados).
_predict_lock = asyncio.Lock()


async def rerun_nba_predictions(reason: str):
    """Vuelve a predecir los partidos NBA de hoy y mañana (hora ET). Gratis: no usa APIs de pago."""
    from app.db_models import SportType
    from app.services.prediction_engine import PredictionEngine
    today = today_et()
    for day in (today, today + timedelta(days=1)):
        try:
            async with _predict_lock, get_session_factory()() as db:
                await PredictionEngine(db).run_predictions_for_date(day, sports=[SportType.NBA])
            logger.info(f"Predicciones NBA de {day} recalculadas ({reason})")
        except Exception as e:
            logger.warning(f"Predicciones NBA de {day}: no se pudieron recalcular ({type(e).__name__}: {e})")


async def settle_predictions():
    """Liquida las predicciones registradas cuyos partidos ya terminaron."""
    from app.services.nba_tracking import settle
    try:
        async with get_session_factory()() as db:
            return await settle(db)
    except Exception as e:
        logger.warning(f"NBA: no se pudieron liquidar predicciones ({type(e).__name__}: {e})")
        return 0


async def ingest_nba_schedule():
    """Partidos NBA de ayer, hoy y mañana desde ESPN (gratis; no usa API-Sports ni The Odds API).
    Ayer, para actualizar marcadores finales."""
    from app.services.data_ingestion import DataIngestionService
    today = today_et()
    total = 0
    changed = 0
    for day in (today - timedelta(days=1), today, today + timedelta(days=1)):
        try:
            async with get_session_factory()() as db:
                svc = DataIngestionService(db)
                total += await svc._ingest_nba_espn(day)
                changed += svc.odds_changed
        except Exception as e:
            logger.warning(f"Calendario NBA de {day}: no se pudo cargar ({type(e).__name__}: {e})")
    if changed:
        logger.info(f"Líneas NBA (ESPN/DraftKings): {changed} actualizadas")
    ingest_nba_schedule.last_odds_changed = changed
    return total


def ratings_are_stale() -> bool:
    """True si los ratings guardados son de antes de hoy (falta ponerse al día)."""
    from sqlalchemy import create_engine, text
    from app.services.nba_ratings import MODEL_VERSION
    engine = create_engine(get_settings().mysql_url_sync, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            last = conn.execute(text("SELECT MAX(as_of_date) FROM nba_player_ratings WHERE model_version = :v"),
                                {"v": MODEL_VERSION}).scalar()
        return last is None or last < today_et()
    finally:
        engine.dispose()


async def catch_up(reason: str) -> bool:
    """Si los ratings no son de hoy: histórico + ratings, calendario de ESPN y predicciones."""
    if not await asyncio.to_thread(ratings_are_stale):
        return False
    logger.info(f"🏀 Poniéndose al día con la NBA ({reason})...")
    await job_update_history(rerun=False)
    await ingest_nba_schedule()
    await settle_predictions()
    await rerun_nba_predictions(reason)
    return True


async def job_sync_injuries():
    from app.services.nba_injuries import sync_injuries
    try:
        await catch_up("ratings de un día anterior")
    except Exception as e:
        logger.warning(f"NBA: no se pudo poner al día ({type(e).__name__}: {e})")
    try:
        async with get_session_factory()() as db:
            result = await sync_injuries(db)
    except Exception as e:
        logger.warning(f"Lesiones NBA: no se pudo actualizar ({type(e).__name__}: {e})")
        return
    # Calendario, marcadores y líneas de ESPN (gratis): mantiene el mercado al día
    await ingest_nba_schedule()
    odds_changed = getattr(ingest_nba_schedule, "last_odds_changed", 0)
    await settle_predictions()
    # Reporte oficial de la NBA (PDF cada 15 min); si falla, queda ESPN
    official = {}
    try:
        from sqlalchemy import create_engine
        from app.services.nba_official_injuries import sync_official_sync
        eng = create_engine(get_settings().mysql_url_sync, pool_pre_ping=True)
        try:
            official = await asyncio.to_thread(sync_official_sync, eng, today_et())
        finally:
            eng.dispose()
    except Exception as e:
        logger.warning(f"Reporte oficial NBA: no se pudo leer ({type(e).__name__}: {e})")
    # Se re-simula solo si cambió algo en alguno de los dos reportes
    if result.get("new") or result.get("changed") or result.get("cleared") or official.get("changed") or odds_changed:
        await rerun_nba_predictions(f"lesiones: ESPN {result} · oficial {official.get('changed', 0)} cambios · líneas {odds_changed}")


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
                ratings = build_and_store(engine, first_output=None, today=today_et())
                # Probabilidad de jugar (por estado y descanso en back-to-back) con lo nuevo
                try:
                    from app.services.nba_availability import update as update_availability
                    update_availability(engine)
                except Exception as e:
                    logger.warning(f"Disponibilidad NBA: no se pudo recalcular ({type(e).__name__}: {e})")
            finally:
                lock_conn.execute(text("SELECT RELEASE_LOCK(:n)"), {"n": _LOCK_NAME})
        return {"skipped": False, "games": games, "stints_done": done, "stints_failed": failed, "ratings": ratings}
    finally:
        engine.dispose()


async def job_update_history(rerun: bool = True):
    try:
        result = await asyncio.to_thread(update_history_sync)
        logger.info(f"Histórico NBA actualizado: {result}")
        if rerun and not result.get("skipped"):
            await ingest_nba_schedule()
            await rerun_nba_predictions("ratings del día")
    except Exception as e:
        logger.warning(f"Histórico NBA: falló la actualización ({type(e).__name__}: {e})")


async def job_startup():
    global _last_injury_check
    """Al arrancar: ponerse al día si hace falta; si ya estaba al día, al menos cargar y predecir hoy."""
    try:
        if not await catch_up("arranque"):
            if await ingest_nba_schedule():
                await rerun_nba_predictions("arranque")
    except Exception as e:
        logger.warning(f"NBA: falló la puesta al día del arranque ({type(e).__name__}: {e})")
    _last_injury_check = datetime.utcnow()
    await job_sync_injuries()


async def job_nfl():
    """NFL: partidos de la semana desde ESPN (con marcadores y líneas de DraftKings) y predicciones
    de los que faltan por jugar. Gratis: no usa API-Sports ni The Odds API."""
    from sqlalchemy import select, distinct
    from app.db_models import Game, GameStatus, SportType
    from app.services.data_ingestion import DataIngestionService
    from app.services.prediction_engine import PredictionEngine
    try:
        async with get_session_factory()() as db:
            svc = DataIngestionService(db)
            n = await svc.ingest_us_sports_games(today_et(), SportType.NFL)
            days = (await db.execute(select(distinct(Game.game_date)).where(
                Game.sport == SportType.NFL, Game.status == GameStatus.SCHEDULED,
                Game.game_time > datetime.utcnow(), Game.game_time < datetime.utcnow() + timedelta(days=8)))).scalars().all()
        for day in sorted(days):
            async with _predict_lock, get_session_factory()() as db:
                await PredictionEngine(db).run_predictions_for_date(day, sports=[SportType.NFL])
        logger.info(f"🏈 NFL: {n} partidos de la semana; predicciones de {len(days)} días")
    except Exception as e:
        logger.warning(f"NFL: no se pudo actualizar ({type(e).__name__}: {e})")


# Revisión de lesiones: cada 10 min en la hora y media antes de algún partido, cada 30 el resto
NEAR_TIP_MINUTES = 90
NEAR_INTERVAL = 10
FAR_INTERVAL = 30
_last_injury_check: Optional[datetime] = None


def _minutes_to_next_tip(sport: str = "NBA") -> Optional[float]:
    """Minutos hasta el próximo partido programado de ese deporte (hora UTC de la base)."""
    from sqlalchemy import create_engine, text
    engine = create_engine(get_settings().mysql_url_sync, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            nxt = conn.execute(text(
                "SELECT MIN(game_time) FROM games WHERE sport = :s AND status = 'SCHEDULED' AND game_time > UTC_TIMESTAMP()"),
                {"s": sport}).scalar()
        return None if nxt is None else (nxt - datetime.utcnow()).total_seconds() / 60.0
    finally:
        engine.dispose()


async def job_injury_tick():
    """Cada 5 min decide si toca revisar: más seguido cerca del inicio de un partido."""
    global _last_injury_check
    now = datetime.utcnow()
    try:
        to_tip = await asyncio.to_thread(_minutes_to_next_tip)
    except Exception:
        to_tip = None
    every = NEAR_INTERVAL if (to_tip is not None and to_tip <= NEAR_TIP_MINUTES) else FAR_INTERVAL
    if _last_injury_check is not None and (now - _last_injury_check).total_seconds() < every * 60 - 30:
        return
    _last_injury_check = now
    await job_sync_injuries()


async def job_soccer():
    """Fútbol v2: partidos de ayer a 7 días desde ESPN (marcadores y momios) y predicciones."""
    from sqlalchemy import select, distinct
    from app.db_models import Game, GameStatus, SportType
    from app.services.data_ingestion import DataIngestionService
    from app.services.prediction_engine import PredictionEngine
    try:
        today = today_et()
        n = 0
        for k in range(-1, 8):
            async with get_session_factory()() as db:
                n += await DataIngestionService(db).ingest_soccer_espn(today + timedelta(days=k))
        async with get_session_factory()() as db:
            days = (await db.execute(select(distinct(Game.game_date)).where(
                Game.sport == SportType.SOCCER, Game.status == GameStatus.SCHEDULED, Game.external_id.like("espn_%"),
                Game.game_time > datetime.utcnow(), Game.game_time < datetime.utcnow() + timedelta(days=8)))).scalars().all()
        for day in sorted(days):
            async with _predict_lock, get_session_factory()() as db:
                await PredictionEngine(db).run_predictions_for_date(day, sports=[SportType.SOCCER])
        await settle_predictions()
        logger.info(f"⚽ Fútbol: {n} partidos (ayer a 7 días); predicciones de {len(days)} días")
    except Exception as e:
        logger.warning(f"Fútbol: no se pudo actualizar ({type(e).__name__}: {e})")


# NFL: los inactivos oficiales salen 90 min antes del partido. Cada 10 min en las 2 horas previas a
# algún partido; cada hora el resto del tiempo.
NFL_NEAR_MINUTES = 120
NFL_NEAR_INTERVAL = 10
NFL_FAR_INTERVAL = 60
_last_nfl_check: Optional[datetime] = None


async def job_nfl_tick():
    global _last_nfl_check
    now = datetime.utcnow()
    try:
        to_kick = await asyncio.to_thread(_minutes_to_next_tip, "NFL")
    except Exception:
        to_kick = None
    every = NFL_NEAR_INTERVAL if (to_kick is not None and to_kick <= NFL_NEAR_MINUTES) else NFL_FAR_INTERVAL
    if _last_nfl_check is not None and (now - _last_nfl_check).total_seconds() < every * 60 - 30:
        return
    _last_nfl_check = now
    await job_nfl()


async def _nfl_startup():
    global _last_nfl_check
    _last_nfl_check = datetime.utcnow()
    await job_nfl()


def start_nba_scheduler() -> Optional[AsyncIOScheduler]:
    global _scheduler
    if _scheduler or not get_settings().nba_scheduler_enabled:
        return _scheduler
    _scheduler = AsyncIOScheduler(timezone=TIMEZONE)
    common = dict(max_instances=1, coalesce=True, misfire_grace_time=600)
    _scheduler.add_job(job_injury_tick, IntervalTrigger(minutes=5), id="nba_injuries", **common)
    _scheduler.add_job(job_update_history, CronTrigger(hour=4, minute=30), id="nba_history", **common)
    _scheduler.add_job(job_nfl_tick, IntervalTrigger(minutes=5), id="nfl", **common)
    _scheduler.add_job(job_soccer, IntervalTrigger(minutes=60), id="soccer", **common)
    _scheduler.start()
    asyncio.get_running_loop().create_task(job_startup())            # ponerse al día al arrancar
    asyncio.get_running_loop().create_task(_nfl_startup())
    asyncio.get_running_loop().create_task(job_soccer())
    logger.info("🏀 Tareas NBA programadas: lesiones cada 10 min cerca de los partidos (30 min el resto), histórico diario 04:30 ET; NFL cada 10 min en las 2 h previas a un partido (cada hora el resto); fútbol cada hora")
    return _scheduler


def stop_nba_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
