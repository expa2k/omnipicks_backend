"""
OmniPicks — NBA Advanced Metrics
Fuente: nba_api (stats.nba.com, gratis). Se piden las tablas de liga UNA vez por
corrida (cache con TTL) en lugar de 4 llamadas por partido.
Back-to-backs: scoreboard de ESPN del día anterior (gratis).
"""
import asyncio
import logging
import time
from datetime import date, datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 3 * 60 * 60
_league_cache: Dict[str, tuple] = {}   # season -> (timestamp, payload)


def nba_season_for(d: date) -> str:
    """Temporada NBA en formato nba_api. Arranca en octubre: 2026-11-10 → '2026-27'."""
    start = d.year if d.month >= 10 else d.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def _fetch_league_tables_sync(season: str) -> Dict:
    from nba_api.stats.endpoints import leaguedashteamstats
    from nba_api.stats.static import teams as static_teams

    adv = leaguedashteamstats.LeagueDashTeamStats(
        season=season, measure_type_detailed_defense="Advanced", timeout=30
    ).get_data_frames()[0]
    ff = leaguedashteamstats.LeagueDashTeamStats(
        season=season, measure_type_detailed_defense="Four Factors", timeout=30
    ).get_data_frames()[0]
    last10 = leaguedashteamstats.LeagueDashTeamStats(
        season=season, measure_type_detailed_defense="Advanced", last_n_games=10, timeout=30
    ).get_data_frames()[0]

    # Temporada anterior: sirve como "prior" para no confiar ciegamente en 10 partidos
    start = int(season[:4]) - 1
    prev_season = f"{start}-{(start + 1) % 100:02d}"
    try:
        prev = leaguedashteamstats.LeagueDashTeamStats(
            season=prev_season, measure_type_detailed_defense="Advanced", timeout=30
        ).get_data_frames()[0]
        prev_by_id = {int(r.TEAM_ID): r for r in prev.itertuples()}
    except Exception as e:
        logger.warning(f"nba_api temporada previa {prev_season} no disponible: {e}")
        prev_by_id = {}

    nick = {t["id"]: t["nickname"] for t in static_teams.get_teams()}
    ff_by_id = {int(r.TEAM_ID): r for r in ff.itertuples()}
    l10_by_id = {int(r.TEAM_ID): r for r in last10.itertuples()}

    by_team = {}
    for r in adv.itertuples():
        tid = int(r.TEAM_ID)
        f = ff_by_id.get(tid)
        l10 = l10_by_id.get(tid)
        p = prev_by_id.get(tid)
        by_team[tid] = {
            "prev_off_rating": float(p.OFF_RATING) if p is not None else None,
            "prev_def_rating": float(p.DEF_RATING) if p is not None else None,
            "prev_pace": float(p.PACE) if p is not None else None,
            "name": r.TEAM_NAME,
            "nickname": nick.get(tid, ""),
            "gp": int(r.GP),
            "off_rating": float(r.OFF_RATING),
            "def_rating": float(r.DEF_RATING),
            "net_rating": float(r.NET_RATING),
            "pace": float(r.PACE),
            "efg": float(f.EFG_PCT) if f is not None else None,
            "tov_pct": float(f.TM_TOV_PCT) if f is not None else None,
            "orb_pct": float(f.OREB_PCT) if f is not None else None,
            "ft_rate": float(f.FTA_RATE) if f is not None else None,
            "l10_net_rating": float(l10.NET_RATING) if l10 is not None else None,
        }

    n = max(len(by_team), 1)
    league = {
        "off_rating": sum(t["off_rating"] for t in by_team.values()) / n,
        "pace": sum(t["pace"] for t in by_team.values()) / n,
    }
    # Normalizar la temporada previa al nivel de liga actual (la ofensiva sube año con año)
    prev_teams = [t for t in by_team.values() if t["prev_off_rating"] is not None]
    if prev_teams:
        prev_off_avg = sum(t["prev_off_rating"] for t in prev_teams) / len(prev_teams)
        prev_pace_avg = sum(t["prev_pace"] for t in prev_teams) / len(prev_teams)
        for t in prev_teams:
            t["prev_off_rating"] += league["off_rating"] - prev_off_avg
            t["prev_def_rating"] += league["off_rating"] - prev_off_avg
            t["prev_pace"] += league["pace"] - prev_pace_avg
    return {"teams": by_team, "league": league}


async def get_league_tables(season: str) -> Dict:
    cached = _league_cache.get(season)
    if cached and time.time() - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]
    # nba_api es síncrono: lo corremos en un hilo para no bloquear FastAPI
    payload = await asyncio.to_thread(_fetch_league_tables_sync, season)
    _league_cache[season] = (time.time(), payload)
    logger.info(f"nba_api {season}: {len(payload['teams'])} equipos cargados")
    return payload


def _find_team(tables: Dict, team: Dict) -> Optional[Dict]:
    name = (team.get("name") or "").lower()
    nickname = (team.get("nickname") or "").lower()
    for t in tables["teams"].values():
        if t["name"].lower() == name or (nickname and t["nickname"].lower() == nickname):
            return t
    return None


async def _played_previous_day(team_name: str, game_date: date) -> bool:
    from app.services.espn_nba_client import get_nba_games_espn
    prev = await get_nba_games_espn(game_date - timedelta(days=1), use_cache=True)
    for g in prev:
        if team_name in (g["teams"]["home"]["name"], g["teams"]["away"]["name"]):
            return True
    return False


async def extract_advanced_nba_features(game_data: dict, game_date_str: str) -> dict:
    """Factores reales del partido. Si nba_api falla, marca api_error (el front lo muestra)."""
    teams = game_data.get("teams", {})
    home, away = teams.get("home", {}), teams.get("away", {})
    try:
        game_date = date.fromisoformat(game_date_str) if game_date_str else date.today()
    except ValueError:
        game_date = date.today()

    season = nba_season_for(game_date)
    features: Dict = {
        "home_team_name": home.get("name"),
        "away_team_name": away.get("name"),
        "temporada": season,
        "home_record": home.get("record"),
        "away_record": away.get("record"),
    }

    try:
        tables = await get_league_tables(season)
    except Exception as e:
        logger.error(f"nba_api error ({season}): {e}")
        features["api_error"] = f"No se pudieron obtener estadísticas de NBA ({type(e).__name__})"
        return features

    features["league_off_rating"] = round(tables["league"]["off_rating"], 2)
    features["league_pace"] = round(tables["league"]["pace"], 2)

    for prefix, team in (("home", home), ("away", away)):
        t = _find_team(tables, team)
        if not t:
            features["api_error"] = f"Equipo no encontrado en nba_api: {team.get('name')}"
            return features
        features[f"{prefix}_gp"] = t["gp"]
        for k in ("off_rating", "def_rating", "net_rating", "pace",
                  "efg", "tov_pct", "orb_pct", "ft_rate", "l10_net_rating",
                  "prev_off_rating", "prev_def_rating", "prev_pace"):
            features[f"{prefix}_{k}"] = t[k]

    try:
        features["home_b2b"] = 1.0 if await _played_previous_day(home.get("name"), game_date) else 0.0
        features["away_b2b"] = 1.0 if await _played_previous_day(away.get("name"), game_date) else 0.0
    except Exception as e:
        logger.warning(f"No se pudo calcular back-to-back: {e}")
        features["home_b2b"] = features["away_b2b"] = 0.0

    espn_odds = game_data.get("espn_odds") or {}
    if espn_odds.get("details"):
        features["linea_espn"] = espn_odds["details"]
        features["total_espn"] = espn_odds.get("over_under")

    return features
