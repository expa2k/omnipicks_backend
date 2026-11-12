"""
OmniPicks — ESPN Client (NBA)
Calendario diario de NBA desde la API pública de ESPN (gratis, sin cuota).
El parámetro `dates=YYYYMMDD` de ESPN usa el día local de EE.UU. (ET),
así que la fecha que pedimos es la fecha real del partido.
"""
import httpx
import logging
from datetime import date
from typing import Dict, List

logger = logging.getLogger(__name__)

ESPN_NBA_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

# ESPN status.type.name → código corto que entiende DataIngestionService._map_us_status
_STATUS_MAP = {
    "STATUS_SCHEDULED":   "NS",
    "STATUS_IN_PROGRESS": "live",
    "STATUS_HALFTIME":    "live",
    "STATUS_END_PERIOD":  "live",
    "STATUS_FINAL":       "FT",
    "STATUS_FINAL_OT":    "FT",
    "STATUS_POSTPONED":   "PST",
    "STATUS_CANCELED":    "canc",
}

# Cache por fecha dentro del mismo proceso (el scoreboard de ayer se usa para back-to-backs)
_scoreboard_cache: Dict[str, List[Dict]] = {}


def _num(d: Dict, *path):
    """Número de un camino anidado de ESPN ("-410", "+9.5", "o221.5", "EVEN") o None."""
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    if d is None:
        return None
    s = str(d).strip().lower().lstrip("ou")
    if s in ("even", "ev", "pk", "pick"):
        return 100.0 if path[-1] == "odds" else 0.0
    try:
        return float(s)
    except ValueError:
        return None


async def get_nba_games_espn(target_date: date, use_cache: bool = False) -> List[Dict]:
    """Partidos de NBA de un día, normalizados al formato que usa el resto del backend."""
    key = target_date.strftime("%Y%m%d")
    if use_cache and key in _scoreboard_cache:
        return _scoreboard_cache[key]

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(ESPN_NBA_SCOREBOARD, params={"dates": key})
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.error(f"ESPN NBA scoreboard error ({key}): {e}")
        return []

    games = []
    for event in data.get("events", []):
        try:
            comp = (event.get("competitions") or [{}])[0]
            competitors = comp.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away = next((c for c in competitors if c.get("homeAway") == "away"), {})
            status_name = comp.get("status", {}).get("type", {}).get("name", "STATUS_SCHEDULED")

            def _team(c: Dict) -> Dict:
                t = c.get("team", {})
                records = {r.get("type"): r.get("summary") for r in c.get("records", []) if r.get("type")}
                return {
                    "id": t.get("id"),
                    "name": t.get("displayName", ""),
                    "nickname": t.get("name", ""),
                    "abbreviation": t.get("abbreviation", ""),
                    "logo": t.get("logo") or (t.get("logos") or [{}])[0].get("href", ""),
                    "record": records.get("total"),
                }

            def _score(c: Dict):
                if status_name not in ("STATUS_FINAL", "STATUS_FINAL_OT"):
                    return None
                try:
                    return int(c.get("score"))
                except (TypeError, ValueError):
                    return None

            odds = (comp.get("odds") or [{}])[0]
            ml = odds.get("moneyline") or {}
            ps = odds.get("pointSpread") or {}
            tt = odds.get("total") or {}
            games.append({
                "id": event.get("id"),
                "date": event.get("date", ""),          # ISO UTC, p.ej. 2026-11-11T00:30Z
                "status": {"short": _STATUS_MAP.get(status_name, "NS"), "espn": status_name},
                "venue": {"name": comp.get("venue", {}).get("fullName", "")},
                "teams": {"home": _team(home), "away": _team(away)},
                "scores": {"home": {"total": _score(home)}, "away": {"total": _score(away)}},
                # Línea de ESPN (DraftKings): gratis y siempre disponible; es la fuente principal del mercado NBA
                "espn_odds": {
                    "provider": odds.get("provider", {}).get("name"),
                    "details": odds.get("details"),
                    "spread": odds.get("spread"),
                    "over_under": odds.get("overUnder"),
                    "ml_home": _num(ml, "home", "close", "odds"), "ml_away": _num(ml, "away", "close", "odds"),
                    "ml_home_open": _num(ml, "home", "open", "odds"), "ml_away_open": _num(ml, "away", "open", "odds"),
                    "spread_home_line": _num(ps, "home", "close", "line"),
                    "spread_home_odds": _num(ps, "home", "close", "odds"), "spread_away_odds": _num(ps, "away", "close", "odds"),
                    "spread_home_open": _num(ps, "home", "open", "line"),
                    "total_line": _num(tt, "over", "close", "line"), "over_odds": _num(tt, "over", "close", "odds"),
                    "under_odds": _num(tt, "under", "close", "odds"), "total_open": _num(tt, "over", "open", "line"),
                    "favorite": "home" if (odds.get("homeTeamOdds") or {}).get("favorite") else
                                "away" if (odds.get("awayTeamOdds") or {}).get("favorite") else None,
                } if odds else None,
                "_source": "espn",
            })
        except Exception as e:
            logger.warning(f"ESPN NBA event parse error: {e}")

    _scoreboard_cache[key] = games
    logger.info(f"ESPN NBA: {len(games)} games for {target_date.isoformat()}")
    return games
