"""
OmniPicks — ESPN Client (NFL + Soccer)
Fuente de datos ESPN: schedule actual + estadísticas de temporada.
ESPN API pública — sin restricciones de plan, datos en tiempo real.
"""
import httpx
import logging
from typing import Dict, List, Any, Optional
from datetime import date

logger = logging.getLogger(__name__)

# Cache para evitar llamadas repetidas en el mismo ciclo
_stats_cache: Dict[str, Any] = {}

# ─── ESPN BASE URLS ───────────────────────────────────────────────────────────
ESPN_SITE = "https://site.api.espn.com/apis/site/v2/sports"
ESPN_CORE = "https://sports.core.api.espn.com/v2/sports"

# ─── NFL TEAM MAP ─────────────────────────────────────────────────────────────
# ESPN team id → display name (abbr for matching)
NFL_ESPN_TEAMS: Dict[str, int] = {
    "Arizona Cardinals": 22, "Atlanta Falcons": 1,   "Baltimore Ravens": 33,
    "Buffalo Bills": 2,      "Carolina Panthers": 29, "Chicago Bears": 3,
    "Cincinnati Bengals": 4, "Cleveland Browns": 5,   "Dallas Cowboys": 6,
    "Denver Broncos": 7,     "Detroit Lions": 8,      "Green Bay Packers": 9,
    "Houston Texans": 34,    "Indianapolis Colts": 11,"Jacksonville Jaguars": 30,
    "Kansas City Chiefs": 12,"Las Vegas Raiders": 13, "Los Angeles Chargers": 24,
    "Los Angeles Rams": 14,  "Miami Dolphins": 15,    "Minnesota Vikings": 16,
    "New England Patriots": 17,"New Orleans Saints": 18,"New York Giants": 19,
    "New York Jets": 20,     "Philadelphia Eagles": 21,"Pittsburgh Steelers": 23,
    "San Francisco 49ers": 25,"Seattle Seahawks": 26, "Tampa Bay Buccaneers": 27,
    "Tennessee Titans": 10,  "Washington Commanders": 28,
}
NFL_ESPN_ID_TO_NAME = {v: k for k, v in NFL_ESPN_TEAMS.items()}

# ─── SOCCER ESPN LEAGUE SLUGS ────────────────────────────────────────────────
SOCCER_ESPN_LEAGUES = {
    "UEFA Champions League": "uefa.champions",
    "UEFA Europa League":    "uefa.europa",
    "Premier League":        "eng.1",
    "La Liga":               "esp.1",
    "Serie A":               "ita.1",
    "Bundesliga":            "ger.1",
    "Ligue 1":               "fra.1",
    "Liga MX":               "mex.1",
    "World Cup":             "fifa.world",
    "Copa America":          "conmebol.america",
    "Euro Championship":     "uefa.euro",
}


# ═══════════════════════════════════════════════════════════════════════════════
# NFL
# ═══════════════════════════════════════════════════════════════════════════════

async def get_nfl_games_espn(target_date: Optional[date] = None) -> List[Dict]:
    """
    Obtiene los partidos de NFL de la semana actual desde ESPN.
    Si se pasa target_date, filtra por esa fecha; si no, devuelve toda la semana.
    """
    url = f"{ESPN_SITE}/football/nfl/scoreboard"
    params = {}
    if target_date:
        params["dates"] = target_date.strftime("%Y%m%d")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.error(f"ESPN NFL scoreboard error: {e}")
        return []

    games = []
    for event in data.get("events", []):
        try:
            comp = event.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away = next((c for c in competitors if c.get("homeAway") == "away"), {})

            status_name = comp.get("status", {}).get("type", {}).get("name", "STATUS_SCHEDULED")
            status_map = {
                "STATUS_SCHEDULED":  "NS",
                "STATUS_IN_PROGRESS": "P",
                "STATUS_FINAL":       "FT",
                "STATUS_POSTPONED":   "PST",
            }

            home_team = home.get("team", {})
            away_team = away.get("team", {})

            home_score = None
            away_score = None
            if status_name == "STATUS_FINAL":
                home_score = int(home.get("score", 0) or 0)
                away_score = int(away.get("score", 0) or 0)

            game_iso = event.get("date", "")
            games.append({
                "game": {
                    "id": event.get("id"),
                    "date": {
                        "date": game_iso.split("T")[0],
                        "time": game_iso,
                    },
                    "status": {"short": status_map.get(status_name, "NS")},
                    "venue": {"name": comp.get("venue", {}).get("fullName", "")},
                },
                "teams": {
                    "home": {
                        "id":   int(home_team.get("id", 0)),
                        "name": home_team.get("displayName", ""),
                        "logo": (home_team.get("logos") or [{}])[0].get("href", ""),
                    },
                    "away": {
                        "id":   int(away_team.get("id", 0)),
                        "name": away_team.get("displayName", ""),
                        "logo": (away_team.get("logos") or [{}])[0].get("href", ""),
                    },
                },
                "scores": {
                    "home": {"total": home_score},
                    "away": {"total": away_score},
                },
                "_source": "espn",
            })
        except Exception as e:
            logger.warning(f"ESPN NFL event parse error: {e}")
    logger.info(f"ESPN NFL: {len(games)} games found")
    return games


async def get_nfl_team_stats_espn(espn_team_id: int) -> Dict[str, float]:
    """
    Obtiene estadísticas de temporada de un equipo NFL desde ESPN Core API.
    Intenta temporada actual primero; si está vacía (0s), usa la anterior.
    Returns flat dict con métricas del algoritmo.
    """
    current_year = date.today().year

    for season in [current_year, current_year - 1]:
        cache_key = f"espn_nfl_{espn_team_id}_{season}"
        if cache_key in _stats_cache:
            cached = _stats_cache[cache_key]
            if cached:
                return cached
            continue  # empty cache → try next season

        url = f"{ESPN_CORE}/football/leagues/nfl/seasons/{season}/types/2/teams/{espn_team_id}/statistics"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(url)
                if r.status_code == 404:
                    _stats_cache[cache_key] = {}
                    continue
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            logger.error(f"ESPN NFL stats error team {espn_team_id} season {season}: {e}")
            _stats_cache[cache_key] = {}
            continue

        metrics = _parse_espn_nfl_stats(data)

        # Check if season actually has data (gamesPlayed > 0)
        games_played = metrics.get("general_gamesPlayed", 0)
        _stats_cache[cache_key] = metrics

        if games_played > 0:
            logger.info(f"ESPN NFL: team {espn_team_id} season {season} — {int(games_played)} games played")
            return metrics
        else:
            logger.info(f"ESPN NFL: season {season} empty for team {espn_team_id}, trying {season-1}")

    return {}


def _parse_espn_nfl_stats(data: Dict) -> Dict[str, float]:
    """Flattens ESPN stats categories into a dict keyed as category_statName."""
    metrics: Dict[str, float] = {}
    for cat in data.get("splits", {}).get("categories", []):
        cat_name = cat.get("name", "")
        for stat in cat.get("stats", []):
            name = stat.get("name", "")
            value = stat.get("value")
            if value is not None:
                try:
                    metrics[f"{cat_name}_{name}"] = float(value)
                except (ValueError, TypeError):
                    pass
    return metrics


def get_espn_nfl_id(team_name: str) -> Optional[int]:
    """Maps an NFL team name to its ESPN ID (direct + partial match)."""
    if team_name in NFL_ESPN_TEAMS:
        return NFL_ESPN_TEAMS[team_name]
    name_lower = team_name.lower()
    for name, espn_id in NFL_ESPN_TEAMS.items():
        if name_lower in name.lower() or name.lower() in name_lower:
            return espn_id
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SOCCER
# ═══════════════════════════════════════════════════════════════════════════════

async def get_soccer_games_espn(league_slug: str, target_date: Optional[date] = None) -> List[Dict]:
    """Obtiene partidos de fútbol de una liga específica desde ESPN."""
    url = f"{ESPN_SITE}/soccer/{league_slug}/scoreboard"
    params = {}
    if target_date:
        params["dates"] = target_date.strftime("%Y%m%d")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return r.json().get("events", [])
    except Exception as e:
        logger.error(f"ESPN Soccer [{league_slug}] scoreboard error: {e}")
        return []


async def get_soccer_team_stats_espn(
    espn_team_id: int,
    league_slug: str,
    league_api_name: str,
) -> Dict[str, float]:
    """
    Obtiene estadísticas de temporada de un equipo de fútbol desde ESPN Core API.
    Intenta temporada actual → si no tiene datos → fallback temporada anterior.
    """
    current_year = date.today().year

    for season in [current_year, current_year - 1]:
        cache_key = f"espn_soccer_{league_slug}_{espn_team_id}_{season}"
        if cache_key in _stats_cache:
            cached = _stats_cache[cache_key]
            if cached:
                return cached
            continue

        url = f"{ESPN_CORE}/soccer/leagues/{league_api_name}/seasons/{season}/teams/{espn_team_id}/statistics"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(url)
                if r.status_code == 404:
                    _stats_cache[cache_key] = {}
                    continue
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            logger.error(f"ESPN Soccer stats error team {espn_team_id} season {season}: {e}")
            _stats_cache[cache_key] = {}
            continue

        metrics = _parse_espn_soccer_stats(data)
        games_played = metrics.get("general_gamesPlayed", metrics.get("general_games", 0))
        _stats_cache[cache_key] = metrics

        if games_played > 0:
            logger.info(f"ESPN Soccer: team {espn_team_id} league {league_slug} season {season} — {int(games_played)} games")
            return metrics
        else:
            logger.info(f"ESPN Soccer: season {season} empty for team {espn_team_id}, trying {season-1}")

    return {}


def _parse_espn_soccer_stats(data: Dict) -> Dict[str, float]:
    """Flattens ESPN soccer stats into a flat dict."""
    metrics: Dict[str, float] = {}
    for cat in data.get("splits", {}).get("categories", []):
        cat_name = cat.get("name", "")
        for stat in cat.get("stats", []):
            name = stat.get("name", "")
            value = stat.get("value")
            if value is not None:
                try:
                    metrics[f"{cat_name}_{name}"] = float(value)
                except (ValueError, TypeError):
                    pass
    return metrics
