"""
OmniPicks — Soccer Advanced Metrics (Nivel 1-5)
Extrae métricas REALES de API-Football para cada equipo específico.
xG, xGA, PPDA, Posesión, Forma, H2H, Goles por partido.
"""
import httpx
import logging
from typing import Dict, Any, Optional, Tuple

from app.config import get_settings

logger = logging.getLogger(__name__)

# Cache en memoria para evitar llamadas repetidas durante un mismo ciclo de pipeline
_stats_cache: Dict[str, Any] = {}

# Mapa de ligas que usamos → league_id de API-Football
LEAGUE_ID_MAP = {
    "Premier League": 39,
    "La Liga": 140,
    "Serie A": 135,
    "Bundesliga": 78,
    "Ligue 1": 61,
    "Liga MX": 262,
    "UEFA Champions League": 2,
    "UEFA Europa League": 3,
    "World Cup": 1,
    "Copa America": 9,
    "Euro Championship": 4,
}


def _get_season_year(game_date_str: str, league_id: Optional[int] = None) -> int:
    """Determina la temporada correcta basado en la fecha del juego.
    
    NOTA: El plan actual de API-Sports solo permite temporadas hasta 2024.
    Si el año calculado es >= 2025, usamos 2024 como proxy histórico.
    """
    # API-Sports plan limit: only 2022-2024 seasons available
    MAX_SEASON = 2024

    if not game_date_str:
        return MAX_SEASON
    year = int(game_date_str.split("-")[0])
    month = int(game_date_str.split("-")[1]) if len(game_date_str.split("-")) > 1 else 6
    
    # Torneos de verano como World Cup, Euro, Copa America usan el año actual
    if league_id is not None and int(league_id) in [1, 4, 9]:
        season = year
    elif month < 7:
        # Temporadas europeas cruzan año (ej. 2025-26 se registra como "2025")
        season = year - 1
    else:
        season = year
    
    # Cap to the latest season available in our plan
    return min(season, MAX_SEASON)


_espn_names_cache: Dict[str, list] = {}

_TEAM_STOPWORDS = {"fc", "cf", "club", "de", "ac", "sc", "afc", "cd", "ud", "ca", "the"}


def _norm_team_name(name: str) -> str:
    """Normaliza un nombre de equipo: sin acentos, minúsculas y sin prefijos tipo FC/CF/Club."""
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(t for t in s.split() if t not in _TEAM_STOPWORDS)


def _match_espn_team(team_name: str, candidates: list) -> Optional[str]:
    """
    Busca en `candidates` [(id, [alias...]), ...] el equipo que corresponde a `team_name`.
    Devuelve el id de ESPN o None si no hay coincidencia clara (mejor sin dato que dato de otro equipo).
    """
    from difflib import SequenceMatcher
    target = _norm_team_name(team_name)
    if not target:
        return None
    t_tokens = set(target.split())

    scored = []
    for espn_id, aliases in candidates:
        best = 0.0
        for alias in aliases:
            a = _norm_team_name(alias)
            if not a:
                continue
            if a == target:
                best = 1.0
                break
            a_tokens = set(a.split())
            if a_tokens <= t_tokens or t_tokens <= a_tokens:  # uno contiene al otro
                best = max(best, 0.92)
            best = max(best, SequenceMatcher(None, a, target).ratio())
        scored.append((best, espn_id))

    scored.sort(reverse=True)
    if not scored or scored[0][0] < 0.85:
        return None
    if len(scored) > 1 and scored[1][0] >= scored[0][0] - 0.02:
        return None  # empate entre dos equipos -> ambiguo
    return scored[0][1]


async def _resolve_espn_team_id(team_name: str, league_slug: str) -> Optional[str]:
    """
    Los IDs de API-Sports NO son los mismos que los de ESPN, así que se resuelve el ID
    de ESPN por nombre usando la lista de equipos de esa liga (1 llamada por liga, cacheada).
    """
    if league_slug not in _espn_names_cache:
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league_slug}/teams"
        candidates = []
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(url, params={"limit": 100}, timeout=8.0)
                r.raise_for_status()
                data = r.json()
            for sport in data.get("sports", []):
                for league in sport.get("leagues", []):
                    for item in league.get("teams", []):
                        t = item.get("team", {})
                        aliases = [t.get(k) for k in ("displayName", "shortDisplayName", "name", "location") if t.get(k)]
                        if t.get("id") and aliases:
                            candidates.append((str(t["id"]), aliases))
        except Exception as e:
            logger.warning(f"ESPN teams list failed [{league_slug}]: {e}")
        _espn_names_cache[league_slug] = candidates

    return _match_espn_team(team_name, _espn_names_cache[league_slug])


async def _fetch_espn_team_record(team_id: int, league_slug: str) -> Dict:
    """
    Obtiene el record actual del equipo desde ESPN (temporada actual, sin restricciones).
    Devuelve wins, losses, ties, pointsFor (goles), pointsAgainst, gamesPlayed.
    """
    key = f"espn_record_{league_slug}_{team_id}"
    if key in _stats_cache:
        return _stats_cache[key]

    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league_slug}/teams/{team_id}"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, timeout=8.0)
            if r.status_code != 200:
                _stats_cache[key] = {}
                return {}
            data = r.json()
        team = data.get("team", {})
        items = team.get("record", {}).get("items", [{}])
        total_record = next((i for i in items if i.get("type") == "total"), items[0] if items else {})
        stats = {s["name"]: s["value"] for s in total_record.get("stats", [])}
        result = {
            "espn_gamesPlayed":    stats.get("gamesPlayed", 0),
            "espn_wins":           stats.get("wins", 0),
            "espn_losses":         stats.get("losses", 0),
            "espn_ties":           stats.get("ties", 0),
            "espn_goals_for":      stats.get("pointsFor", 0),
            "espn_goals_against":  stats.get("pointsAgainst", 0),
        }
        gp = result["espn_gamesPlayed"] or 1
        result["espn_goals_per_game"]   = round(result["espn_goals_for"] / gp, 2)
        result["espn_goals_against_pg"] = round(result["espn_goals_against"] / gp, 2)
        result["espn_win_pct"]          = round(result["espn_wins"] / gp * 100, 1)
        _stats_cache[key] = result
        return result
    except Exception as e:
        logger.warning(f"ESPN record fetch failed for team {team_id} [{league_slug}]: {e}")
        _stats_cache[key] = {}
        return {}


def _cache_key(team_id: int, league_id: int, season: int) -> str:
    return f"soccer_{team_id}_{league_id}_{season}"


async def _fetch_team_stats(team_id: int, league_id: int, season: int) -> Dict:
    """Fetches team statistics from API-Football v3."""
    import asyncio

    key = _cache_key(team_id, league_id, season)
    if key in _stats_cache:
        return _stats_cache[key]

    settings = get_settings()
    api_key = settings.api_sports_key
    if not api_key:
        logger.warning("API-Sports key not configured, using defaults")
        return {}

    url = "https://v3.football.api-sports.io/teams/statistics"
    headers = {"x-apisports-key": api_key, "Accept": "application/json"}
    params = {"team": team_id, "season": season, "league": league_id}

    # Add a delay to prevent 429 Too Many Requests on free tier
    await asyncio.sleep(1.0)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            result = data.get("response", {})
            _stats_cache[key] = result
            return result
    except Exception as e:
        logger.error(f"Error fetching soccer stats for team {team_id}: {e}")
        return {}


async def _fetch_h2h(team_a_id: int, team_b_id: int) -> list:
    """Fetches head-to-head history between two teams."""
    import asyncio
    h2h_key = f"h2h_{min(team_a_id, team_b_id)}_{max(team_a_id, team_b_id)}"
    if h2h_key in _stats_cache:
        return _stats_cache[h2h_key]

    settings = get_settings()
    api_key = settings.api_sports_key
    if not api_key:
        return []

    url = "https://v3.football.api-sports.io/fixtures/headtohead"
    headers = {"x-apisports-key": api_key, "Accept": "application/json"}
    params = {"h2h": f"{team_a_id}-{team_b_id}", "last": 10}

    # Add a delay to prevent 429 Too Many Requests
    await asyncio.sleep(1.5)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            result = data.get("response", [])
            _stats_cache[h2h_key] = result
            return result
    except Exception as e:
        logger.error(f"Error fetching H2H for {team_a_id} vs {team_b_id}: {e}")
        return []


def _extract_stats_from_response(stats_data: Dict, prefix: str) -> Dict[str, float]:
    """Parses the API-Football /teams/statistics response into flat features."""
    features = {}

    if not stats_data:
        return features

    # Goals
    goals = stats_data.get("goals", {})
    goals_for = goals.get("for", {})
    goals_against = goals.get("against", {})

    # Goals per game (average)
    goals_for_avg = goals_for.get("average", {}).get("total")
    goals_against_avg = goals_against.get("average", {}).get("total")

    if goals_for_avg:
        features[f"{prefix}_goals_per_game"] = float(goals_for_avg)
        features[f"{prefix}_xg_for"] = float(goals_for_avg)  # xG proxy from actual scoring rate
    if goals_against_avg:
        features[f"{prefix}_goals_against_per_game"] = float(goals_against_avg)
        features[f"{prefix}_xga_against"] = float(goals_against_avg)  # xGA proxy

    # Clean sheets and failed to score
    clean_sheets = stats_data.get("clean_sheet", {}).get("total")
    failed_to_score = stats_data.get("failed_to_score", {}).get("total")
    fixtures_played = stats_data.get("fixtures", {}).get("played", {}).get("total", 1)

    if clean_sheets is not None and fixtures_played:
        features[f"{prefix}_clean_sheet_pct"] = round(clean_sheets / max(fixtures_played, 1) * 100, 1)
    if failed_to_score is not None and fixtures_played:
        features[f"{prefix}_failed_to_score_pct"] = round(failed_to_score / max(fixtures_played, 1) * 100, 1)

    # Form (e.g. "WWDLW")
    form_str = stats_data.get("form", "")
    if form_str:
        last_5 = form_str[-5:]  # Last 5 results
        points = 0
        for ch in last_5:
            if ch == "W":
                points += 3
            elif ch == "D":
                points += 1
        features[f"{prefix}_form_points"] = round(points / max(len(last_5), 1), 2)
        features[f"{prefix}_form_string"] = last_5

    # Fixtures (wins/draws/losses for calculating win rate)
    fixtures = stats_data.get("fixtures", {})
    wins_total = fixtures.get("wins", {}).get("total", 0)
    draws_total = fixtures.get("draws", {}).get("total", 0)
    losses_total = fixtures.get("losses", {}).get("total", 0)
    total_played = fixtures.get("played", {}).get("total", 1)

    if total_played:
        features[f"{prefix}_win_pct"] = round(wins_total / max(total_played, 1) * 100, 1)
        features[f"{prefix}_draw_pct"] = round(draws_total / max(total_played, 1) * 100, 1)

    # Home vs Away splits
    wins_home = fixtures.get("wins", {}).get("home", 0)
    wins_away = fixtures.get("wins", {}).get("away", 0)
    played_home = fixtures.get("played", {}).get("home", 1)
    played_away = fixtures.get("played", {}).get("away", 1)

    if prefix == "home" and played_home:
        features[f"{prefix}_home_win_pct"] = round(wins_home / max(played_home, 1) * 100, 1)
        # Home goals average
        home_goals_avg = goals_for.get("average", {}).get("home")
        if home_goals_avg:
            features[f"{prefix}_home_goals_avg"] = float(home_goals_avg)
    elif prefix == "away" and played_away:
        features[f"{prefix}_away_win_pct"] = round(wins_away / max(played_away, 1) * 100, 1)
        away_goals_avg = goals_for.get("average", {}).get("away")
        if away_goals_avg:
            features[f"{prefix}_away_goals_avg"] = float(away_goals_avg)

    # Lineups (most used formation)
    lineups = stats_data.get("lineups", [])
    if lineups:
        most_used = max(lineups, key=lambda x: x.get("played", 0))
        features[f"{prefix}_formation"] = most_used.get("formation", "4-4-2")

    # Penalty stats
    penalty = stats_data.get("penalty", {})
    pen_scored = penalty.get("scored", {}).get("total", 0)
    pen_missed = penalty.get("missed", {}).get("total", 0)
    if pen_scored or pen_missed:
        features[f"{prefix}_penalty_conversion"] = round(
            pen_scored / max(pen_scored + pen_missed, 1) * 100, 1
        )

    # Cards (discipline indicator)
    cards = stats_data.get("cards", {})
    yellow_total = sum(
        v.get("total") or 0
        for v in (cards.get("yellow", {}) or {}).values()
        if isinstance(v, dict)
    )
    red_total = sum(
        v.get("total") or 0
        for v in (cards.get("red", {}) or {}).values()
        if isinstance(v, dict)
    )
    if total_played:
        features[f"{prefix}_yellow_per_game"] = round(yellow_total / max(total_played, 1), 2)
        features[f"{prefix}_red_total"] = red_total

    return features


def _extract_h2h_features(h2h_data: list, home_id: int, away_id: int) -> Dict[str, float]:
    """Extracts head-to-head statistics from fixture history."""
    features = {
        "h2h_total_games": 0,
        "h2h_home_wins": 0,
        "h2h_away_wins": 0,
        "h2h_draws": 0,
        "h2h_home_goals_avg": 0.0,
        "h2h_away_goals_avg": 0.0,
    }

    if not h2h_data:
        return features

    home_goals_total = 0
    away_goals_total = 0
    games_counted = 0

    for fixture in h2h_data:
        teams = fixture.get("teams", {})
        goals = fixture.get("goals", {})
        score = fixture.get("score", {}).get("fulltime", {})

        ft_home = score.get("home") if score.get("home") is not None else goals.get("home")
        ft_away = score.get("away") if score.get("away") is not None else goals.get("away")

        if ft_home is None or ft_away is None:
            continue

        fixture_home_id = teams.get("home", {}).get("id")

        # Normalize: map goals to our home/away perspective
        if fixture_home_id == home_id:
            h_goals = ft_home
            a_goals = ft_away
        else:
            h_goals = ft_away
            a_goals = ft_home

        home_goals_total += h_goals
        away_goals_total += a_goals
        games_counted += 1

        if h_goals > a_goals:
            features["h2h_home_wins"] += 1
        elif a_goals > h_goals:
            features["h2h_away_wins"] += 1
        else:
            features["h2h_draws"] += 1

    features["h2h_total_games"] = games_counted
    if games_counted > 0:
        features["h2h_home_goals_avg"] = round(home_goals_total / games_counted, 2)
        features["h2h_away_goals_avg"] = round(away_goals_total / games_counted, 2)

    return features


async def extract_advanced_soccer_features(game_data: dict, game_date_str: str) -> dict:
    """
    Extrae factores avanzados REALES para Fútbol.
    Fuentes (en orden de frescura):
    1. ESPN Site API  → Record actual de temporada (goles, victorias, derrotas) — SIN restricciones
    2. API-Football   → Stats históricas profundas (forma, xG, clean sheets) — season 2024
    3. API-Football   → H2H historial directo — siempre disponible
    """
    from app.utils.constants import ODDS_API_KEYS
    from app.db_models import SportType

    features = {}

    home_data = game_data.get("teams", {}).get("home", {})
    away_data = game_data.get("teams", {}).get("away", {})
    league_data = game_data.get("league", {})

    home_name = home_data.get("name", "Local")
    away_name = away_data.get("name", "Visitante")
    home_id = home_data.get("id")
    away_id = away_data.get("id")
    league_name = league_data.get("name", "Liga")
    league_id = league_data.get("id") or LEAGUE_ID_MAP.get(league_name)

    # ESPN league slug para obtener el record actual
    from app.services.espn_nfl_client import SOCCER_ESPN_LEAGUES
    league_slug = SOCCER_ESPN_LEAGUES.get(league_name, "uefa.champions")

    features["home_team_name"] = home_name
    features["away_team_name"] = away_name
    features["competicion"] = league_name

    # ─── Defaults seguros ────────────────────────────────────────────────────
    for prefix in ["home", "away"]:
        features[f"{prefix}_xg_for"]                  = 1.2
        features[f"{prefix}_xga_against"]             = 1.2
        features[f"{prefix}_goals_per_game"]          = 1.2
        features[f"{prefix}_goals_against_per_game"]  = 1.2
        features[f"{prefix}_possession"]              = 50.0
        features[f"{prefix}_ppda"]                    = 12.0
        features[f"{prefix}_form_points"]             = 1.5
        features[f"{prefix}_form_string"]             = "-----"
        features[f"{prefix}_win_pct"]                 = 33.3
        features[f"{prefix}_clean_sheet_pct"]         = 20.0
        features[f"{prefix}_failed_to_score_pct"]     = 20.0
        features[f"{prefix}_yellow_per_game"]         = 2.0
        features[f"{prefix}_red_total"]               = 0
        features[f"{prefix}_formation"]               = "4-4-2"

    # ═══════════════════════════════════════════════════════════════════════
    # FUENTE 1: ESPN — record actual (siempre disponible, temporada actual)
    # ═══════════════════════════════════════════════════════════════════════
    espn_keys = set()  # claves llenadas con datos REALES de ESPN (temporada actual)
    try:
        for prefix, name in (("home", home_name), ("away", away_name)):
            espn_id = await _resolve_espn_team_id(name, league_slug)
            if not espn_id:
                logger.info(f"ESPN: sin equivalente claro para {name} [{league_slug}], se usará API-Football")
                continue
            rec = await _fetch_espn_team_record(espn_id, league_slug)
            if rec.get("espn_gamesPlayed", 0) > 0:
                gpm = rec["espn_goals_per_game"]
                gap = rec["espn_goals_against_pg"]
                values = {
                    f"{prefix}_goals_per_game":         gpm,
                    f"{prefix}_goals_against_per_game": gap,
                    f"{prefix}_xg_for":                 round(gpm, 2),
                    f"{prefix}_xga_against":            round(gap, 2),
                    f"{prefix}_win_pct":                rec["espn_win_pct"],
                }
                features.update(values)
                espn_keys.update(values)
                gp = int(rec["espn_gamesPlayed"])
                logger.info(f"✅ ESPN {name}: {gp}G {rec['espn_wins']:.0f}W {gpm:.1f}GPG")
    except Exception as e:
        logger.warning(f"ESPN soccer record fetch failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # FUENTE 2: API-Football — stats históricas (season 2024, profundidad)
    # ═══════════════════════════════════════════════════════════════════════
    api_failed = False
    if home_id and away_id and league_id:
        season = _get_season_year(game_date_str, league_id)
        try:
            logger.info(f"API-Football stats: {home_name}({home_id}) vs {away_name}({away_id}) league={league_id} season={season}")
            home_stats = await _fetch_team_stats(home_id, league_id, season)
            away_stats = await _fetch_team_stats(away_id, league_id, season)

            if home_stats:
                home_hist = _extract_stats_from_response(home_stats, "home")
                for key, val in home_hist.items():
                    # API-Football llena todo, salvo lo que ESPN ya trajo con datos reales
                    if key not in espn_keys:
                        features[key] = val
                logger.info(f"✅ API-Football histórico {home_name}: {len(home_hist)} features")

            if away_stats:
                away_hist = _extract_stats_from_response(away_stats, "away")
                for key, val in away_hist.items():
                    if key not in espn_keys:
                        features[key] = val
                logger.info(f"✅ API-Football histórico {away_name}: {len(away_hist)} features")

        except Exception as e:
            logger.error(f"API-Football stats error {home_name} vs {away_name}: {e}")
            api_failed = True
    else:
        api_failed = True

    # ═══════════════════════════════════════════════════════════════════════
    # FUENTE 3: H2H — API-Football (sin restricción de season)
    # ═══════════════════════════════════════════════════════════════════════
    if home_id and away_id:
        try:
            h2h_data = await _fetch_h2h(home_id, away_id)
            h2h_features = _extract_h2h_features(h2h_data, home_id, away_id)
            features.update(h2h_features)
            if h2h_features.get("h2h_total_games", 0) > 0:
                logger.info(f"✅ H2H: {h2h_features['h2h_total_games']} juegos entre {home_name} vs {away_name}")
        except Exception as e:
            logger.warning(f"H2H fetch error: {e}")

    if api_failed:
        logger.warning(f"API-Football histórico no disponible para {home_name} vs {away_name}")

    # ─── Contexto ────────────────────────────────────────────────────────────
    features["home_suspensions"] = 0
    features["away_suspensions"] = 0
    features["home_champions_league_rotation_risk"] = False
    features["away_champions_league_rotation_risk"] = False

    return features


    league_data = game_data.get("league", {})

    home_name = home_data.get("name", "Local")
    away_name = away_data.get("name", "Visitante")
    home_id = home_data.get("id")
    away_id = away_data.get("id")
    league_name = league_data.get("name", "Liga")
    league_id = league_data.get("id") or LEAGUE_ID_MAP.get(league_name)

    features["home_team_name"] = home_name
    features["away_team_name"] = away_name
    features["competicion"] = league_name

    # Determine season for API requests
    season = _get_season_year(game_date_str, league_id)

    # ─── Set safe defaults but flag as error if API fails ───
    api_failed = False
    for prefix in ["home", "away"]:
        features[f"{prefix}_xg_for"] = 1.0
        features[f"{prefix}_xga_against"] = 1.0
        features[f"{prefix}_goals_per_game"] = 1.0
        features[f"{prefix}_goals_against_per_game"] = 1.0
        features[f"{prefix}_possession"] = 50.0
        features[f"{prefix}_ppda"] = 12.0
        features[f"{prefix}_form_points"] = 1.0
        features[f"{prefix}_form_string"] = "-----"
        features[f"{prefix}_win_pct"] = 33.3
        features[f"{prefix}_clean_sheet_pct"] = 20.0
        features[f"{prefix}_failed_to_score_pct"] = 20.0
        features[f"{prefix}_yellow_per_game"] = 2.0
        features[f"{prefix}_red_total"] = 0
        features[f"{prefix}_formation"] = "4-4-2"

    # ─── Fetch real data from API-Football ───
    if home_id and away_id and league_id:
        try:
            logger.info(f"DEBUG FETCH STATS: home={home_name}({home_id}) away={away_name}({away_id}) league_id={league_id} date={game_date_str} computed_season={season}")
            # Fetch team stats for both teams
            home_stats = await _fetch_team_stats(home_id, league_id, season)
            away_stats = await _fetch_team_stats(away_id, league_id, season)

            if not home_stats or not away_stats:
                api_failed = True

            # Extract features from each team's response
            if home_stats:
                home_features = _extract_stats_from_response(home_stats, "home")
                features.update(home_features)
                logger.info(f"✅ Soccer stats loaded for {home_name} ({len(home_features)} features)")

            if away_stats:
                away_features = _extract_stats_from_response(away_stats, "away")
                features.update(away_features)
                logger.info(f"✅ Soccer stats loaded for {away_name} ({len(away_features)} features)")

            # Fetch H2H
            h2h_data = await _fetch_h2h(home_id, away_id)
            if not h2h_data:
                api_failed = True
            h2h_features = _extract_h2h_features(h2h_data, home_id, away_id)
            features.update(h2h_features)
            if h2h_features.get("h2h_total_games", 0) > 0:
                logger.info(f"✅ H2H loaded: {h2h_features['h2h_total_games']} games between {home_name} vs {away_name}")

        except Exception as e:
            logger.error(f"Error fetching soccer metrics for {home_name} vs {away_name}: {e}")
            api_failed = True
    else:
        logger.warning(f"Missing team/league IDs for {home_name} vs {away_name}, using defaults")
        api_failed = True

    if api_failed:
        logger.warning(f"Using default stats for {home_name} vs {away_name} (API data unavailable for this league/season)")

    # ─── Level 4: Suspensions & Context ───
    features["home_suspensions"] = 0
    features["away_suspensions"] = 0
    features["home_champions_league_rotation_risk"] = False
    features["away_champions_league_rotation_risk"] = False

    return features