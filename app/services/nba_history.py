"""
OmniPicks — NBA v2: histórico propio desde nba_api (stats.nba.com, gratis).

- Partidos y box scores por jugador: 2 llamadas por temporada.
- Stints (tramos con los mismos 10 jugadores): rotaciones + play-by-play, 2 llamadas por partido.

Tiempos en décimas de segundo desde el inicio del partido (igual que GameRotation):
cuarto = 7200, tiempo extra = 3000.
"""
import logging
import re
import threading
import time
from bisect import bisect_left
from datetime import date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SEASON_TYPES = ("Regular Season", "Playoffs")
_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")
_FINAL_FT_RE = re.compile(r"(\d) of \1")          # "2 of 2", "3 of 3" (también "1 of 1")

# ── Límite de velocidad global (stats.nba.com bloquea si se le pide muy rápido) ──
_rate_lock = threading.Lock()
_last_call = [0.0]
MIN_INTERVAL = 0.45


def _throttle():
    with _rate_lock:
        wait = MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()


def call_nba(endpoint_cls, retries: int = 5, **kwargs):
    """Llama un endpoint de nba_api con límite de velocidad y reintentos."""
    last_err = None
    for attempt in range(retries):
        _throttle()
        try:
            return endpoint_cls(timeout=30, **kwargs).get_data_frames()
        except Exception as e:                      # timeouts, JSON vacío, bloqueos temporales
            last_err = e
            time.sleep(min(60, 2 ** attempt * 2))
    raise last_err


def season_of(d: date) -> str:
    start = d.year if d.month >= 10 else d.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


# ══════════════════════════════════════════════════════════════════════
# Partidos y box scores (por temporada)
# ══════════════════════════════════════════════════════════════════════

def fetch_season(season: str, season_type: str):
    """Devuelve (games, player_games) como listas de dicts listas para insertar."""
    from nba_api.stats.endpoints import leaguegamelog, playergamelogs

    lg = call_nba(leaguegamelog.LeagueGameLog, season=season, season_type_all_star=season_type)[0]
    if lg.empty:
        return [], []

    games: Dict[str, Dict] = {}
    opp: Dict[tuple, int] = {}
    for gid, grp in lg.groupby("GAME_ID"):
        if len(grp) != 2:
            continue
        a, b = grp.iloc[0], grp.iloc[1]
        home, away = (b, a) if " @ " in a.MATCHUP else (a, b)
        games[gid] = {
            "game_id": gid, "season": season, "season_type": season_type,
            "game_date": date.fromisoformat(str(home.GAME_DATE)[:10]),
            "home_team_id": int(home.TEAM_ID), "away_team_id": int(away.TEAM_ID),
            "home_pts": None if home.PTS != home.PTS else int(home.PTS),
            "away_pts": None if away.PTS != away.PTS else int(away.PTS),
        }
        opp[(gid, int(home.TEAM_ID))] = int(away.TEAM_ID)
        opp[(gid, int(away.TEAM_ID))] = int(home.TEAM_ID)

    pg = call_nba(playergamelogs.PlayerGameLogs, season_nullable=season,
                  season_type_nullable=season_type)[0]
    rows = []
    for r in pg.itertuples():
        g = games.get(r.GAME_ID)
        if not g:
            continue
        team_id = int(r.TEAM_ID)
        n = lambda v: 0 if v != v or v is None else int(v)      # NaN → 0
        rows.append({
            "game_id": r.GAME_ID, "season": season, "game_date": g["game_date"],
            "team_id": team_id, "opp_team_id": opp[(r.GAME_ID, team_id)],
            "is_home": team_id == g["home_team_id"],
            "player_id": int(r.PLAYER_ID), "player_name": r.PLAYER_NAME,
            "minutes": 0.0 if r.MIN != r.MIN else float(r.MIN),
            "pts": n(r.PTS), "fgm": n(r.FGM), "fga": n(r.FGA), "fg3m": n(r.FG3M), "fg3a": n(r.FG3A),
            "ftm": n(r.FTM), "fta": n(r.FTA), "oreb": n(r.OREB), "dreb": n(r.DREB),
            "ast": n(r.AST), "stl": n(r.STL), "blk": n(r.BLK), "tov": n(r.TOV), "pf": n(r.PF),
            "plus_minus": 0.0 if r.PLUS_MINUS != r.PLUS_MINUS else float(r.PLUS_MINUS),
        })
    return list(games.values()), rows


# ══════════════════════════════════════════════════════════════════════
# Stints (por partido)
# ══════════════════════════════════════════════════════════════════════

def _event_time(period: int, clock: str) -> Optional[int]:
    m = _CLOCK_RE.match(clock or "")
    if not m:
        return None
    remaining = int((int(m.group(1)) * 60 + float(m.group(2))) * 10)
    if period <= 4:
        return (period - 1) * 7200 + 7200 - remaining
    return 28800 + (period - 5) * 3000 + 3000 - remaining


def build_stints(game_id: str, home_team_id: int) -> List[Dict]:
    """Tramos del partido con sus 10 jugadores, puntos y posesiones de cada lado."""
    from nba_api.stats.endpoints import gamerotation, playbyplayv3

    rot_frames = call_nba(gamerotation.GameRotation, game_id=game_id)
    pbp = call_nba(playbyplayv3.PlayByPlayV3, game_id=game_id)[0]

    # 1. Intervalos en cancha por jugador
    intervals = []                                  # (in, out, team_id, player_id)
    for df in rot_frames[:2]:
        for r in df.itertuples():
            intervals.append((int(r.IN_TIME_REAL), int(r.OUT_TIME_REAL), int(r.TEAM_ID), int(r.PERSON_ID)))
    if not intervals:
        return []

    bounds = sorted({t for i in intervals for t in (i[0], i[1])})
    segments = []
    for a, b in zip(bounds, bounds[1:]):
        home = sorted(p for (i, o, t, p) in intervals if i <= a and o >= b and t == home_team_id)
        away = sorted(p for (i, o, t, p) in intervals if i <= a and o >= b and t != home_team_id)
        segments.append({
            "game_id": game_id, "start_t": a, "end_t": b,
            "home_players": ",".join(map(str, home)), "away_players": ",".join(map(str, away)),
            "home_pts": 0, "away_pts": 0, "home_poss": 0, "away_poss": 0,
            "_valid": len(home) == 5 and len(away) == 5,
        })

    def seg_for(t: int) -> Dict:
        # Un evento justo en un cambio pertenece al tramo que termina (la jugada es de ese quinteto)
        idx = bisect_left(bounds, t) - 1
        return segments[max(0, min(idx, len(segments) - 1))]

    # 2. Puntos y posesiones desde el play-by-play
    prev_home = prev_away = 0
    last_miss_team = None
    for e in pbp.sort_values("actionNumber").itertuples():
        t = _event_time(int(e.period), e.clock)
        if t is None:
            continue
        seg = seg_for(t)
        team = int(e.teamId) if e.teamId == e.teamId and e.teamId else 0
        side = "home" if team == home_team_id else "away"
        desc = e.description or ""

        if e.scoreHome not in ("", None) and e.scoreAway not in ("", None):
            sh, sa = int(e.scoreHome), int(e.scoreAway)
            seg["home_pts"] += max(0, sh - prev_home)
            seg["away_pts"] += max(0, sa - prev_away)
            prev_home, prev_away = sh, sa

        if not team:
            continue
        if e.actionType == "Made Shot":
            seg[f"{side}_poss"] += 1
        elif e.actionType == "Turnover":
            seg[f"{side}_poss"] += 1
        elif e.actionType == "Missed Shot":
            last_miss_team = team
        elif e.actionType == "Free Throw":
            if desc.startswith("MISS"):
                last_miss_team = team
            elif _FINAL_FT_RE.search(e.subType or "") and "1 of 1" not in (e.subType or ""):
                seg[f"{side}_poss"] += 1
        elif e.actionType == "Rebound":
            if last_miss_team and team != last_miss_team:       # rebote defensivo: termina la posesión rival
                other = "home" if last_miss_team == home_team_id else "away"
                seg[f"{other}_poss"] += 1
            last_miss_team = None

    out = []
    for s in segments:
        valid = s.pop("_valid")
        if valid and (s["home_poss"] or s["away_poss"] or s["home_pts"] or s["away_pts"]):
            out.append(s)
    return out
