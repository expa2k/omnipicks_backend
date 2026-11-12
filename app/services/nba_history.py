"""
OmniPicks — NBA v2: histórico propio desde nba_api (stats.nba.com, gratis).

- Partidos y box scores por jugador: 2 llamadas por temporada.
- Stints (tramos con los mismos 10 jugadores): rotaciones + play-by-play, 2 llamadas por partido.
  El crudo de cada partido se guarda en ml/datasets/nba_raw/ para poder rearmar los
  stints sin volver a descargar.

Tiempos en décimas de segundo desde el inicio del partido (igual que GameRotation):
cuarto = 7200, tiempo extra = 3000.
"""
import gzip
import json
import logging
import os
import re
import threading
import time
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Dict, List, Optional, Tuple

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
    from datetime import datetime
    from zoneinfo import ZoneInfo
    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    for gid, grp in lg.groupby("GAME_ID"):
        if len(grp) != 2:
            continue
        # Solo partidos terminados: la NBA publica los que siguen en juego (sin W/L y con puntos parciales),
        # y los de hoy pueden tener estadísticas todavía incompletas
        if "WL" in grp and grp["WL"].isna().any():
            continue
        if date.fromisoformat(str(grp.iloc[0].GAME_DATE)[:10]) >= today_et:
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


def fetch_player_bio(season: str) -> List[Dict]:
    """Edad, estatura y peso de cada jugador en la temporada (1 llamada)."""
    from nba_api.stats.endpoints import leaguedashplayerbiostats

    df = call_nba(leaguedashplayerbiostats.LeagueDashPlayerBioStats, season=season,
                  season_type_all_star="Regular Season")[0]
    f = lambda v: None if v is None or v != v or v == "" else float(v)
    return [{"season": season, "player_id": int(r.PLAYER_ID), "player_name": r.PLAYER_NAME,
             "age": f(r.AGE), "height_in": f(r.PLAYER_HEIGHT_INCHES), "weight_lb": f(r.PLAYER_WEIGHT)}
            for r in df.itertuples()]


def fetch_players_index() -> List[Dict]:
    """Posición, estatura y peso de todos los jugadores de la historia (1 llamada)."""
    from nba_api.stats.endpoints import playerindex

    df = call_nba(playerindex.PlayerIndex, league_id="00", season=season_of(date.today()), historical_nullable="1")[0]

    def inches(h):
        try:
            ft, inch = str(h).split("-")
            return float(ft) * 12 + float(inch)
        except (ValueError, AttributeError):
            return None

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return [{"player_id": int(r.PERSON_ID), "player_name": f"{r.PLAYER_FIRST_NAME} {r.PLAYER_LAST_NAME}".strip(),
             "position": (r.POSITION or None), "height_in": inches(r.HEIGHT), "weight_lb": num(r.WEIGHT)}
            for r in df.itertuples()]


def load_players_index(engine) -> int:
    from sqlalchemy.dialects.mysql import insert as mysql_insert
    from app.db_models import NBAPlayer

    rows = fetch_players_index()
    with engine.begin() as conn:
        for part in _chunks(rows):
            stmt = mysql_insert(NBAPlayer).values(part)
            conn.execute(stmt.on_duplicate_key_update(
                position=stmt.inserted.position, height_in=stmt.inserted.height_in, weight_lb=stmt.inserted.weight_lb))
    return len(rows)


# ══════════════════════════════════════════════════════════════════════
# Stints (por partido)
# ══════════════════════════════════════════════════════════════════════

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ml", "datasets", "nba_raw"))
_ROT_COLS = ["TEAM_ID", "PERSON_ID", "IN_TIME_REAL", "OUT_TIME_REAL"]
_PBP_COLS = ["actionNumber", "period", "clock", "teamId", "personId", "actionType", "subType",
             "description", "shotValue", "shotResult", "shotDistance", "xLegacy", "yLegacy",
             "scoreHome", "scoreAway"]


def _raw_path(game_id: str) -> str:
    return os.path.join(RAW_DIR, game_id[:5], f"{game_id}.json.gz")


def fetch_game_raw(game_id: str, use_cache: bool = True) -> Dict:
    """Rotaciones y play-by-play crudos de un partido: {'rot': [...], 'pbp': [...]}.

    Los eventos del play-by-play conservan el orden en que los entrega la NBA, que es
    el cronológico (actionNumber no lo es: las repeticiones se numeran al final).
    """
    path = _raw_path(game_id)
    if use_cache and os.path.exists(path):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            pass                                    # archivo truncado: se vuelve a descargar

    from nba_api.stats.endpoints import gamerotation, playbyplayv3

    rot_frames = call_nba(gamerotation.GameRotation, game_id=game_id)
    pbp = call_nba(playbyplayv3.PlayByPlayV3, game_id=game_id)[0]
    rot = []
    for df in rot_frames[:2]:
        rot.extend(json.loads(df[_ROT_COLS].to_json(orient="records")))
    raw = {"rot": rot, "pbp": json.loads(pbp[[c for c in _PBP_COLS if c in pbp.columns]].to_json(orient="records"))}

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{threading.get_ident()}.tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(raw, f, separators=(",", ":"))
    os.replace(tmp, path)
    return raw


def _event_time(period: int, clock: str) -> Optional[int]:
    m = _CLOCK_RE.match(clock or "")
    if not m:
        return None
    remaining = int((int(m.group(1)) * 60 + float(m.group(2))) * 10)
    if period <= 4:
        return (period - 1) * 7200 + 7200 - remaining
    return 28800 + (period - 5) * 3000 + 3000 - remaining


def stints_from_raw(game_id: str, home_team_id: int, raw: Dict) -> Tuple[List[Dict], Dict]:
    """Arma los tramos de un partido a partir del crudo.

    Devuelve (stints, calidad). `calidad` trae los puntos de cada lado según el
    play-by-play completo y según los stints válidos, y los minutos cubiertos; sirve
    para detectar partidos con rotaciones corruptas en el origen.
    """
    # 1. Intervalos en cancha por jugador
    intervals = [(int(r["IN_TIME_REAL"]), int(r["OUT_TIME_REAL"]), int(r["TEAM_ID"]), int(r["PERSON_ID"]))
                 for r in raw["rot"] if r["IN_TIME_REAL"] is not None and r["OUT_TIME_REAL"] is not None]
    quality = {"pbp_home": 0, "pbp_away": 0, "stint_home": 0, "stint_away": 0, "minutes": 0.0}
    if not intervals:
        return [], quality

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

    # 2. Puntos y posesiones desde el play-by-play, en orden cronológico.
    #    Los puntos salen de cada evento (canasta o tiro libre anotado), no de la
    #    diferencia de marcador: el marcador del PBP trae retrocesos y erratas.
    events = []
    for n, e in enumerate(raw["pbp"]):
        t = _event_time(int(e["period"]), e.get("clock"))
        if t is not None:
            events.append((t, n, e))
    events.sort(key=lambda x: (x[0], x[1]))

    last_miss_team = None
    last_miss_seg = None
    for t, _, e in events:
        seg = seg_for(t)
        action = e.get("actionType") or ""
        sub = e.get("subType") or ""
        desc = e.get("description") or ""

        if action == "period":
            # Fin de periodo con un tiro fallado sin rebote: esa posesión también terminó
            if sub == "end" and last_miss_team:
                side = "home" if last_miss_team == home_team_id else "away"
                (last_miss_seg or seg)[f"{side}_poss"] += 1
            last_miss_team = last_miss_seg = None
            continue

        team = int(e["teamId"]) if e.get("teamId") else 0
        if not team:
            continue
        side = "home" if team == home_team_id else "away"

        if action == "Made Shot":
            seg[f"{side}_pts"] += int(e.get("shotValue") or 2)
            seg[f"{side}_poss"] += 1
            last_miss_team = last_miss_seg = None
        elif action == "Turnover":
            seg[f"{side}_poss"] += 1
            last_miss_team = last_miss_seg = None
        elif action == "Missed Shot":
            last_miss_team, last_miss_seg = team, seg
        elif action == "Free Throw":
            final = bool(_FINAL_FT_RE.search(sub))
            if desc.startswith("MISS"):
                if final:
                    last_miss_team, last_miss_seg = team, seg
            else:
                seg[f"{side}_pts"] += 1
                if final and "1 of 1" not in sub:
                    seg[f"{side}_poss"] += 1
                    last_miss_team = last_miss_seg = None
        elif action == "Rebound":
            if last_miss_team and team != last_miss_team:       # rebote defensivo: termina la posesión rival
                other = "home" if last_miss_team == home_team_id else "away"
                seg[f"{other}_poss"] += 1
            last_miss_team = last_miss_seg = None

    out = []
    for s in segments:
        valid = s.pop("_valid")
        quality["pbp_home"] += s["home_pts"]
        quality["pbp_away"] += s["away_pts"]
        if not valid:
            continue
        quality["stint_home"] += s["home_pts"]
        quality["stint_away"] += s["away_pts"]
        quality["minutes"] += (s["end_t"] - s["start_t"]) / 600.0
        if s["home_poss"] or s["away_poss"] or s["home_pts"] or s["away_pts"]:
            out.append(s)
    return out, quality


def build_stints(game_id: str, home_team_id: int, use_cache: bool = True) -> List[Dict]:
    """Tramos del partido con sus 10 jugadores, puntos y posesiones de cada lado."""
    return stints_from_raw(game_id, home_team_id, fetch_game_raw(game_id, use_cache))[0]


# ══════════════════════════════════════════════════════════════════════
# Carga a la base (la usan el backfill y la actualización diaria)
# ══════════════════════════════════════════════════════════════════════

def current_seasons(first_year: Optional[int] = None) -> List[str]:
    last = int(season_of(date.today())[:4])
    first = last if first_year is None else first_year
    return [f"{y}-{(y + 1) % 100:02d}" for y in range(first, last + 1)]


def _chunks(rows, n=1000):
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


def load_season(engine, season: str) -> int:
    """Inserta o actualiza los partidos y box scores de una temporada. Devuelve partidos vistos."""
    from sqlalchemy.dialects.mysql import insert as mysql_insert
    from app.db_models import NBAGame, NBAPlayerGame, NBAPlayerSeason

    try:
        bio = fetch_player_bio(season)
        with engine.begin() as conn:
            for part in _chunks(bio):
                stmt = mysql_insert(NBAPlayerSeason).values(part)
                conn.execute(stmt.on_duplicate_key_update(
                    age=stmt.inserted.age, height_in=stmt.inserted.height_in, weight_lb=stmt.inserted.weight_lb))
    except Exception as e:                          # la ficha es secundaria: no detiene la carga
        logger.warning(f"{season}: no se pudo leer la ficha de jugadores ({type(e).__name__}: {e})")

    total = 0
    for season_type in SEASON_TYPES:
        games, player_games = fetch_season(season, season_type)
        if not games:
            logger.info(f"{season} {season_type}: sin partidos")
            continue
        with engine.begin() as conn:
            for part in _chunks(games):
                stmt = mysql_insert(NBAGame).values(part)
                conn.execute(stmt.on_duplicate_key_update(
                    home_pts=stmt.inserted.home_pts, away_pts=stmt.inserted.away_pts))
            for part in _chunks(player_games):
                stmt = mysql_insert(NBAPlayerGame).values(part)
                cols = ("minutes", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb",
                        "ast", "stl", "blk", "tov", "pf", "plus_minus")
                conn.execute(stmt.on_duplicate_key_update(**{c: stmt.inserted[c] for c in cols}))
        total += len(games)
        logger.info(f"{season} {season_type}: {len(games)} partidos, {len(player_games)} filas de jugador")
    return total


def load_stints(engine, seasons: List[str], workers: int = 3) -> Tuple[int, int]:
    """Arma y guarda los stints de los partidos pendientes. Devuelve (hechos, fallidos)."""
    from sqlalchemy import bindparam, text
    from app.db_models import NBAStint

    with engine.connect() as conn:
        pending = conn.execute(text(
            "SELECT game_id, home_team_id, home_pts, away_pts FROM nba_games "
            "WHERE stints_done = 0 AND home_pts IS NOT NULL AND season IN :s ORDER BY game_date"
        ).bindparams(bindparam("s", expanding=True)), {"s": seasons}).fetchall()

    total = len(pending)
    logger.info(f"Stints pendientes: {total} partidos")
    done = failed = suspect = 0
    start = time.time()

    def work(row):
        return stints_from_raw(row.game_id, row.home_team_id, fetch_game_raw(row.game_id))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, r): r for r in pending}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                stints, q = fut.result()
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM nba_stints WHERE game_id = :g"), {"g": row.game_id})
                    if stints:
                        conn.execute(NBAStint.__table__.insert(), stints)
                    conn.execute(text("UPDATE nba_games SET stints_done = 1 WHERE game_id = :g"), {"g": row.game_id})
                done += 1
                lost = abs(q["stint_home"] - row.home_pts) + abs(q["stint_away"] - row.away_pts)
                if lost > 6:
                    suspect += 1
                    logger.warning(
                        f"Partido {row.game_id}: stints {q['stint_home']}-{q['stint_away']} en {q['minutes']:.0f} min "
                        f"contra marcador {row.home_pts}-{row.away_pts} (PBP {q['pbp_home']}-{q['pbp_away']})")
            except Exception as e:
                failed += 1
                logger.warning(f"Partido {row.game_id} falló ({type(e).__name__}: {str(e)[:80]}); se reintenta en la próxima corrida")
            n = done + failed
            if n % 50 == 0 or n == total:
                rate = n / max(time.time() - start, 1)
                eta = (total - n) / max(rate, 0.01) / 60
                logger.info(f"Stints {n}/{total} (ok {done}, fallidos {failed}, incompletos {suspect}) · faltan ~{eta:.0f} min")
    return done, failed



# ══════════════════════════════════════════════════════════════════════
# Quién reemplaza a quién (de las rotaciones)
# ══════════════════════════════════════════════════════════════════════

SUBS_CACHE = os.path.join(RAW_DIR, "..", "nba_subs.pkl")


def substitution_pairs(raw: Dict) -> List[Tuple[int, int, int, float]]:
    """[(equipo, sale, entra, peso)] de un partido, a partir de las rotaciones.

    En cada momento en que un equipo cambia jugadores, cada uno de los que salen se
    empareja con cada uno de los que entran (peso 1/k si salen k a la vez). No cuentan
    el inicio del partido ni los que siguen en cancha.
    """
    starts: Dict[Tuple[int, int], set] = {}
    ends: Dict[Tuple[int, int], set] = {}
    for r in raw.get("rot", []):
        if r["IN_TIME_REAL"] is None or r["OUT_TIME_REAL"] is None:
            continue
        team, pid = int(r["TEAM_ID"]), int(r["PERSON_ID"])
        starts.setdefault((team, int(r["IN_TIME_REAL"])), set()).add(pid)
        ends.setdefault((team, int(r["OUT_TIME_REAL"])), set()).add(pid)
    out = []
    for (team, t), outs in ends.items():
        ins = starts.get((team, t), set())
        outs, ins = outs - ins, ins - outs              # quien sale y vuelve a entrar en el mismo instante no cambia
        if not outs or not ins or t == 0:
            continue
        w = 1.0 / max(len(outs), len(ins))
        out.extend((team, o, i, w) for o in outs for i in ins)
    return out


def load_substitutions(game_ids) -> Dict[str, List[Tuple[int, int, int, float]]]:
    """Pares de sustitución por partido, con caché en disco (solo se leen los crudos nuevos)."""
    import pickle

    cache: Dict[str, list] = {}
    if os.path.exists(SUBS_CACHE):
        try:
            with open(SUBS_CACHE, "rb") as f:
                cache = pickle.load(f)
        except (OSError, ValueError, EOFError, pickle.UnpicklingError):
            cache = {}
    missing = [g for g in game_ids if g not in cache and os.path.exists(_raw_path(g))]
    for g in missing:
        try:
            with gzip.open(_raw_path(g), "rt", encoding="utf-8") as f:
                cache[g] = substitution_pairs(json.load(f))
        except (OSError, ValueError):
            continue
    if missing:
        tmp = f"{SUBS_CACHE}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(cache, f)
        os.replace(tmp, SUBS_CACHE)
    return {g: cache[g] for g in game_ids if g in cache}
