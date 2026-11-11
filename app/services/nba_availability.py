"""
OmniPicks — NBA: probabilidad de jugar aprendida de nuestros propios datos.

1. Por estado del reporte de lesiones (Out, Day-To-Day, Questionable…): de los jugadores que
   tenían ese estado el día del partido, cuántos jugaron. Se mezcla con la probabilidad
   inicial (como si fueran PRIOR_GAMES casos), así que con pocos datos manda la inicial.
2. Descanso en back-to-back: cuánto más falta cada jugador de rotación en el segundo día de
   un back-to-back que en un partido normal. Solo se aplica a quien NO está en el reporte
   (si está, manda el reporte), y solo si el exceso es claro.

El resultado se guarda en ml/saved_models/nba_availability.json y lo recalcula la tarea
diaria; `play_probability(status)` y `b2b_rest_probability(player)` lo leen.
"""
import json
import logging
import os
from datetime import datetime, time, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ml", "saved_models", "nba_availability.json"))
PRIOR_GAMES = 20.0                  # peso de la probabilidad inicial, en casos observados
REPORT_CUTOFF = time(16, 0)         # estado vigente a las 16:00 hora local del servidor (~18:00 ET)
B2B_PRIOR_GAMES = 8.0               # peso de "no descansa más de lo normal" por jugador
B2B_MIN_BASE = 24.0                 # solo jugadores de rotación fuerte
B2B_MIN_EXCESS = 0.05               # por debajo de esto no se aplica

_cache: Dict = {"mtime": None, "data": {}}


def _load() -> Dict:
    try:
        mtime = os.path.getmtime(PATH)
    except OSError:
        return {}
    if _cache["mtime"] != mtime:
        try:
            with open(PATH) as f:
                _cache["data"] = json.load(f)
            _cache["mtime"] = mtime
        except (OSError, ValueError):
            return {}
    return _cache["data"]


def play_probability(status: str) -> float:
    from app.services.nba_injuries import PLAY_PROBABILITY
    learned = (_load().get("status") or {}).get(status)
    if learned is not None:
        return float(learned["p"])
    return PLAY_PROBABILITY.get(status, 0.5)


B2B_APPLY = 0.5                     # se aplica la mitad del exceso: parte de esas ausencias fueron
                                    # lesiones que sí salen en el reporte (sin reportes viejos no se puede separar)


def b2b_rest_probability(player_id: int) -> float:
    """Probabilidad de jugar el segundo día de un back-to-back si no está en el reporte."""
    rec = (_load().get("b2b_rest") or {}).get(str(player_id))
    return 1.0 if rec is None else 1.0 - B2B_APPLY * (1.0 - float(rec["p_play"]))


# ══════════════════════════════════════════════════════════════════════
# Cálculo
# ══════════════════════════════════════════════════════════════════════

def calibrate_status(engine) -> Dict:
    from app.services.nba_injuries import PLAY_PROBABILITY, norm_name
    from app.services.nba_lineups import team_ids_by_name

    with engine.connect() as conn:
        rep = pd.read_sql(text("SELECT player_name, team_name, status, captured_at, ended_at FROM nba_injury_reports"), conn)
        if rep.empty:
            return {}
        first = rep.captured_at.min().date()
        games = pd.read_sql(text(
            "SELECT game_id, game_date, home_team_id, away_team_id FROM nba_games WHERE game_date >= :d AND home_pts IS NOT NULL"),
            conn, params={"d": first})
        box = pd.read_sql(text(
            "SELECT game_id, team_id, player_name, minutes FROM nba_player_games WHERE game_date >= :d"), conn, params={"d": first})
    ids = team_ids_by_name()
    rep["team_id"] = rep.team_name.str.lower().map(ids)
    rep["key"] = rep.player_name.map(norm_name)
    played = {(r.game_id, int(r.team_id), norm_name(r.player_name)) for r in box.itertuples() if r.minutes > 0}
    # Jugadores que jugaron con ese equipo en los últimos ~12 meses (para no contar a quien ya no está)
    with engine.connect() as conn:
        roster = pd.read_sql(text("SELECT DISTINCT team_id, player_name FROM nba_player_games WHERE game_date >= :d"),
                             conn, params={"d": first - timedelta(days=365)})
    on_team = {(int(r.team_id), norm_name(r.player_name)) for r in roster.itertuples()}

    obs = []
    for g in games.itertuples():
        cutoff = datetime.combine(g.game_date, REPORT_CUTOFF)
        for tid in (int(g.home_team_id), int(g.away_team_id)):
            cur = rep[(rep.team_id == tid) & (rep.captured_at <= cutoff) & (rep.ended_at.isna() | (rep.ended_at > cutoff))]
            for r in cur.drop_duplicates("key", keep="last").itertuples():
                if r.status == "Out For Season":
                    continue
                obs.append({"status": r.status, "played": (g.game_id, tid, r.key) in played,
                            "known": (tid, r.key) in on_team})
    # Reporte oficial de la NBA: estado del último reporte del día antes de las 18:00 ET
    with engine.connect() as conn:
        try:
            off = pd.read_sql(text(
                "SELECT report_time, game_date, team_name, player_name, status FROM nba_official_injuries"), conn)
        except Exception:
            off = pd.DataFrame()
    if len(off):
        off["team_id"] = off.team_name.str.lower().map(ids)
        off["key"] = off.player_name.map(norm_name)
        off = off[off.report_time <= pd.to_datetime(off.game_date) + pd.Timedelta(hours=18)]
        last = off.sort_values("report_time").drop_duplicates(["game_date", "team_id", "key"], keep="last")
        gmap = {}
        for g in games.itertuples():
            gmap[(g.game_date, int(g.home_team_id))] = g.game_id
            gmap[(g.game_date, int(g.away_team_id))] = g.game_id
        for r in last.itertuples():
            gid = gmap.get((r.game_date, int(r.team_id) if r.team_id == r.team_id else -1))
            if gid is None:
                continue
            obs.append({"status": r.status, "played": (gid, int(r.team_id), r.key) in played,
                        "known": (int(r.team_id), r.key) in on_team})
    df = pd.DataFrame(obs)
    if len(df):
        df = df[df.known]                     # solo jugadores que de verdad están en la plantilla
    out = {}
    for status, prior in PLAY_PROBABILITY.items():
        sub = df[df.status == status] if len(df) else df
        n, k = (len(sub), int(sub.played.sum())) if len(sub) else (0, 0)
        p = (k + PRIOR_GAMES * prior) / (n + PRIOR_GAMES)
        out[status] = {"p": round(p, 3), "observed": n, "played": k, "prior": prior}
    return out


def b2b_rest_rates(engine, seasons: int = 3) -> Dict:
    """Exceso de ausencias en el segundo día de un back-to-back, por jugador de rotación."""
    with engine.connect() as conn:
        games = pd.read_sql(text(
            "SELECT game_id, game_date, season, season_type, home_team_id, away_team_id FROM nba_games "
            "WHERE home_pts IS NOT NULL ORDER BY game_date"), conn)
        box = pd.read_sql(text("SELECT game_id, team_id, player_id, minutes FROM nba_player_games WHERE minutes > 0"), conn)
    keep = sorted(games.season.unique())[-seasons:]
    games = games[games.season.isin(keep) & (games.season_type == "Regular Season")]
    tg = pd.concat([games.assign(team_id=games.home_team_id), games.assign(team_id=games.away_team_id)])
    tg = tg.sort_values(["team_id", "game_date"])
    tg["prev"] = tg.groupby("team_id").game_date.shift(1)
    tg["b2b"] = (pd.to_datetime(tg.game_date) - pd.to_datetime(tg.prev)).dt.days == 1
    played = box.groupby(["game_id", "team_id"]).player_id.apply(set).to_dict()
    mins = box.set_index(["game_id", "team_id", "player_id"]).minutes

    # Jugadores de rotación de cada equipo: ventana móvil de 10 partidos
    stats: Dict[int, Dict[str, float]] = {}
    for tid, grp in tg.groupby("team_id"):
        window = []
        for g in grp.itertuples():
            here = played.get((g.game_id, tid), set())
            if len(window) >= 5:
                counts: Dict[int, list] = {}
                for gid, pl in window[-10:]:
                    for p in pl:
                        counts.setdefault(p, []).append(float(mins.get((gid, tid, p), 0.0)))
                for p, ms in counts.items():
                    if len(ms) >= 7 and np.mean(ms) >= B2B_MIN_BASE:
                        s = stats.setdefault(p, {"b2b_n": 0, "b2b_miss": 0, "n": 0, "miss": 0})
                        key = "b2b_" if g.b2b else ""
                        s[f"{key}n"] += 1
                        s[f"{key}miss"] += int(p not in here)
            window.append((g.game_id, here))
    league_b2b = sum(s["b2b_miss"] for s in stats.values()) / max(sum(s["b2b_n"] for s in stats.values()), 1)
    league = sum(s["miss"] for s in stats.values()) / max(sum(s["n"] for s in stats.values()), 1)
    out = {}
    for p, s in stats.items():
        if s["b2b_n"] < 3:
            continue
        base = (s["miss"] + 10 * league) / (s["n"] + 10)
        b2b = (s["b2b_miss"] + B2B_PRIOR_GAMES * (base + (league_b2b - league))) / (s["b2b_n"] + B2B_PRIOR_GAMES)
        excess = b2b - base
        if excess >= B2B_MIN_EXCESS:
            out[str(p)] = {"p_play": round(1.0 - excess, 3), "b2b_games": s["b2b_n"], "b2b_missed": s["b2b_miss"],
                           "normal_miss_rate": round(base, 3)}
    return {"players": out, "league_b2b_miss": round(league_b2b, 3), "league_miss": round(league, 3)}


def update(engine) -> Dict:
    status = calibrate_status(engine)
    b2b = b2b_rest_rates(engine)
    data = {"updated": datetime.now().isoformat(timespec="seconds"), "status": status,
            "b2b_rest": b2b["players"], "b2b_league": {k: v for k, v in b2b.items() if k != "players"}}
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    logger.info(f"Disponibilidad NBA: estados {[(k, v['p'], v['observed']) for k, v in status.items()]}, "
                f"{len(b2b['players'])} jugadores con descanso en back-to-back")
    return data
