"""
OmniPicks — NBA v2 (N6): predicción de los partidos del día con el modelo de jugadores.

Junta todo lo anterior:
  ratings de jugador (N1, tabla nba_player_ratings) + plantillas y minutos (N2) +
  reporte de lesiones vigente + contexto (N4) + simulación con sorteo de bajas (N3).

Todo es síncrono y gratis (MySQL + una llamada por equipo a nba_api para la plantilla,
en caché por día); se llama desde el motor de predicción con asyncio.to_thread.
"""
import logging
import time
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.services.nba_game_model import (MODEL_NAME, ModelParams, PaceState, ScheduleState, availability_scenarios,
                                         simulate_mixture, structural, team_strength)
from app.services.nba_injuries import PLAY_PROBABILITY, norm_name
from app.services.nba_lineups import (ROTATION_MIN, Candidate, absence_impact, expected_minutes, load_tracker,
                                      team_ids_by_name)
from app.services.nba_ratings import MODEL_VERSION, meta_for

logger = logging.getLogger(__name__)

_roster_cache: Dict[Tuple[int, str], Tuple[float, set]] = {}
ROSTER_TTL = 6 * 3600


def current_roster(team_id: int, season: str) -> Optional[set]:
    """Ids de la plantilla actual (nba_api CommonTeamRoster), con caché de 6 horas."""
    from app.services.nba_history import call_nba
    from nba_api.stats.endpoints import commonteamroster

    hit = _roster_cache.get((team_id, season))
    if hit and time.time() - hit[0] < ROSTER_TTL:
        return hit[1]
    try:
        df = call_nba(commonteamroster.CommonTeamRoster, retries=2, team_id=team_id, season=season)[0]
        ids = set(df["PLAYER_ID"].astype(int))
    except Exception as e:
        logger.warning(f"NBA v2: sin plantilla de {team_id} ({type(e).__name__}); se usa la del box score")
        return hit[1] if hit else None
    _roster_cache[(team_id, season)] = (time.time(), ids)
    return ids


def _team_candidates(tracker, team_id: int, day: date, roster: Optional[set]) -> List[Candidate]:
    """Candidatos del equipo: los del box score que siguen en la plantilla, más los recién llegados."""
    cands = tracker.candidates(team_id, day)
    if roster is None:
        return cands
    have = {c.player for c in cands}
    cands = [c for c in cands if c.player in roster]
    for p in roster - have:                               # traspasos: sus minutos vienen del equipo anterior
        games = tracker.player_games.get(p)
        if not games or (day - games[-1][0]).days > 200:
            continue
        mins = [m for (_, _, m) in games]
        base = 0.6 * float(np.mean(mins[-5:])) + 0.4 * float(np.mean(mins))
        recent = (day - games[-1][0]).days <= 10
        cands.append(Candidate(p, base, 1.0 if recent else 0.5, 0))
    return cands


class NBAv2Predictor:
    """Se arma una vez por corrida (lee la base) y predice cada partido."""

    def __init__(self, engine, day: date):
        self.day = day
        self.engine = engine
        self.params = ModelParams.load()
        self.meta = meta_for(day)
        self.team_id = team_ids_by_name()
        self._load()

    def _load(self):
        day = self.day
        season_start_year = day.year if day.month >= 10 else day.year - 1
        self.season = f"{season_start_year}-{(season_start_year + 1) % 100:02d}"
        with self.engine.connect() as conn:
            ratings = pd.read_sql(text(
                "SELECT player_id, o_rating, d_rating FROM nba_player_ratings WHERE model_version = :v AND as_of_date = "
                "(SELECT MAX(as_of_date) FROM nba_player_ratings WHERE model_version = :v AND as_of_date <= :d)"),
                conn, params={"v": MODEL_VERSION, "d": day})
            games = pd.read_sql(text(
                "SELECT game_id, season, season_type, game_date, home_team_id, away_team_id, home_pts, away_pts "
                "FROM nba_games WHERE game_date >= :a AND game_date < :d AND home_pts IS NOT NULL ORDER BY game_date, game_id"),
                conn, params={"a": date(season_start_year - 1, 9, 1), "d": day})
            poss = pd.read_sql(text(
                "SELECT game_id, team_id, SUM(fga) fga, SUM(fta) fta, SUM(tov) tov, SUM(oreb) oreb FROM nba_player_games "
                "WHERE game_date >= :a AND game_date < :d GROUP BY game_id, team_id"),
                conn, params={"a": date(season_start_year - 1, 9, 1), "d": day})
            inj = pd.read_sql(text(
                "SELECT player_name, team_name, status, injury_type, comment FROM nba_injury_reports WHERE is_current = 1"), conn)
        if ratings.empty:
            raise RuntimeError("no hay ratings de jugador guardados (correr ml/scripts/nba_ratings_build.py)")
        self.o = dict(zip(ratings.player_id.astype(int), ratings.o_rating.astype(float)))
        self.d = dict(zip(ratings.player_id.astype(int), ratings.d_rating.astype(float)))
        tot = ratings.o_rating + ratings.d_rating
        low = ratings[tot <= tot.quantile(0.15)]                 # nivel de un suplente para quien no tiene rating
        self.repl_o, self.repl_d = float(low.o_rating.mean()), float(low.d_rating.mean())

        # Calendario, ritmo y eficiencia de liga, reconstruidos en orden
        poss["p"] = poss.fga + 0.44 * poss.fta + poss.tov - poss.oreb
        game_poss = poss.groupby("game_id")["p"].mean()
        self.sched, self.pace = ScheduleState(), PaceState()
        games["game_date"] = pd.to_datetime(games.game_date).dt.date
        for season, g in games.groupby("season"):
            reg = g[g.season_type == "Regular Season"]
            if len(reg):
                self.sched.season_start[season] = reg.game_date.min()
                self.sched.season_end[season] = reg.game_date.max()
        self.sched.season_start.setdefault(self.season, day)
        for g in games.itertuples():
            self.sched.add(g.game_date, int(g.home_team_id), int(g.away_team_id))
            if g.game_id in game_poss.index:
                self.pace.new_season(g.season)
                self.pace.add(int(g.home_team_id), int(g.away_team_id), float(game_poss[g.game_id]), g.home_pts + g.away_pts)
        self.pace.new_season(self.season)

        self.tracker, self.names = load_tracker(self.engine, day)
        self.injuries: Dict[int, List[Dict]] = {}
        for team_name, grp in inj.groupby("team_name"):
            tid = self.team_id.get(team_name.lower())
            if tid is not None:
                self.injuries[tid] = grp.to_dict("records")

    # ── Un equipo ─────────────────────────────────────────────────────
    def _team(self, tid: int):
        cands = _team_candidates(self.tracker, tid, self.day, current_roster(tid, self.season))
        by_name = {norm_name(self.names.get(c.player, "")): c.player for c in cands}
        probs, report = {}, []
        for r in self.injuries.get(tid, []):
            pid = by_name.get(norm_name(r["player_name"]))
            if pid is None:
                continue
            p = PLAY_PROBABILITY.get(r["status"], 0.5)
            probs[pid] = p
            report.append((pid, r, p))
        reported = set(probs)
        # Vuelve de una ausencia: jugador de rotación que ya no aparece en el reporte
        returning = {c.player for c in cands if c.player not in reported and c.rate < 0.6 and c.base >= ROTATION_MIN}
        o = np.array([self.o.get(c.player, self.repl_o) for c in cands])
        d = np.array([self.d.get(c.player, self.repl_d) for c in cands])
        return cands, probs, report, returning, o, d

    def _confidence(self, h_id: int, a_id: int, uncertain: Dict[int, float]) -> float:
        """Qué tanto respaldo tiene la predicción (mismo sentido que v1: cuántos datos hay).

        Sube con los partidos jugados esta temporada por el equipo con menos y baja con
        cada jugador en duda (sus minutos son un sorteo).
        """
        start = self.sched.season_start.get(self.season, self.day)
        gp = min(sum(1 for d, _ in self.sched.games.get(t, []) if d >= start) for t in (h_id, a_id))
        conf = 0.55 + 0.30 * min(1.0, gp / 30.0) - 0.02 * len(uncertain)
        return round(float(min(max(conf, 0.40), 0.85)), 2)

    # ── Un partido ────────────────────────────────────────────────────
    def predict(self, home_name: str, away_name: str, season_type: str = "Regular Season") -> Dict:
        h_id, a_id = self.team_id.get(home_name.lower()), self.team_id.get(away_name.lower())
        if h_id is None or a_id is None:
            raise KeyError(f"equipo desconocido: {home_name} / {away_name}")
        H, A = self._team(h_id), self._team(a_id)
        ctx = self.sched.game(self.day, self.season, season_type, h_id, a_id)
        # Minutos de rotación ausentes (esperados), local − visita, por cada 10
        miss = 0.0
        for sign, (cands, probs, *_rest) in ((1.0, H), (-1.0, A)):
            miss += sign * sum(c.base * (1.0 - probs[c.player]) for c in cands
                               if c.player in probs and c.base >= ROTATION_MIN) / 10.0
        ctx["miss_min"] = miss
        dm, dt = self.params.correction(ctx)

        certain = {**{p: v for p, v in H[1].items() if v in (0.0, 1.0)}, **{p: v for p, v in A[1].items() if v in (0.0, 1.0)}}
        uncertain = {p: v for team in (H, A) for p, v in team[1].items() if 0.0 < v < 1.0}
        rng = np.random.default_rng(abs(hash((home_name, away_name, self.day.isoformat()))) % 2 ** 32)
        means = []
        lp, lppp = self.pace.league_pace(), self.pace.league_ppp()
        eff, edge = self.meta["league_eff"], self.meta["home_edge"]
        for scen, w in availability_scenarios(uncertain, rng):
            st = []
            for cands, probs, _, returning, o, d in (H, A):
                pp = {p: scen.get(p, certain.get(p, probs[p])) for p in probs}
                m, _ = expected_minutes(cands, pp, returning)
                st.append(team_strength(m, o, d))
            hp, ap = structural(st[0], st[1], self.pace.pace(h_id), self.pace.pace(a_id), lp, lppp, eff, edge)
            hp, ap = hp + (dm + dt) / 2.0, ap + (dt - dm) / 2.0
            means.append((hp, ap, w))
        sim = simulate_mixture(means, self.params)

        injuries = []
        scale = self.pace.pace(h_id) * self.pace.pace(a_id) / lp * lppp / 100.0 / eff
        for side, (cands, probs, report, returning, o, d) in (("home", H), ("away", A)):
            for pid, r, p in report:
                imp = absence_impact(cands, o + d, pid, pace=scale, other_probs=probs)
                if imp is None or imp["minutes"] < 1.0:
                    continue
                injuries.append({
                    "team": side, "player": r["player_name"], "status": r["status"], "injury": r["injury_type"],
                    "play_probability": p, "usual_minutes": round(imp["minutes"], 1),
                    "impact_if_out": round(imp["margin"], 1),           # puntos de margen para su equipo
                    "expected_impact": round(imp["margin"] * (1.0 - p), 1),
                    "minutes_to": [{"player": self.names.get(q, str(q)), "minutes": round(mm, 1)} for q, mm in imp["minutes_to"]],
                })
        injuries.sort(key=lambda x: x["impact_if_out"])

        hw = sim["home_win_prob"]
        return {
            "home_win_prob": round(hw, 4), "away_win_prob": round(1.0 - hw, 4), "draw_prob": 0.0,
            "predicted_home_score": round(sim["exp_home"], 1), "predicted_away_score": round(sim["exp_away"], 1),
            "predicted_total": round(sim["exp_total"], 1), "predicted_spread": round(-sim["exp_margin"], 1),
            "confidence": self._confidence(h_id, a_id, uncertain),
            "simulation_details": {
                "source": "NBA v2: ratings de jugador (RAPM + box score) × minutos con bajas + contexto",
                "model": MODEL_NAME,
                "num_simulations": sim["num_simulations"], "availability_scenarios": sim["scenarios"],
                "margin_range_80": [round(sim["margin_p10"], 1), round(sim["margin_p90"], 1)],
                "total_range_80": [round(sim["total_p10"], 1), round(sim["total_p90"], 1)],
                "most_likely_score": sim["most_likely_score"],
                "home_win_prob_uncalibrated": round(sim["home_win_prob_raw"], 4),
                "context": {k: round(v, 2) for k, v in ctx.items()},
                "context_adjustment": {"margin": round(dm, 2), "total": round(dt, 2)},
                "injuries": injuries,
            },
        }


def predict_games_sync(day: date, games: List[Tuple[int, str, str]]) -> Dict[int, Dict]:
    """[(id del partido, local, visita)] → {id: resultado}. Abre su propia conexión síncrona."""
    from sqlalchemy import create_engine
    from app.config import get_settings

    engine = create_engine(get_settings().mysql_url_sync, pool_pre_ping=True)
    try:
        model = NBAv2Predictor(engine, day)
        out = {}
        for gid, home, away in games:
            try:
                out[gid] = model.predict(home, away)
            except Exception as e:
                logger.warning(f"NBA v2: {away} @ {home} sin predicción ({type(e).__name__}: {e})")
        return out
    finally:
        engine.dispose()
