"""
OmniPicks — NBA v2 (N6): predicción de los partidos del día con el modelo de jugadores.

Junta todo:
  ratings de jugador (N1, tabla nba_player_ratings) + plantillas, quién reemplaza a quién y
  minutos con LightGBM (N2) + reporte de lesiones vigente + contexto y uso absorbido (N4) +
  simulación con sorteo de bajas (N3).

Para cada baja calcula su impacto (con y sin él, con incertidumbre), a quién van sus
minutos y cómo rindió su equipo con él y sin él en cancha esta temporada.

Todo es síncrono y gratis (MySQL + una llamada por equipo a nba_api para la plantilla,
en caché); se llama desde el motor de predicción con asyncio.to_thread.
"""
import logging
import time
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.services.nba_game_model import (MODEL_NAME, ModelParams, PaceState, ScheduleState, availability_scenarios,
                                         simulate_mixture, structural, team_strength)
from app.services.nba_injuries import PLAY_PROBABILITY, norm_name
from app.services.nba_lineups import (ROTATION_MIN, Candidate, MinutesModel, ReplacementContext, expected_minutes,
                                      load_tracker, minutes_features, team_ids_by_name)
from app.services.nba_ratings import MODEL_VERSION, meta_for

logger = logging.getLogger(__name__)

_roster_cache: Dict[Tuple[int, str], Tuple[float, set]] = {}
ROSTER_TTL = 6 * 3600
POSITION_LABEL = {"G": "base/escolta", "G-F": "escolta/alero", "F-G": "alero/escolta", "F": "alero",
                  "F-C": "ala-pívot", "C-F": "pívot/ala", "C": "pívot"}


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


class TeamState:
    """Un equipo listo para simular: candidatos, reporte de lesiones y ratings."""

    def __init__(self, pred: "NBAv2Predictor", tid: int):
        self.tid = tid
        cands = _team_candidates(pred.tracker, tid, pred.day, current_roster(tid, pred.season))
        by_name = {norm_name(pred.names.get(c.player, "")): c.player for c in cands}
        self.probs: Dict[int, float] = {}
        self.report = []
        for r in pred.injuries.get(tid, []):
            pid = by_name.get(norm_name(r["player_name"]))
            if pid is None:
                continue
            p = PLAY_PROBABILITY.get(r["status"], 0.5)
            self.probs[pid] = p
            self.report.append((pid, r, p))
        # Vuelve de una ausencia: jugador de rotación que ya no aparece en el reporte
        self.returning = {c.player for c in cands if c.player not in self.probs and c.rate < 0.6 and c.base >= ROTATION_MIN}
        self.cands = cands
        self.ids = [c.player for c in cands]
        self.o = np.array([pred.o.get(p, pred.repl_o) for p in self.ids])
        self.d = np.array([pred.d.get(p, pred.repl_d) for p in self.ids])
        self.unc = np.array([pred.unc.get(p, 3.0) for p in self.ids])
        self.usage = np.array([pred.usage.get(p, 18.0) for p in self.ids])
        self.ctx = ReplacementContext(pred.tracker.replacements.get(tid, {}), pred.height, pred.position)
        last = pred.sched.games.get(tid, [])
        self.rest = (pred.day - last[-1][0]).days if last else 3
        self.pred = pred

    def minutes(self, pp: Dict[int, float]) -> np.ndarray:
        """Minutos esperados con `pp` = probabilidad de jugar de los del reporte."""
        outs = {p: 0.0 for p, v in pp.items() if v <= 0.0}
        sure = {p: 1.0 for p, v in pp.items() if v > 0.0}
        m_prop = expected_minutes(self.cands, {**sure, **outs}, self.returning)[0]
        model = self.pred.minutes_model
        if model is None:
            return expected_minutes(self.cands, pp, self.returning)[0]
        m_rep = expected_minutes(self.cands, {**sure, **outs}, self.returning, ctx=self.ctx)[0]
        X = minutes_features(self.pred.tracker, self.cands, self.pred.day, outs, self.returning,
                             m_prop, m_rep, self.o + self.d, self.rest)
        prob = np.array([pp.get(p, 1.0) if 0.0 < pp.get(p, 1.0) < 1.0 else 1.0 for p in self.ids])
        out_mask = np.array([p in outs for p in self.ids])
        return model.predict(X, out_mask, prob)

    def absence_features(self) -> Tuple[float, float]:
        """Minutos de rotación y uso que se esperan ausentes (para la corrección de contexto)."""
        miss = usage = 0.0
        for i, c in enumerate(self.cands):
            p = self.probs.get(c.player)
            if p is None or c.base < ROTATION_MIN:
                continue
            miss += (1.0 - p) * c.base / 10.0
            usage += (1.0 - p) * self.usage[i] * c.base / 48.0 / 10.0
        return miss, usage


class NBAv2Predictor:
    """Se arma una vez por corrida (lee la base) y predice cada partido."""

    def __init__(self, engine, day: date):
        self.day = day
        self.engine = engine
        self.params = ModelParams.load()
        self.meta = meta_for(day)
        self.team_id = team_ids_by_name()
        self.minutes_model = MinutesModel.load()
        if self.minutes_model is None:
            logger.warning("NBA v2: sin modelo de minutos (ml/saved_models/nba_minutes.txt); se usa el reparto proporcional")
        self._load()

    def _load(self):
        day = self.day
        season_start_year = day.year if day.month >= 10 else day.year - 1
        self.season = f"{season_start_year}-{(season_start_year + 1) % 100:02d}"
        with self.engine.connect() as conn:
            ratings = pd.read_sql(text(
                "SELECT player_id, o_rating, d_rating, uncertainty, usage_rate FROM nba_player_ratings WHERE model_version = :v "
                "AND as_of_date = (SELECT MAX(as_of_date) FROM nba_player_ratings WHERE model_version = :v AND as_of_date <= :d)"),
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
            players = pd.read_sql(text("SELECT player_id, position, height_in FROM nba_players"), conn)
            stints = pd.read_sql(text(
                "SELECT s.game_id, g.home_team_id, g.away_team_id, s.home_players, s.away_players, s.home_pts, s.away_pts, "
                "s.home_poss, s.away_poss FROM nba_stints s JOIN nba_games g USING (game_id) WHERE g.season = :s"),
                conn, params={"s": self.season})
        if ratings.empty:
            raise RuntimeError("no hay ratings de jugador guardados (correr ml/scripts/nba_ratings_build.py)")
        pid = ratings.player_id.astype(int)
        self.o = dict(zip(pid, ratings.o_rating.astype(float)))
        self.d = dict(zip(pid, ratings.d_rating.astype(float)))
        self.unc = dict(zip(pid, ratings.uncertainty.fillna(3.0).astype(float)))
        self.usage = dict(zip(pid, ratings.usage_rate.fillna(18.0).astype(float)))
        tot = ratings.o_rating + ratings.d_rating
        low = ratings[tot <= tot.quantile(0.15)]                 # nivel de un suplente para quien no tiene rating
        self.repl_o, self.repl_d = float(low.o_rating.mean()), float(low.d_rating.mean())
        self.height = {int(r.player_id): float(r.height_in) for r in players.itertuples() if r.height_in == r.height_in and r.height_in}
        self.position = {int(r.player_id): r.position for r in players.itertuples() if r.position}
        self.stints = stints

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

    # ── Con y sin él en cancha (stints de esta temporada) ─────────────
    def on_off(self, tid: int, player: int) -> Optional[Dict]:
        st = self.stints
        if st.empty:
            return None
        home = st.home_team_id == tid
        away = st.away_team_id == tid
        mine = st[home | away]
        if mine.empty:
            return None
        is_home = (mine.home_team_id == tid).to_numpy()
        lineup = np.where(is_home, mine.home_players, mine.away_players)
        pf = np.where(is_home, mine.home_pts, mine.away_pts).astype(float)
        pa = np.where(is_home, mine.away_pts, mine.home_pts).astype(float)
        possf = np.where(is_home, mine.home_poss, mine.away_poss).astype(float)
        possa = np.where(is_home, mine.away_poss, mine.home_poss).astype(float)
        on = np.array([str(player) in s.split(",") for s in lineup])

        def net(mask):
            pf_, pa_, a, b = pf[mask].sum(), pa[mask].sum(), possf[mask].sum(), possa[mask].sum()
            if a < 50 or b < 50:
                return None, a + b
            return 100.0 * (pf_ / a - pa_ / b), a + b

        on_net, on_poss = net(on)
        off_net, off_poss = net(~on)
        if on_net is None and off_net is None:
            return None
        return {"on_net": None if on_net is None else round(on_net, 1),
                "off_net": None if off_net is None else round(off_net, 1),
                "on_poss": int(on_poss / 2), "off_poss": int(off_poss / 2)}

    # ── Un partido ────────────────────────────────────────────────────
    def predict(self, home_name: str, away_name: str, season_type: str = "Regular Season") -> Dict:
        h_id, a_id = self.team_id.get(home_name.lower()), self.team_id.get(away_name.lower())
        if h_id is None or a_id is None:
            raise KeyError(f"equipo desconocido: {home_name} / {away_name}")
        H, A = TeamState(self, h_id), TeamState(self, a_id)
        ctx = self.sched.game(self.day, self.season, season_type, h_id, a_id)
        (mh, uh), (ma, ua) = H.absence_features(), A.absence_features()
        ctx["miss_min"], ctx["usage_gap"] = mh - ma, uh - ua
        dm, dt = self.params.correction(ctx)

        lp, lppp = self.pace.league_pace(), self.pace.league_ppp()
        eff, edge = self.meta["league_eff"], self.meta["home_edge"]
        pace_h, pace_a = self.pace.pace(h_id), self.pace.pace(a_id)
        scale = pace_h * pace_a / lp * lppp / 100.0 / eff          # pts/100 → puntos del partido

        certain = {T.tid: {p: v for p, v in T.probs.items() if v in (0.0, 1.0)} for T in (H, A)}
        uncertain = {p: v for T in (H, A) for p, v in T.probs.items() if 0.0 < v < 1.0}
        rng = np.random.default_rng(abs(hash((home_name, away_name, self.day.isoformat()))) % 2 ** 32)
        means, nets = [], {h_id: 0.0, a_id: 0.0}
        for scen, w in availability_scenarios(uncertain, rng):
            st = []
            for T in (H, A):
                pp = {p: scen.get(p, certain[T.tid].get(p, v)) for p, v in T.probs.items()}
                m = T.minutes(pp)
                s = team_strength(m, T.o, T.d)
                nets[T.tid] += w * (s.off + s.deff)
                st.append(s)
            hp, ap = structural(st[0], st[1], pace_h, pace_a, lp, lppp, eff, edge)
            means.append((hp + (dm + dt) / 2.0, ap + (dt - dm) / 2.0, w))
        sim = simulate_mixture(means, self.params)

        injuries, teams = [], {}
        for side, T in (("home", H), ("away", A)):
            full = T.minutes({p: 1.0 for p in T.probs})
            full_net = float((full * (T.o + T.d)).sum() / 48.0)
            teams[side] = {"strength_full": round(full_net, 2), "strength_today": round(nets[T.tid], 2),
                           "points_lost": round(scale * (nets[T.tid] - full_net), 1)}
            for pid, r, p in T.report:
                i = T.ids.index(pid)
                others = {q: v for q, v in T.probs.items() if q != pid}
                with_m = T.minutes({**others, pid: 1.0})
                without_m = T.minutes({**others, pid: 0.0})
                if with_m[i] < 1.0:
                    continue                                    # lleva tanto fuera que ya no cuenta en la rotación
                impact = scale * float(((without_m - with_m) * (T.o + T.d)).sum() / 48.0)
                delta = without_m - with_m
                gains = sorted(((j, float(delta[j])) for j in range(len(T.ids)) if j != i and delta[j] > 0.5), key=lambda x: -x[1])
                injuries.append({
                    "team": side, "player": r["player_name"], "status": r["status"], "injury": r["injury_type"],
                    "play_probability": p, "usual_minutes": round(float(with_m[i]), 1),
                    "position": self.position.get(pid), "position_label": POSITION_LABEL.get(self.position.get(pid) or "", None),
                    "o_rating": round(float(T.o[i]), 1), "d_rating": round(float(T.d[i]), 1),
                    "impact_if_out": round(impact, 1),                  # puntos de margen para su equipo
                    "impact_sd": round(scale * float(T.unc[i]) * float(with_m[i]) / 48.0, 1),
                    "expected_impact": round(impact * (1.0 - p), 1),
                    "on_off": self.on_off(T.tid, pid),
                    "minutes_to": [{"player": self.names.get(T.ids[j], str(T.ids[j])), "minutes": round(mm, 1),
                                    "position": self.position.get(T.ids[j])} for j, mm in gains[:4]],
                })
        injuries.sort(key=lambda x: x["impact_if_out"])

        hw = sim["home_win_prob"]
        return {
            "home_win_prob": round(hw, 4), "away_win_prob": round(1.0 - hw, 4), "draw_prob": 0.0,
            "predicted_home_score": round(sim["exp_home"], 1), "predicted_away_score": round(sim["exp_away"], 1),
            "predicted_total": round(sim["exp_total"], 1), "predicted_spread": round(-sim["exp_margin"], 1),
            "confidence": self._confidence(h_id, a_id, uncertain),
            "simulation_details": {
                "source": "NBA v2: ratings de jugador (RAPM + box score) × minutos con bajas (LightGBM) + contexto",
                "model": MODEL_NAME,
                "num_simulations": sim["num_simulations"], "availability_scenarios": sim["scenarios"],
                "margin_range_80": [round(sim["margin_p10"], 1), round(sim["margin_p90"], 1)],
                "total_range_80": [round(sim["total_p10"], 1), round(sim["total_p90"], 1)],
                "most_likely_score": sim["most_likely_score"],
                "home_win_prob_uncalibrated": round(sim["home_win_prob_raw"], 4),
                "context": {k: round(v, 2) for k, v in ctx.items()},
                "context_adjustment": {"margin": round(dm, 2), "total": round(dt, 2)},
                "minutes_model": "lightgbm" if self.minutes_model else "proporcional",
                "teams": teams,
                "injuries": injuries,
            },
        }

    def _confidence(self, h_id: int, a_id: int, uncertain: Dict[int, float]) -> float:
        """Qué tanto respaldo tiene la predicción (mismo sentido que v1: cuántos datos hay).

        Sube con los partidos jugados esta temporada por el equipo con menos y baja con
        cada jugador en duda (sus minutos son un sorteo).
        """
        start = self.sched.season_start.get(self.season, self.day)
        gp = min(sum(1 for d, _ in self.sched.games.get(t, []) if d >= start) for t in (h_id, a_id))
        conf = 0.55 + 0.30 * min(1.0, gp / 30.0) - 0.02 * len(uncertain)
        return round(float(min(max(conf, 0.40), 0.85)), 2)


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
