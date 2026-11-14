"""
OmniPicks — NFL v2: pronóstico de producción (fases 1–5 de docs/PLAN_NFL.md).

  margen = regresión sobre [EPA ajustado por rival, rating por marcador, ajuste y calidad del QB titular]
  total  = regresión sobre [rating de anotación, EPA esperado de los dos ataques, ritmo, clima]

Las probabilidades usan la distribución real del margen (números clave 3 y 7) y se calibran con
el histórico. Contra la línea de cierre el backtest no encontró ventaja (fase 5): las de spread y
total quedan casi en 50%, así que el modelo rara vez recomienda algo, y no recomienda moneyline.

Entrenar (`train`) recorre todo el histórico semana a semana igual que el backtest (~15 s) y
guarda los parámetros en ml/saved_models/nfl_v2.json. Se vuelve a entrenar una vez al día.
"""
import json
import logging
import os
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from app.services.nfl_history import (current_season, game_weather, load_pbp, load_schedules, normalize_teams,
                                      team_games)
from app.services.nfl_qb import QBParams, QBRatingEngine, qb_games, qb_margin_features
from app.services.nfl_ratings import (NFLRatingEngine, PointsRatingEngine, RatingParams, ScoringRatingEngine,
                                      expected_margin_epa)
from app.services.nfl_score import MarginDistribution, total_probs

logger = logging.getLogger(__name__)

MODEL_NAME = "nfl_v2"
PARAMS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ml", "saved_models", "nfl_v2.json"))
FIRST_TRAIN_SEASON = 2011       # 2010 calienta los ratings

EPA_P = RatingParams(half_life=16, offseason_gap=12, ridge=300)
PTS_P, PTS_RIDGE = RatingParams(half_life=16, offseason_gap=12), 8.0
SCORE_P, SCORE_RIDGE = RatingParams(half_life=8, offseason_gap=12), 8.0
QB_P = QBParams()

NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC", "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "Seattle Seahawks": "SEA",
    "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


def abbr(name: str) -> str:
    return NAME_TO_ABBR.get(name, (name or "?")[:3].upper())


# ══════════════════════════════════════════════════════════════════════
# Variables de cada partido (las mismas del backtest)
# ══════════════════════════════════════════════════════════════════════

MARGIN_FEATURES = ["epa", "points", "qb_adj", "qb_raw"]
# Lluvia: en el histórico resta ~3.4 pts y en esos partidos el error baja de 10.9 a 10.4.
# La nieve (39 partidos) empeoraba el error, así que no entra.
# QB en el total (calidad de los dos titulares y su cambio contra el habitual) y EPA sin pérdidas de
# balón: error del total 2021–2026 de 10.57 a 10.43 (mercado 10.34). Ver ml/scripts/nfl_features_eval.py.
TOTAL_FEATURES = ["scoring", "epa_total", "pace", "indoor", "wind", "cold", "rain", "qb_sum", "qb_adj_sum", "epant_total"]
CALIB_HALF_LIFE = 3.0     # temporadas: la calibración contra la línea pesa más lo reciente


class Features:
    """Motores de ratings y cachés por semana. `row(...)` da las variables de un partido."""

    def __init__(self, sched: pl.DataFrame, pbp: pl.DataFrame):
        played = sched.filter(pl.col("result").is_not_null())
        neutral = sched.filter(pl.col("location") == "Neutral")["game_id"].to_list()
        tg, qbg = team_games(pbp), qb_games(pbp)
        self.epa = NFLRatingEngine(tg, EPA_P, neutral_games=neutral)
        self.epant = NFLRatingEngine(tg, EPA_P, neutral_games=neutral, value="epant")
        self.pts = PointsRatingEngine(played, PTS_P)
        self.score = ScoringRatingEngine(played, SCORE_P)
        self.qb = QBRatingEngine(qbg, QB_P)
        self.qb_names = dict(zip(qbg["id"].to_list(), qbg["name"].to_list()))
        wx = game_weather(pbp)
        self.wx = {g: (r, s) for g, r, s in zip(wx["game_id"].to_list(), wx["rain"].to_list(), wx["snow"].to_list())}
        self._cache: Dict[Tuple[int, int], Tuple] = {}
        self._pace = self._pace_snapshots(pbp, sched)

    @staticmethod
    def _pace_snapshots(pbp, sched, half_life_games: float = 8.0):
        plays = normalize_teams(pbp).group_by(["season", "week", "posteam"]).agg(n=pl.len())
        d = 0.5 ** (1.0 / half_life_games)
        by_week = defaultdict(list)
        for r in plays.iter_rows(named=True):
            by_week[(r["season"], r["week"])].append((r["posteam"], r["n"]))
        keys = sorted(set(by_week) | set(zip(sched["season"].to_list(), sched["week"].to_list())))
        num, den, snap = defaultdict(float), defaultdict(float), {}
        for k in keys:
            lg = sum(num.values()) / max(sum(den.values()), 1e-9) if den else 62.0
            snap[k] = ({t: num[t] / den[t] for t in den}, lg)
            for t, n in by_week.get(k, []):
                num[t] = d * num[t] + n
                den[t] = d * den[t] + 1.0
        return snap

    def week(self, season: int, week: int):
        key = (season, week)
        if key not in self._cache:
            self._cache[key] = (self.epa.at(season, week), self.pts.at(season, week, ridge=PTS_RIDGE),
                                self.score.at(season, week, ridge=SCORE_RIDGE), self.qb.at(season, week),
                                self.epant.at(season, week))
        return self._cache[key]

    def row(self, r: Dict) -> Tuple[Dict[str, float], Dict[str, float], Dict]:
        season, wk, home, away = r["season"], r["week"], r["home_team"], r["away_team"]
        neutral = r.get("location") == "Neutral"
        we, (hfa, st), (mu_s, h_s, off_s, def_s), wq, wnt = self.week(season, wk)
        x_epa, x_te = expected_margin_epa(we, home, away, neutral)
        adj, raw = qb_margin_features(wq, home, away, r.get("home_qb_id"), r.get("away_qb_id"))
        m = {"epa": x_epa, "points": (0.0 if neutral else hfa) + st.get(home, 0.0) - st.get(away, 0.0),
             "qb_adj": adj, "qb_raw": raw}
        pace, lg = self._pace.get((season, wk), ({}, 62.0))
        indoor = r.get("roof") in ("dome", "closed")
        wind = 0.0 if indoor or r.get("wind") is None else float(r["wind"])
        temp = 65.0 if indoor or r.get("temp") is None else float(r["temp"])
        # Lluvia y nieve: la observada (histórico) o la probabilidad del pronóstico (partidos por jugar)
        rain, snow = self.wx.get(r.get("game_id"), (0.0, 0.0))
        if r.get("rain") is not None:
            rain = float(r["rain"])
        if r.get("snow") is not None:
            snow = float(r["snow"])
        if indoor:
            rain = snow = 0.0
        t = {"scoring": 2 * mu_s + (0.0 if neutral else h_s) + off_s.get(home, 0) - def_s.get(away, 0)
                        + off_s.get(away, 0) - def_s.get(home, 0),
             "epa_total": x_te, "pace": pace.get(home, lg) + pace.get(away, lg) - 2 * lg,
             "indoor": 1.0 if indoor else 0.0, "wind": max(0.0, wind - 10.0), "cold": max(0.0, 40.0 - temp),
             "rain": rain, "snow": snow,
             "qb_sum": (wq.of(r.get("home_qb_id")) + wq.of(r.get("away_qb_id"))) * 38.0,
             "qb_adj_sum": (wq.adjustment(home, r.get("home_qb_id")) + wq.adjustment(away, r.get("away_qb_id"))) * 38.0,
             "epant_total": expected_margin_epa(wnt, home, away, neutral)[1]}
        info = {
            "home_net_epa": round(we.net(home), 3), "away_net_epa": round(we.net(away), 3),
            "home_points_rating": round(st.get(home, 0.0), 1), "away_points_rating": round(st.get(away, 0.0), 1),
            "home_field": round(0.0 if neutral else hfa, 1),
            "home_qb": self._qb_info(wq, home, r.get("home_qb_id"), r.get("home_qb_name")),
            "away_qb": self._qb_info(wq, away, r.get("away_qb_id"), r.get("away_qb_name")),
            "indoor": indoor, "wind": r.get("wind"), "temp": r.get("temp"),
            "rain": round(rain, 2), "snow": round(snow, 2), "weather_source": r.get("weather_source"),
        }
        return m, t, info

    def _qb_info(self, wq, team, qb_id, qb_name):
        return {"name": qb_name or self.qb_names.get(qb_id), "rating": round(wq.of(qb_id), 3),
                "dropbacks": round(wq.dropbacks.get(qb_id, 0.0)), "known": qb_id in wq.rating,
                "team_usual": round(wq.team_qb.get(team, wq.of(qb_id)), 3),
                "adjustment_pts_epa": round(wq.adjustment(team, qb_id) * 38.0, 1)}


# ══════════════════════════════════════════════════════════════════════
# Entrenamiento y parámetros
# ══════════════════════════════════════════════════════════════════════

def _fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.c_[np.ones(len(y)), X]
    return np.linalg.solve(A.T @ A + 1e-6 * np.eye(A.shape[1]), A.T @ y)


def _calib_k(z: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    w = np.ones(len(z)) if w is None else w
    nll = lambda k: -np.sum(w * (y * np.log(norm.cdf(k * z) + 1e-12) + (1 - y) * np.log(1 - norm.cdf(k * z) + 1e-12)))
    return float(minimize_scalar(nll, bounds=(0.0, 1.5), method="bounded").x)


# ══════════════════════════════════════════════════════════════════════
# Bajas por jugador (v3, fases 1 y 3): RAPM + probabilidad de jugar de todas las posiciones
# ══════════════════════════════════════════════════════════════════════

PLAYS_PER_GAME = 62.0
INJURY_QUANTILE = 70          # el ajuste por bajas se aplica cuando la diferencia está en el 30% más alto


def _train_players(pbp, played: pl.DataFrame, mu: np.ndarray, y: np.ndarray) -> Optional[Dict]:
    """Ajuste por bajas: margen = a + b·v2 + c·fuerza por jugadores + d·bajas previstas, con las
    temporadas que tienen participación (2018–). Se aplica solo si la diferencia de bajas es grande."""
    from app.services.nfl_players import (RapmParams, expected_usage, fit_rapm, injury_adjustment, learn_availability,
                                          load_injury_reports, load_participation, play_table, replacement_levels,
                                          team_strength, usage_from_plays, values_to_dict)
    cur = current_season()
    prm = RapmParams(season_half_life=2.0, ridge={"QB": 50, "default": 800})
    plays = play_table(pbp, load_participation(range(2016, cur)))
    if plays.height == 0:
        return None
    use = usage_from_plays(plays)
    exp = expected_usage(use)
    inj = load_injury_reports(range(2016, cur))
    table = learn_availability(inj, use, exp)
    report = {}
    for r in inj.iter_rows(named=True):
        report.setdefault((r["season"], r["week"], r["team"]), {})[r["gsis_id"]] = (r["report_status"], r["practice_status"])
    seasons = sorted(set(plays["season"].unique().to_list()))
    values = {s: fit_rapm(plays, s, prm) for s in seasons if s >= 2018}
    repl = {s: replacement_levels(v) for s, v in values.items()}
    X, Y, XI = [], [], []
    for i, r in enumerate(played.iter_rows(named=True)):
        s = r["season"]
        if s not in values or np.isnan(mu[i]):
            continue
        eh, ea = exp.get((s, r["week"], r["home_team"])), exp.get((s, r["week"], r["away_team"]))
        if not (eh and ea and eh["o"] and ea["o"]):
            continue
        v, rp = values[s], repl[s]
        oh, dh = team_strength(v, eh); oa, da = team_strength(v, ea)
        lh, _ = injury_adjustment(v, eh, report.get((s, r["week"], r["home_team"]), {}), table, rp)
        la, _ = injury_adjustment(v, ea, report.get((s, r["week"], r["away_team"]), {}), table, rp)
        xi = (lh - la) * PLAYS_PER_GAME
        X.append([mu[i], ((oh + dh) - (oa + da)) * PLAYS_PER_GAME, xi]); Y.append(y[i]); XI.append(xi)
    if len(Y) < 500:
        return None
    X, Y = np.array(X), np.array(Y)
    coef = _fit(X, Y)
    now = fit_rapm(plays, cur, prm)
    return {"stack_coef": coef.tolist(), "threshold": float(np.percentile(np.abs(XI), INJURY_QUANTILE)),
            "availability": table, "games": int(len(Y)), "values": values_to_dict(now),
            "repl": {f"{k[0]}|{k[1]}": v for k, v in replacement_levels(now).items()}}


class PlayersNow:
    """Lo que hace falta para el ajuste por bajas en la semana actual: uso esperado por snaps, reporte
    de lesiones (nflverse + ESPN) y los valores RAPM guardados al entrenar."""

    def __init__(self, P: Dict, qbs=None):
        from app.services.nfl_players import expected_usage, snaps_usage, values_from_dict
        self.P = P
        self.values = values_from_dict(P["values"])
        self.repl = {tuple(k.split("|")): v for k, v in P["repl"].items()}
        cur = current_season()
        self.exp = expected_usage(snaps_usage(cur))
        self.report: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        try:
            import nflreadpy as nfl
            inj = nfl.load_injuries([cur])
            wk = inj["week"].max()
            for r in inj.filter(pl.col("week") == wk).iter_rows(named=True):
                if r["gsis_id"]:
                    self.report[r["gsis_id"]] = (r["report_status"], r["practice_status"])
        except Exception as e:
            logger.warning(f"NFL v2: sin reporte de lesiones de nflverse ({type(e).__name__}: {e})")
        if qbs is not None:                      # ESPN: estado del día cuando nflverse aún no tiene el oficial
            for eid, info in qbs.injuries.items():
                g = qbs.e2g.get(eid)
                st = (info.get("status") or "")
                if not g or st.lower() == "active":
                    continue
                rs, pr = self.report.get(g, (None, None))
                self.report[g] = (rs or st, pr)
        from app.services.nfl_qb_status import espn_to_gsis
        self.names = espn_to_gsis()[1]

    def adjust(self, r: Dict, mu_v2: float) -> Tuple[float, Dict]:
        from app.services.nfl_players import injury_adjustment, team_strength
        s, wk = r["season"], r["week"]
        out = {"applied": False}
        usages = {}
        for side in ("home", "away"):
            t = r[f"{side}_team"]
            usages[side] = self.exp.get((s, wk, t)) or self.exp.get((s, -1, t))
        if not all(u and u["o"] for u in usages.values()):
            return mu_v2, out
        st, lost, det = {}, {}, {}
        for side, u in usages.items():
            o, d = team_strength(self.values, u)
            st[side] = o + d
            lost[side], det[side] = injury_adjustment(self.values, u, self.report, self.P["availability"], self.repl)
        xe = (st["home"] - st["away"]) * PLAYS_PER_GAME
        xi = (lost["home"] - lost["away"]) * PLAYS_PER_GAME
        c = self.P["stack_coef"]
        pts_per_epa = -c[3]                       # puntos de margen por EPA de bajas
        mu_adj = c[0] + c[1] * mu_v2 + c[2] * xe + c[3] * xi
        applied = abs(xi) >= self.P["threshold"]
        for side in ("home", "away"):
            rows = sorted(det[side], key=lambda x: -x["lost"])
            out[f"{side}_injuries"] = [{
                "name": self.names.get(x["player"], x["player"]), "group": x["group"],
                "side": "ataque" if x["side"] == "o" else "defensa", "status": self.report.get(x["player"], (None, None))[0],
                "practice": self.report.get(x["player"], (None, None))[1], "play_prob": x["play_prob"],
                "usual_share": x["share"], "impact_pts": round(x["lost"] * PLAYS_PER_GAME * pts_per_epa, 2),
            } for x in rows if abs(x["lost"] * PLAYS_PER_GAME * pts_per_epa) >= 0.05][:8]
            out[f"{side}_injury_pts"] = round(lost[side] * PLAYS_PER_GAME * pts_per_epa, 2)
        out.update({"applied": bool(applied), "difference_epa": round(xi, 2), "threshold_epa": round(self.P["threshold"], 2),
                    "margin_v2": round(mu_v2, 2), "margin_with_injuries": round(mu_adj, 2)})
        return (mu_adj if applied else mu_v2), out


def load_data(refresh: bool = False, refresh_schedules: Optional[bool] = None):
    """Calendario y jugadas. refresh: vuelve a bajar la temporada actual de jugadas;
    refresh_schedules: el calendario (QB titulares, resultados), por defecto igual que refresh."""
    rs = refresh if refresh_schedules is None else refresh_schedules
    sched = normalize_teams(load_schedules(refresh=rs), cols=("home_team", "away_team"))
    sched = sched.filter(pl.col("season") >= 2010).sort(["season", "week", "gameday"])
    pbp = load_pbp(range(2010, current_season() + 1), refresh_current=refresh)
    return sched, pbp


def train(refresh: bool = True) -> Dict:
    """Recorre el histórico semana a semana, ajusta las regresiones y la calibración, y guarda."""
    sched, pbp = load_data(refresh)
    feats = Features(sched, pbp)
    played = sched.filter(pl.col("result").is_not_null())
    rows = list(played.iter_rows(named=True))
    Xm, Xt = [], []
    for r in rows:
        m, t, _ = feats.row(r)
        Xm.append([m[k] for k in MARGIN_FEATURES])
        Xt.append([t[k] for k in TOTAL_FEATURES])
    Xm, Xt = np.array(Xm), np.array(Xt)
    season = played["season"].to_numpy()
    margin = played["result"].to_numpy().astype(float)
    total = played["total"].to_numpy().astype(float)
    spread = played["spread_line"].to_numpy().astype(float)
    tline = played["total_line"].to_numpy().astype(float)
    use = season >= FIRST_TRAIN_SEASON

    # Predicciones fuera de muestra (cada temporada con las anteriores) para sigma y calibración
    mu, mt = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    for s in np.unique(season):
        tr = use & (season < s)
        if tr.sum() < 500:
            continue
        te = season == s
        cm, ct = _fit(Xm[tr], margin[tr]), _fit(Xt[tr], total[tr])
        mu[te] = np.c_[np.ones(te.sum()), Xm[te]] @ cm
        mt[te] = np.c_[np.ones(te.sum()), Xt[te]] @ ct
    ok = ~np.isnan(mu)
    sigma = float(np.std(margin[ok] - mu[ok]))
    sigma_t = float(np.nanstd(total[ok] - mt[ok]))
    dist = MarginDistribution.fit(mu[ok], margin[ok], sigma)

    zc, yc, zo, yo, zw, yw = [], [], [], [], [], []
    for i in np.where(ok)[0]:
        pr = dist.probs(mu[i], -spread[i])
        if margin[i] - spread[i] != 0:
            zc.append(norm.ppf(np.clip(pr["home_cover"] / max(1 - pr["push"], 1e-9), 1e-4, 1 - 1e-4)))
            yc.append(1.0 if margin[i] - spread[i] > 0 else 0.0)
        if not np.isnan(tline[i]) and total[i] != tline[i]:
            tp = total_probs(mt[i], sigma_t, tline[i])
            zo.append(norm.ppf(np.clip(tp["over"] / max(1 - tp["push"], 1e-9), 1e-4, 1 - 1e-4)))
            yo.append(1.0 if total[i] > tline[i] else 0.0)
        if margin[i] != 0:
            zw.append(norm.ppf(np.clip(pr["home_win"], 1e-4, 1 - 1e-4)))
            yw.append(1.0 if margin[i] > 0 else 0.0)
    # Calibración con todas las temporadas fuera de muestra, pesando más las recientes (vida media de
    # CALIB_HALF_LIFE temporadas): el valor contra la línea viene bajando año con año.
    last = int(season[ok].max())
    wts = lambda arr_s: 0.5 ** ((last - np.array(arr_s)) / CALIB_HALF_LIFE)
    seasons_c = [season[i] for i in np.where(ok)[0] if margin[i] - spread[i] != 0]
    seasons_o = [season[i] for i in np.where(ok)[0] if not np.isnan(tline[i]) and total[i] != tline[i]]
    seasons_w = [season[i] for i in np.where(ok)[0] if margin[i] != 0]
    k_cover = _calib_k(np.array(zc), np.array(yc), wts(seasons_c))
    k_over = _calib_k(np.array(zo), np.array(yo), wts(seasons_o))
    k_win = _calib_k(np.array(zw), np.array(yw), wts(seasons_w))

    cm, ct = _fit(Xm[use], margin[use]), _fit(Xt[use], total[use])
    try:
        players = _train_players(pbp, played, mu, margin)
    except Exception as e:
        logger.warning(f"NFL v2: sin ajuste por bajas de jugadores ({type(e).__name__}: {e})")
        players = None
    try:
        from app.services.nfl_qb_status import learn_play_probs
        qb_play = learn_play_probs()
    except Exception as e:
        logger.warning(f"NFL v2: no se pudo aprender la probabilidad de que juegue el QB ({type(e).__name__}: {e})")
        qb_play = (load_params() or {}).get("qb_play")
    params = {
        "trained_at": datetime.utcnow().isoformat(timespec="seconds"),
        "last_week": [int(played["season"].max()), int(played.filter(pl.col("season") == played["season"].max())["week"].max())],
        "margin_coef": cm.tolist(), "total_coef": ct.tolist(),
        "margin_features": MARGIN_FEATURES, "total_features": TOTAL_FEATURES,
        "sigma_margin": sigma, "sigma_total": sigma_t,
        "key_weights": {str(k): v for k, v in dist.weights.items()},
        "k_cover": k_cover, "k_over": k_over, "k_win": k_win, "qb_play": qb_play, "players": players,
        "metrics": {"games": int(ok.sum()), "mae_margin": float(np.mean(np.abs(mu[ok] - margin[ok]))),
                    "mae_total": float(np.nanmean(np.abs(mt[ok] - total[ok]))),
                    "mae_market_margin": float(np.mean(np.abs(spread[ok] - margin[ok]))),
                    "winner_pct": float(np.mean((mu[ok] > 0) == (margin[ok] > 0)))},
    }
    os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)
    with open(PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=1)
    logger.info(f"🏈 NFL v2 entrenado: MAE margen {params['metrics']['mae_margin']:.2f} · k spread {k_cover:.2f} · "
                f"k total {k_over:.2f} · k ganador {k_win:.2f}")
    return params


def load_params() -> Optional[Dict]:
    try:
        with open(PARAMS_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def params_are_stale(p: Optional[Dict]) -> bool:
    if not p:
        return True
    try:
        return datetime.fromisoformat(p["trained_at"]) < datetime.utcnow() - timedelta(hours=20)
    except (KeyError, ValueError):
        return True


# ══════════════════════════════════════════════════════════════════════
# Probabilidades de mercado (las usa también la API para "lo que recomienda el modelo")
# ══════════════════════════════════════════════════════════════════════

def _dist(p: Dict) -> MarginDistribution:
    return MarginDistribution(sigma=p["sigma_margin"], weights={int(k): v for k, v in p["key_weights"].items()})


def _cal(prob: float, k: float) -> float:
    return float(norm.cdf(k * norm.ppf(min(max(prob, 1e-4), 1 - 1e-4))))


def market_probs(mu: float, total: Optional[float], market_margin: Optional[float], market_total: Optional[float],
                 p: Optional[Dict] = None) -> Dict[str, Optional[float]]:
    """Probabilidades calibradas: que el local cubra (sin contar push) y que salga over (sin push)."""
    p = p or load_params()
    out = {"home_cover": None, "over": None}
    if not p:
        return out
    if market_margin is not None:
        pr = _dist(p).probs(mu, -market_margin)
        out["home_cover"] = _cal(pr["home_cover"] / max(1 - pr["push"], 1e-9), p["k_cover"])
    if total is not None and market_total is not None:
        tp = total_probs(total, p["sigma_total"], market_total)
        out["over"] = _cal(tp["over"] / max(1 - tp["push"], 1e-9), p["k_over"])
    return out


# ══════════════════════════════════════════════════════════════════════
# Predicción de los partidos de la base
# ══════════════════════════════════════════════════════════════════════

def _match(sched: pl.DataFrame, home: str, away: str, day: date) -> Optional[Dict]:
    h, a = abbr(home), abbr(away)
    cand = sched.filter((pl.col("home_team") == h) & (pl.col("away_team") == a))
    best = None
    for r in cand.iter_rows(named=True):
        try:
            gd = date.fromisoformat(r["gameday"])
        except (TypeError, ValueError):
            continue
        dd = abs((gd - day).days)
        if dd <= 1 and (best is None or dd < best[0]):
            best = (dd, r)
    return None if best is None else best[1]


def espn_forecast(event_id: Optional[str]) -> Optional[Dict]:
    """Pronóstico del clima de ESPN para un partido por jugar (AccuWeather): probabilidad de lluvia,
    ráfagas y temperatura. None si ESPN no lo tiene."""
    if not event_id:
        return None
    from app.services.nfl_qb_status import _get
    d = _get("https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary", event=str(event_id)) or {}
    w = (d.get("gameInfo") or {}).get("weather")
    if not w:
        return None
    cond = int(w.get("conditionId") or 0)
    snowy = 19 <= cond <= 29 and cond != 26           # códigos de AccuWeather: nieve, aguanieve, hielo
    precip = float(w.get("precipitation") or 0) / 100.0
    return {"rain": 0.0 if snowy else precip, "snow": precip if snowy else 0.0,
            "temp": w.get("temperature"), "wind": None if w.get("gust") is None else round(0.7 * float(w["gust"]), 1),
            "precipitation_pct": w.get("precipitation"), "condition_id": cond}


def _row_with_lineups(feats: Features, r: Dict, qbs, home: str, away: str):
    """Variables del partido con el QB de cada equipo según ESPN: si el titular está en duda, las
    del QB se promedian entre él y su suplente con la probabilidad de que juegue."""
    if qbs is None:
        return feats.row(r)
    try:
        lh = qbs.lineup(home, r.get("home_qb_id"), r.get("home_qb_name"))
        la = qbs.lineup(away, r.get("away_qb_id"), r.get("away_qb_name"))
    except Exception as e:
        logger.warning(f"NFL v2: no se pudo revisar el QB de {away} @ {home} ({type(e).__name__}: {e})")
        return feats.row(r)
    m_avg: Dict[str, float] = defaultdict(float)
    t_avg: Dict[str, float] = defaultdict(float)
    info = None
    for hq, ph in lh["mix"]:
        for aq, pa in la["mix"]:
            r2 = {**r, "home_qb_id": hq, "away_qb_id": aq, "home_qb_name": None, "away_qb_name": None}
            m, t, inf = feats.row(r2)
            for k, v in m.items():
                m_avg[k] += ph * pa * v
            for k, v in t.items():
                t_avg[k] += ph * pa * v
            if info is None:
                info = inf
    t = dict(t_avg)
    for side, l in (("home", lh), ("away", la)):
        info[f"{side}_qb"].update({
            "name": l["starter_name"], "status": l["starter_status"], "play_prob": l["starter_prob"],
            "backup": l["backup_name"], "projected": l["projected_name"], "changed": l["changed"],
            "practice": l.get("starter_practice"),
            "ruled_out": l["skipped"],
        })
    return dict(m_avg), t, info


def predict_games_sync(games: List[Tuple], refresh: bool = False) -> Dict[int, Dict]:
    """[(id, local, visita, fecha ET[, id de ESPN])] → {id: resultado}. Entrena si los parámetros tienen más de 20 h."""
    p = load_params()
    if params_are_stale(p):
        p = train(refresh=True)
        refresh = False
    try:
        sched, pbp = load_data(refresh, refresh_schedules=True)       # titulares actualizados
    except Exception as e:
        logger.warning(f"NFL v2: no se pudo actualizar el calendario de nflverse ({type(e).__name__}: {e})")
        sched, pbp = load_data(False, refresh_schedules=False)
    feats = Features(sched, pbp)
    dist = _dist(p)
    try:
        from app.services.nfl_qb_status import QBStatus
        qbs = QBStatus(play_table=(p.get("qb_play") or {}).get("table"))
    except Exception as e:
        logger.warning(f"NFL v2: sin lesiones de ESPN, se usa el titular de nflverse ({type(e).__name__}: {e})")
        qbs = None
    pnow = None
    if p.get("players"):
        try:
            pnow = PlayersNow(p["players"], qbs)
        except Exception as e:
            logger.warning(f"NFL v2: sin bajas por jugador ({type(e).__name__}: {e})")
    out = {}
    for g in games:
        gid, home, away, day = g[:4]
        event_id = g[4] if len(g) > 4 else None
        r = _match(sched, home, away, day)
        if r is None:
            logger.warning(f"NFL v2: {away} @ {home} ({day}) no está en el calendario de nflverse")
            continue
        if r.get("result") is None and r.get("roof") not in ("dome", "closed"):
            try:
                fc = espn_forecast(event_id)
            except Exception as e:
                logger.warning(f"NFL v2: sin pronóstico del clima para {away} @ {home} ({type(e).__name__}: {e})")
                fc = None
            if fc:
                r = {**r, "rain": fc["rain"], "snow": fc["snow"], "weather_source": "pronóstico ESPN",
                     "precipitation_pct": fc["precipitation_pct"],
                     "temp": r.get("temp") if r.get("temp") is not None else fc["temp"],
                     "wind": r.get("wind") if r.get("wind") is not None else fc["wind"]}
        m, t, info = _row_with_lineups(feats, r, qbs, home, away)
        info["precipitation_pct"] = r.get("precipitation_pct")
        mu = float(p["margin_coef"][0] + sum(c * m[k] for c, k in zip(p["margin_coef"][1:], MARGIN_FEATURES)))
        if pnow is not None and r.get("result") is None:
            try:
                mu, inj_info = pnow.adjust(r, mu)
                info["player_injuries"] = inj_info
            except Exception as e:
                logger.warning(f"NFL v2: ajuste por bajas falló en {away} @ {home} ({type(e).__name__}: {e})")
        tot = float(p["total_coef"][0] + sum(c * t[k] for c, k in zip(p["total_coef"][1:], TOTAL_FEATURES)))
        pr = dist.probs(mu)
        hw = _cal(pr["home_win"], p["k_win"])
        out[gid] = {
            "home_win_prob": round(hw, 4), "away_win_prob": round(1 - hw, 4), "draw_prob": None,
            "predicted_home_score": round((tot + mu) / 2, 1), "predicted_away_score": round((tot - mu) / 2, 1),
            "predicted_total": round(tot, 1), "predicted_spread": round(-mu, 1),
            "confidence": round(abs(hw - 0.5) * 2, 3),
            "simulation_details": {
                "model": MODEL_NAME, "nflverse_game_id": r["game_id"], "week": r["week"],
                "margin_features": {k: round(v, 2) for k, v in m.items()},
                "total_features": {k: round(v, 2) for k, v in t.items()},
                **info,
                "sigma_margin": round(p["sigma_margin"], 2), "sigma_total": round(p["sigma_total"], 2),
                "calibration": {"spread": round(p["k_cover"], 2), "total": round(p["k_over"], 2),
                                "winner": round(p["k_win"], 2)},
                "trained_at": p["trained_at"],
                "note": "Backtest 2017–2026 sin ventaja contra la línea de cierre: el valor se calcula con "
                        "probabilidades calibradas y no se recomienda moneyline.",
            },
        }
    return out


def nfl_picks(home_win_prob, pred_margin, pred_total, market_margin, market_total, ml_home=None, ml_away=None,
              sp_home_odds=None, sp_away_odds=None, over_odds=None, under_odds=None, home_pts=None, away_pts=None,
              home_name: str = "", away_name: str = "") -> Dict:
    """Lo que prefiere el modelo en ganador, spread y total (mismo formato que la NBA), con las
    probabilidades calibradas de la NFL. El ganador nunca se recomienda (sin ventaja en el backtest)."""
    from app.services.nba_tracking import market_picks
    pr = market_probs(pred_margin, pred_total, market_margin, market_total)
    d_sp = None if pred_margin is None or market_margin is None else pred_margin - market_margin
    d_tt = None if pred_total is None or market_total is None else pred_total - market_total
    # El lado lo deciden las métricas (la diferencia con la línea), nunca un empate en 50%
    tie = lambda p, d: p if p is None or d is None or abs(p - 0.5) > 1e-6 else 0.5 + (1e-6 if d > 0 else -1e-6 if d < 0 else 0)
    # Probabilidad propia del modelo (sin calibrar contra la línea), para el pick y su confianza
    raw_c = raw_o = None
    p = load_params()
    if p and d_sp is not None:
        r_ = _dist(p).probs(pred_margin, -market_margin)
        raw_c = r_["home_cover"] / max(1 - r_["push"], 1e-9)
    if p and d_tt is not None:
        t_ = total_probs(pred_total, p["sigma_total"], market_total)
        raw_o = t_["over"] / max(1 - t_["push"], 1e-9)
    out = market_picks(home_win_prob, pred_margin, pred_total, market_margin, market_total, ml_home, ml_away,
                       sp_home_odds, sp_away_odds, over_odds, under_odds, home_pts, away_pts,
                       abbr(home_name), abbr(away_name), p_home_cover=tie(pr["home_cover"], d_sp),
                       p_over=tie(pr["over"], d_tt), winner_recommendable=False,
                       p_home_cover_raw=raw_c, p_over_raw=raw_o)
    # Cuántos puntos ve el modelo a favor del lado elegido, contra la línea
    if "spread" in out and d_sp is not None:
        out["spread"]["diff_pts"] = round(d_sp if out["spread"]["side"] == "home" else -d_sp, 1)
    if "total" in out and d_tt is not None:
        out["total"]["diff_pts"] = round(d_tt if out["total"]["side"] == "over" else -d_tt, 1)
    return out
