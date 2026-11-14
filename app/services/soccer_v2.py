"""
OmniPicks — Fútbol v2 en producción (fases 1–4 de docs/PLAN_FUTBOL.md).

  · Ligas (Premier, LaLiga, Serie A, Bundesliga, Ligue 1): Dixon-Coles con ratings de 70% goles +
    30% goles por tiros a puerta; Liga MX: Dixon-Coles con goles.
  · Champions y Europa League: modelo conjunto de ligas y copas con fuerza por liga.
  · Probabilidades 1X2 calibradas con temperatura (T = 0.90: el modelo crudo es algo conservador).

El backtest con momios de cierre no encontró ventaja (fase 4): el modelo es para pronosticar; no
se generan picks de 1X2 por diferencias con el casino.

Los ajustes se rehacen como mucho cada 6 horas (≈ 15 s); el histórico de la temporada en curso se
vuelve a bajar cada 12 horas.
"""
import logging
import threading
import time
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from app.services.soccer_cups import load_cups, match_names
from app.services.soccer_dc import GOALS_PER_SOT, DCParams, fit_dc, fit_rates, markets, score_matrix
from app.services.soccer_history import current_season_start, load_all
from app.services.soccer_joint import EU_LEAGUES, JointModel, JointParams, build_table

logger = logging.getLogger(__name__)

MODEL_NAME = "soccer_v2"
TEMPERATURE = 0.90          # ligas: el modelo crudo es algo conservador (backtest fase 4)
TEMPERATURE_CUPS = 0.95     # copas: casi calibrado; en favoritos ≥80% exagera un poco (85.6% previsto, 83.4% real)
SHOTS_LEAGUES = {"Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"}
LEAGUES = ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1", "Liga MX"]
CUPS = {"UEFA Champions League", "UEFA Europa League"}
NATIONAL = {"UEFA Nations League", "CONCACAF Nations League", "World Cup", "Euro Championship", "Copa America"}
REFIT_SECONDS = 6 * 3600

# Backtest (docs/PLAN_FUTBOL.md, fases 1–4): 6 ligas, 2022–2026, semana a semana
BACKTEST = {"games": 9083, "seasons": "2022–2026", "logloss": 0.9868, "logloss_pinnacle": 0.9667,
            "logloss_base": 1.0714, "mae_total": 1.32, "cups_logloss": 0.9457, "cups_games": 2718}

_lock = threading.Lock()
_state: Dict = {"at": 0.0}


def _fit_all() -> Dict:
    data = load_all(2013, refresh_current=True)
    today = np.datetime64(date.today().isoformat())
    fits, fd_names = {}, {}
    for lg in LEAGUES:
        df = data.filter((pl.col("league") == lg) & pl.col("FTHG").is_not_null() & pl.col("date").is_not_null()).sort("date")
        d = df["date"].to_numpy().astype("datetime64[D]")
        days = (today - d).astype(int).astype(float)
        H, A = df["HomeTeam"].to_list(), df["AwayTeam"].to_list()
        hg, ag = df["FTHG"].to_numpy(), df["FTAG"].to_numpy()
        f = fit_dc(H, A, hg, ag, days, DCParams())
        if lg in SHOTS_LEAGUES and "HST" in df.columns:
            has = (df["HST"].is_not_null() & df["AST"].is_not_null()).to_numpy()
            hst, ast = df["HST"].fill_null(0).to_numpy(), df["AST"].fill_null(0).to_numpy()
            yh = np.where(has, 0.7 * hg + 0.3 * GOALS_PER_SOT * hst, hg)
            ya = np.where(has, 0.7 * ag + 0.3 * GOALS_PER_SOT * ast, ag)
            f = fit_rates(H, A, yh, ya, days, DCParams(), rho=f.rho)
        fits[lg] = f
        cur = df.filter(pl.col("season") >= current_season_start() - 1)
        fd_names[lg] = set(cur["HomeTeam"].to_list()) | set(cur["AwayTeam"].to_list())
    joint = None
    try:
        cups = load_cups(2016)
        eu_fd = set(data.filter(pl.col("league").is_in(EU_LEAGUES))["HomeTeam"].to_list())
        cup_names = set(cups["home"].drop_nulls().to_list()) | set(cups["away"].drop_nulls().to_list())
        nm = match_names(cup_names, eu_fd)
        jm = JointModel(build_table(data, cups, nm), JointParams())
        joint = jm.fit(today)
        fd_names["_cups"] = eu_fd
    except Exception as e:
        logger.warning(f"Fútbol v2: sin modelo de copas ({type(e).__name__}: {e})")
    national = None
    try:
        from app.services.soccer_national import fit_national, load_results
        national = fit_national(load_results(), today)
    except Exception as e:
        logger.warning(f"Fútbol v2: sin modelo de selecciones ({type(e).__name__}: {e})")
    return {"fits": fits, "joint": joint, "national": national, "fd_names": fd_names, "at": time.time()}


def _state_now() -> Dict:
    with _lock:
        if time.time() - _state.get("at", 0) > REFIT_SECONDS:
            _state.update(_fit_all())
            logger.info("⚽ Fútbol v2: modelos ajustados")
        return _state


def _calibrate(p: np.ndarray, temperature: float = TEMPERATURE) -> np.ndarray:
    q = np.clip(p, 1e-9, None) ** (1 / temperature)
    return q / q.sum()


def predict_games_sync(games: List[Tuple[int, str, str, str, bool]]) -> Dict[int, Dict]:
    """[(id, local, visita, competición, sede neutral)] → {id: resultado}."""
    st = _state_now()
    out = {}
    for gid, home, away, comp, neutral in games:
        try:
            if comp in NATIONAL:
                fit = st.get("national")
                if fit is None:
                    continue
                from app.services.soccer_national import team_name
                known = set(fit.teams)
                h, a = team_name(home, known), team_name(away, known)
                if h is None or a is None:
                    logger.warning(f"Fútbol v2: selección sin historial: {home if h is None else away}")
                h, a = h or home, a or away
                m = fit.score_matrix(h, a, neutral)
                ti = {t: k for k, t in enumerate(fit.teams)}
                ratings = {side: {"attack": round(float(fit.attack[ti[t]]), 2), "defense": round(float(fit.defense[ti[t]]), 2)}
                           for side, t in (("home", h), ("away", a)) if t in ti}
            elif comp in CUPS:
                if st.get("joint") is None:
                    continue
                nm = match_names([home, away], st["fd_names"]["_cups"])
                h = nm.get(home, home + " ·")
                a = nm.get(away, away + " ·")
                fit = st["joint"]
                m = fit.score_matrix(comp, h, a, neutral)
                lh, la = fit.rates(comp, h, a, neutral)
                ratings = {"home_league": fit.team_league.get(h, "Otras ligas"), "away_league": fit.team_league.get(a, "Otras ligas")}
                li = {l: k for k, l in enumerate(fit.leagues)}
                ratings["home_league_strength"] = round(float(fit.A[li[ratings["home_league"]]] + fit.D[li[ratings["home_league"]]]), 2)
                ratings["away_league_strength"] = round(float(fit.A[li[ratings["away_league"]]] + fit.D[li[ratings["away_league"]]]), 2)
            elif comp in st["fits"]:
                fit = st["fits"][comp]
                nm = match_names([home, away], st["fd_names"][comp])
                h, a = nm.get(home), nm.get(away)
                if h is None or a is None:
                    logger.warning(f"Fútbol v2: sin pareja de nombres para {home} vs {away} ({comp})")
                h, a = h or home, a or away
                m = fit.score_matrix(h, a, neutral)
                lh, la = fit.rates(h, a, neutral)
                ti = {t: k for k, t in enumerate(fit.teams)}
                ratings = {side: {"attack": round(float(fit.attack[ti[t]]), 2), "defense": round(float(fit.defense[ti[t]]), 2)}
                           for side, t in (("home", h), ("away", a)) if t in ti}
            else:
                continue
            mk = markets(m)
            temp = TEMPERATURE_CUPS if comp in CUPS else 1.0 if comp in NATIONAL else TEMPERATURE
            p = _calibrate(np.array([mk["home"], mk["draw"], mk["away"]]), temp)
            i, j = np.unravel_index(int(np.argmax(m)), m.shape)
            exp_h, exp_a = mk["exp_home"], mk["exp_away"]
            out[gid] = {
                "home_win_prob": round(float(p[0]), 4), "draw_prob": round(float(p[1]), 4), "away_win_prob": round(float(p[2]), 4),
                "predicted_home_score": round(exp_h, 2), "predicted_away_score": round(exp_a, 2),
                "predicted_total": round(exp_h + exp_a, 2), "predicted_spread": round(exp_a - exp_h, 2),
                "confidence": round(float(np.max(p)), 3),
                "simulation_details": {
                    "model": MODEL_NAME, "competition": comp, "matched": {"home": h, "away": a},
                    "expected_goals": {"home": round(exp_h, 2), "away": round(exp_a, 2)},
                    "most_likely_score": [int(i), int(j)], "most_likely_prob": round(float(m[i, j]), 3),
                    "over_2_5": round(mk["over_2.5"], 3), "btts": round(mk["btts"], 3),
                    "ratings": ratings, "temperature": temp,
                    "note": "Backtest 2019–2026 sin ventaja contra los momios de cierre: pronóstico, no recomendación de 1X2.",
                },
            }
        except Exception as e:
            logger.warning(f"Fútbol v2: {home} vs {away} sin predicción ({type(e).__name__}: {e})")
    return out


def soccer_picks(pred, odds_summary: Optional[Dict], home_name: str, away_name: str,
                 home_goals: Optional[float] = None, away_goals: Optional[float] = None) -> Optional[Dict]:
    """Picks del modelo en fútbol: resultado (1X2), más/menos 2.5 y ambos anotan, con la probabilidad del
    modelo y su confianza. El valor contra el casino se muestra como dato, pero no se marca como
    recomendado (el backtest no encontró ventaja)."""
    from app.services.nba_tracking import _implied, confidence_tier
    sd = getattr(pred, "simulation_details", None) or {}
    if sd.get("model") != MODEL_NAME:
        return None
    o = odds_summary or {}
    done = home_goals is not None and away_goals is not None
    ab = lambda n: "".join(w[0] for w in (n or "?").split()[:3]).upper() if len((n or "").split()) > 1 else (n or "?")[:3].upper()

    def item(label, side, prob, odds, won):
        imp = _implied(odds) if odds is not None else None
        return {"side": side, "label": label, "prob": round(prob, 3), "model_prob": round(prob, 3),
                "confidence": confidence_tier(prob), "odds": odds, "implied": None if imp is None else round(imp, 3),
                "edge": None if imp is None else round(prob - imp, 3), "tier": "sin_valor" if imp is not None else "sin_dato",
                "result": None if not done else ("win" if won else "loss")}

    probs = {"home": pred.home_win_prob or 0.0, "draw": pred.draw_prob or 0.0, "away": pred.away_win_prob or 0.0}
    k = max(probs, key=probs.get)
    lab = {"home": ab(home_name), "draw": "Empate", "away": ab(away_name)}[k]
    odds = {"home": (o.get("best_home_ml") or {}).get("odds"), "draw": (o.get("best_draw") or {}).get("odds"),
            "away": (o.get("best_away_ml") or {}).get("odds")}[k]
    res = None if not done else ("home" if home_goals > away_goals else "draw" if home_goals == away_goals else "away")
    out = {"winner": item(lab, k, probs[k], odds, res == k)}
    p_over = sd.get("over_2_5")
    if p_over is not None:
        tt = o.get("total") or {}
        line_ok = tt.get("line") == 2.5
        over = p_over >= 0.5
        out["total"] = item("Más de 2.5" if over else "Menos de 2.5", "over" if over else "under",
                            p_over if over else 1 - p_over,
                            (tt.get("over_odds") if over else tt.get("under_odds")) if line_ok else None,
                            done and ((home_goals + away_goals > 2.5) == over))
    p_b = sd.get("btts")
    if p_b is not None:
        yes = p_b >= 0.5
        out["btts"] = item("Sí" if yes else "No", "yes" if yes else "no", p_b if yes else 1 - p_b, None,
                           done and ((home_goals > 0 and away_goals > 0) == yes))
    return out
