"""
OmniPicks — NFL: ¿qué métricas nuevas agregan información? (margen, total y valor contra la línea)

Candidatas, todas con lo jugado antes de cada semana:
  · epant : EPA por jugada SIN pérdidas de balón, ajustado por rival (los balones sueltos son casi azar)
  · succ  : tasa de éxito por jugada, ajustada por rival (más estable que el EPA)
  · st    : equipos especiales (EPA neto de patadas, despejes, goles de campo), con descuento
  · QB en el total: calidad de los dos QB titulares y su cambio contra el habitual

Para cada combinación: error del margen y del total en 2015–2020 y 2021–2026 (prueba limpia), y el
factor de calibración k contra la línea de cierre (k = 0: la diferencia con la línea no predice nada).

Uso:
    python ml/scripts/nfl_features_eval.py
"""
import os
import sys
from collections import defaultdict

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from scipy.stats import norm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services import nfl_v2
from app.services.nfl_history import current_season, load_special_teams, team_games
from app.services.nfl_ratings import NFLRatingEngine, expected_margin_epa, time_index
from app.services.nfl_score import MarginDistribution, total_probs


def st_snapshots(st: pl.DataFrame, sched: pl.DataFrame, half_life: float = 16.0, gap: float = 12.0, prior: float = 4.0):
    """Equipos especiales por equipo antes de cada semana: media con descuento del EPA neto por partido,
    encogida hacia 0 con `prior` partidos virtuales."""
    st = st.sort(["season", "week"])
    t = time_index(st["season"].to_numpy().astype(float), st["week"].to_numpy().astype(float), gap)
    team, val = st["team"].to_list(), st["st_net"].to_numpy()
    out = {}
    keys = sorted(set(zip(sched["season"].to_list(), sched["week"].to_list())))
    for s, w in keys:
        now = time_index(np.array([s]), np.array([w]), gap)[0]
        age = now - t
        m = (age > 0) & (age <= 4 * half_life)
        d = 0.5 ** (age[m] / half_life)
        num, den = defaultdict(float), defaultdict(float)
        for tm, v, dd in zip(np.array(team, dtype=object)[m], val[m], d):
            num[tm] += dd * v
            den[tm] += dd
        out[(s, w)] = {tm: num[tm] / (den[tm] + prior) for tm in den}
    return out


def walk(X, y, season):
    pred = np.full(len(y), np.nan)
    for s in np.unique(season):
        tr = (season >= 2011) & (season < s)
        if tr.sum() < 500:
            continue
        c = nfl_v2._fit(X[tr], y[tr])
        te = season == s
        pred[te] = np.c_[np.ones(te.sum()), X[te]] @ c
    return pred


def calib_k(pred, y, line, season, sigma, kind, dist=None):
    """k (con temporadas 2021–2026) y Brier de la probabilidad calibrada contra la línea de cierre."""
    z, out, ss = [], [], []
    for i in np.where(~np.isnan(pred) & ~np.isnan(line))[0]:
        if kind == "spread":
            real = y[i] - line[i]
            if real == 0:
                continue
            pr = dist.probs(pred[i], -line[i])
            p = pr["home_cover"] / max(1 - pr["push"], 1e-9)
        else:
            real = y[i] - line[i]
            if real == 0:
                continue
            tp = total_probs(pred[i], sigma, line[i])
            p = tp["over"] / max(1 - tp["push"], 1e-9)
        z.append(norm.ppf(np.clip(p, 1e-4, 1 - 1e-4))); out.append(1.0 if real > 0 else 0.0); ss.append(season[i])
    z, out, ss = np.array(z), np.array(out), np.array(ss)
    h = ss >= 2021
    nll = lambda k: -np.mean(out[h] * np.log(norm.cdf(k * z[h]) + 1e-12) + (1 - out[h]) * np.log(1 - norm.cdf(k * z[h]) + 1e-12))
    k = minimize_scalar(nll, bounds=(0.0, 1.5), method="bounded").x
    # k aprendido con 2015–2020 y aplicado a 2021–2026 (prueba limpia de verdad)
    t = (ss >= 2015) & (ss <= 2020)
    nll_t = lambda k: -np.mean(out[t] * np.log(norm.cdf(k * z[t]) + 1e-12) + (1 - out[t]) * np.log(1 - norm.cdf(k * z[t]) + 1e-12))
    k_t = minimize_scalar(nll_t, bounds=(0.0, 1.5), method="bounded").x
    brier = float(np.mean((norm.cdf(k_t * z[h]) - out[h]) ** 2))
    return k_t, k, brier


def main():
    sched, pbp = nfl_v2.load_data(False, refresh_schedules=False)
    feats = nfl_v2.Features(sched, pbp)
    played = sched.filter(pl.col("result").is_not_null())
    rows = list(played.iter_rows(named=True))
    neutral = sched.filter(pl.col("location") == "Neutral")["game_id"].to_list()
    tg = team_games(pbp)
    eng_nt = NFLRatingEngine(tg, nfl_v2.EPA_P, neutral_games=neutral, value="epant")
    eng_sc = NFLRatingEngine(tg, nfl_v2.EPA_P, neutral_games=neutral, value="succ")
    st = st_snapshots(load_special_teams(range(2010, current_season() + 1)), sched)
    cache = {}

    M, T, E = [], [], defaultdict(list)
    for r in rows:
        m, t, info = feats.row(r)
        M.append([m[k] for k in nfl_v2.MARGIN_FEATURES])
        T.append([t[k] for k in nfl_v2.TOTAL_FEATURES])
        key = (r["season"], r["week"])
        if key not in cache:
            cache[key] = (eng_nt.at(*key), eng_sc.at(*key))
        wnt, wsc = cache[key]
        nt_m, nt_t = expected_margin_epa(wnt, r["home_team"], r["away_team"], r["location"] == "Neutral")
        sc_m, sc_t = expected_margin_epa(wsc, r["home_team"], r["away_team"], r["location"] == "Neutral")
        sk = st.get(key, {})
        E["epant"].append(nt_m); E["epant_t"].append(nt_t)
        E["succ"].append(sc_m); E["succ_t"].append(sc_t)
        E["st"].append(sk.get(r["home_team"], 0.0) - sk.get(r["away_team"], 0.0))
        E["st_t"].append(sk.get(r["home_team"], 0.0) + sk.get(r["away_team"], 0.0))
        hq, aq = info["home_qb"], info["away_qb"]
        E["qb_sum"].append((hq["rating"] + aq["rating"]) * 38.0)
        E["qb_adj_sum"].append(hq["adjustment_pts_epa"] + aq["adjustment_pts_epa"])
    M, T = np.array(M), np.array(T)
    E = {k: np.array(v) for k, v in E.items()}
    season = played["season"].to_numpy()
    margin = played["result"].to_numpy().astype(float)
    total = played["total"].to_numpy().astype(float)
    spread = played["spread_line"].to_numpy().astype(float)
    tline = played["total_line"].to_numpy().astype(float)
    tune = (season >= 2015) & (season <= 2020)
    hold = season >= 2021

    def report(name, pred, y, line, kind):
        ok = ~np.isnan(pred)
        sig = float(np.std((y - pred)[ok]))
        dist = MarginDistribution.fit(pred[ok], y[ok], sig) if kind == "spread" else None
        k_t, k_h, brier = calib_k(pred, y, line, season, sig, kind, dist)
        mae = lambda m: float(np.mean(np.abs(pred[m & ok] - y[m & ok])))
        print(f"  {name:44} MAE 15–20 {mae(tune):6.3f} · 21–26 {mae(hold):6.3f} · k(15–20) {k_t:.2f} · "
              f"k(21–26) {k_h:.2f} · Brier 21–26 {brier:.4f}")

    print("MARGEN (contra el spread de cierre):")
    report("actual (EPA, marcador, QB)", walk(M, margin, season), margin, spread, "spread")
    for extra in (["epant"], ["succ"], ["st"], ["epant", "succ", "st"]):
        X = np.c_[M, *[E[e] for e in extra]]
        report("+ " + ", ".join(extra), walk(X, margin, season), margin, spread, "spread")
    X = np.c_[M[:, 1:], E["epant"], E["succ"], E["st"]]
    report("sin EPA normal, + epant, succ, st", walk(X, margin, season), margin, spread, "spread")

    print("\nTOTAL (contra el total de cierre):")
    report("actual (anotación, EPA, ritmo, clima)", walk(T, total, season), total, tline, "total")
    for extra in (["qb_sum", "qb_adj_sum"], ["qb_sum", "qb_adj_sum", "epant_t"], ["qb_sum", "qb_adj_sum", "epant_t", "st_t"], ["epant_t"], ["succ_t"], ["st_t"],
                  ["qb_sum", "qb_adj_sum", "epant_t", "succ_t", "st_t"]):
        X = np.c_[T, *[E[e] for e in extra]]
        report("+ " + ", ".join(extra), walk(X, total, season), total, tline, "total")
    print("\nReferencia (mercado): MAE margen 21–26 "
          f"{np.mean(np.abs(spread[hold] - margin[hold])):.3f} · total {np.nanmean(np.abs(tline[hold] - total[hold])):.3f}")


if __name__ == "__main__":
    main()
