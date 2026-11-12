"""
OmniPicks — NFL v3, fase 4: ¿simular el marcador serie por serie mejora el pronóstico?

Dos usos de la simulación, ambos con lo que se sabía antes de cada semana:
  1. Como modelo propio: total y margen esperados de la simulación (y como variables extra de la v2).
  2. Como FORMA de la distribución: se escalan las probabilidades por serie para que los puntos
     esperados de cada equipo sean los de la v2, y se compara la probabilidad del margen exacto
     (log-loss) y la de cubrir la línea contra la distribución actual (normal + números clave).

Uso:
    python ml/scripts/nfl_drives_eval.py
"""
import os
import sys

import numpy as np
import polars as pl
from scipy.stats import norm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services import nfl_v2
from app.services.nfl_drives import (MAX_PTS, DriveRates, game_distributions, load_drives, scale_to_points,
                                     team_points_pmf)
from app.services.nfl_history import current_season
from app.services.nfl_score import SUPPORT, MarginDistribution


def walk(X, y, season, first=2011):
    pred = np.full(len(y), np.nan)
    for s in np.unique(season):
        tr = (season >= first) & (season < s) & ~np.isnan(X).any(axis=1) if X.ndim > 1 else (season >= first) & (season < s)
        if tr.sum() < 500:
            continue
        c = nfl_v2._fit(X[tr], y[tr])
        te = season == s
        pred[te] = np.c_[np.ones(te.sum()), X[te]] @ c
    return pred


def main():
    sched, pbp = nfl_v2.load_data(False, refresh_schedules=False)
    feats = nfl_v2.Features(sched, pbp)
    played = sched.filter(pl.col("result").is_not_null())
    rows = list(played.iter_rows(named=True))
    M = np.array([[feats.row(r)[0][k] for k in nfl_v2.MARGIN_FEATURES] for r in rows])
    T = np.array([[feats.row(r)[1][k] for k in nfl_v2.TOTAL_FEATURES] for r in rows])
    y = played["result"].to_numpy().astype(float)
    tot = played["total"].to_numpy().astype(float)
    season = played["season"].to_numpy()
    spread = played["spread_line"].to_numpy().astype(float)
    tline = played["total_line"].to_numpy().astype(float)

    dr = DriveRates(load_drives(range(2010, current_season() + 1)))
    cache, sim = {}, []
    for r in rows:
        key = (r["season"], r["week"])
        if key not in cache:
            cache[key] = dr.at(*key)
        g = dr.game(cache[key], r["home_team"], r["away_team"], r["location"] == "Neutral")
        hp = g["drives"] * (6.97 * g["home"][0] + 3 * g["home"][1]) + g["extra"]
        ap = g["drives"] * (6.97 * g["away"][0] + 3 * g["away"][1]) + g["extra"]
        sim.append((hp, ap, g))
    sim_h = np.array([s[0] for s in sim]); sim_a = np.array([s[1] for s in sim])

    v2m = walk(M, y, season)
    v2t = walk(T, tot, season)
    simm = walk((sim_h - sim_a)[:, None], y, season)
    simt = walk((sim_h + sim_a)[:, None], tot, season)
    v2m_plus = walk(np.c_[M, sim_h - sim_a], y, season)
    v2t_plus = walk(np.c_[T, sim_h + sim_a], tot, season)
    tune, hold = (season >= 2015) & (season <= 2020), season >= 2021
    mae = lambda p, yy, m: float(np.nanmean(np.abs(p[m] - yy[m])))
    print(f"{'':40} {'2015–20':>8} {'2021–26':>8}")
    for name, p, yy in (("Margen v2", v2m, y), ("Margen solo simulación", simm, y), ("Margen v2 + simulación", v2m_plus, y),
                        ("Total v2", v2t, tot), ("Total solo simulación", simt, tot), ("Total v2 + simulación", v2t_plus, tot)):
        print(f"{name:40} {mae(p, yy, tune):8.3f} {mae(p, yy, hold):8.3f}")

    # Forma de la distribución: simulación escalada a las medias de la v2 contra normal + números clave
    ok = ~np.isnan(v2m) & ~np.isnan(v2t)
    ll_sim, ll_key, br_sim, br_key, ss = [], [], [], [], []
    for s in np.unique(season[ok]):
        tr = ok & (season < s)
        if tr.sum() < 500:
            continue
        sigma = float(np.std(y[tr] - v2m[tr]))
        dist = MarginDistribution.fit(v2m[tr], y[tr], sigma)
        for i in np.where(ok & (season == s))[0]:
            g = sim[i][2]
            th, ta = (v2t[i] + v2m[i]) / 2, (v2t[i] - v2m[i]) / 2
            h = scale_to_points(*g["home"], g["drives"], th, g["extra"])
            a = scale_to_points(*g["away"], g["drives"], ta, g["extra"])
            mp, _ = game_distributions(team_points_pmf(*h, g["drives"], g["extra"]), team_points_pmf(*a, g["drives"], g["extra"]))
            k = int(round(y[i]))
            p_sim = mp[k + MAX_PTS] if abs(k) <= MAX_PTS else 1e-9
            pk = dist.pmf(v2m[i])
            p_key = pk[k - SUPPORT[0]] if SUPPORT[0] <= k <= SUPPORT[-1] else 1e-9
            ll_sim.append(-np.log(max(p_sim, 1e-9))); ll_key.append(-np.log(max(p_key, 1e-9))); ss.append(s)
            # probabilidad de cubrir la línea de cierre (sin calibrar)
            line = -spread[i]
            margins = np.arange(-MAX_PTS, MAX_PTS + 1)
            adj = margins + line
            c_sim = mp[adj > 0].sum() / max(1 - mp[np.isclose(adj, 0)].sum(), 1e-9)
            pr = dist.probs(v2m[i], line)
            c_key = pr["home_cover"] / max(1 - pr["push"], 1e-9)
            real = y[i] + line
            if real != 0:
                br_sim.append((s, c_sim, 1.0 if real > 0 else 0.0)); br_key.append((s, c_key, 1.0 if real > 0 else 0.0))
    ll_sim, ll_key, ss = np.array(ll_sim), np.array(ll_key), np.array(ss)
    print("\nMargen exacto (log-loss, menor es mejor):")
    for lab, m in (("2015–20", (ss >= 2015) & (ss <= 2020)), ("2021–26", ss >= 2021)):
        print(f"  {lab}: normal + números clave {ll_key[m].mean():.4f} · simulación por series {ll_sim[m].mean():.4f}")
    bs, bk = np.array(br_sim), np.array(br_key)
    h = bs[:, 0] >= 2021
    print(f"Cubrir la línea (Brier crudo 2021–26): normal + números clave {np.mean((bk[h,1]-bk[h,2])**2):.4f} · "
          f"simulación {np.mean((bs[h,1]-bs[h,2])**2):.4f}")


if __name__ == "__main__":
    main()
