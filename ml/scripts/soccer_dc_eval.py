"""
OmniPicks — Fútbol v2, fase 1: backtest de Dixon-Coles jornada a jornada.

Para cada liga y cada semana se ajusta el modelo SOLO con los partidos anteriores y se predicen los de
esa semana. Se compara contra:
  · base: las frecuencias de local / empate / visitante de la liga (lo mínimo);
  · Pinnacle al cierre, sin su margen (la referencia más dura).

Se eligen ξ y ridge con 2019–2021 y se reporta 2022–2026 aparte (prueba limpia).

Uso:
    python ml/scripts/soccer_dc_eval.py                 # parámetros por defecto
    python ml/scripts/soccer_dc_eval.py --grid          # busca ξ y ridge
"""
import argparse
import itertools
import os
import sys
import time
from datetime import timedelta

import numpy as np
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.soccer_dc import DCParams, fit_dc, markets, score_matrix
from app.services.soccer_history import load_all

LEAGUES = ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1", "Liga MX"]
FIRST_EVAL = 2019


def walk_league(df: pl.DataFrame, params: DCParams):
    df = df.filter(pl.col("FTHG").is_not_null() & pl.col("date").is_not_null()).sort("date")
    dates = df["date"].to_numpy()
    home, away = df["HomeTeam"].to_list(), df["AwayTeam"].to_list()
    hg, ag = df["FTHG"].to_numpy(), df["FTAG"].to_numpy()
    season = df["season"].to_numpy()
    out = np.full((df.height, 5), np.nan)      # p_home, p_draw, p_away, p_exact, exp_total
    fit = None
    # Se reajusta una vez por semana (lunes): predice todos los partidos de esa semana
    week = (dates.astype("datetime64[D]") - np.datetime64("2000-01-03")).astype(int) // 7
    for wk in np.unique(week):
        te = np.where(week == wk)[0]
        if season[te[0]] < FIRST_EVAL:
            continue
        tr = np.where(week < wk)[0]
        if len(tr) < 300:
            continue
        ref = dates[te[0]].astype("datetime64[D]")
        days = (ref - dates[tr].astype("datetime64[D]")).astype(int).astype(float)
        fit = fit_dc([home[i] for i in tr], [away[i] for i in tr], hg[tr], ag[tr], days, params, init=fit)
        for i in te:
            m = fit.score_matrix(home[i], away[i])
            mk = markets(m)
            h, a = int(hg[i]), int(ag[i])
            out[i] = [mk["home"], mk["draw"], mk["away"], m[min(h, 10), min(a, 10)], mk["exp_home"] + mk["exp_away"]]
    return df, out


def scores(df, out, mask):
    y = np.select([df["FTHG"].to_numpy() > df["FTAG"].to_numpy(), df["FTHG"].to_numpy() == df["FTAG"].to_numpy()], [0, 1], 2)
    ok = mask & ~np.isnan(out[:, 0])
    p = out[ok, :3]
    yy = y[ok]
    ll = float(-np.mean(np.log(np.clip(p[np.arange(len(yy)), yy], 1e-9, None))))
    onehot = np.eye(3)[yy]
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
    exact = float(-np.mean(np.log(np.clip(out[ok, 3], 1e-9, None))))
    tot = (df["FTHG"].to_numpy() + df["FTAG"].to_numpy())[ok]
    mae_t = float(np.mean(np.abs(out[ok, 4] - tot)))
    # Pinnacle al cierre sin margen
    ph, pd_, pa = (df[c].to_numpy()[ok] for c in ("PSCH", "PSCD", "PSCA"))
    has = ~np.isnan(ph) & ~np.isnan(pd_) & ~np.isnan(pa)
    if has.any():
        inv = np.c_[1 / ph[has], 1 / pd_[has], 1 / pa[has]]
        q = inv / inv.sum(axis=1, keepdims=True)
        ll_m = float(-np.mean(np.log(q[np.arange(has.sum()), yy[has]])))
        ll_mod_same = float(-np.mean(np.log(np.clip(p[has][np.arange(has.sum()), yy[has]], 1e-9, None))))
        br_m = float(np.mean(np.sum((q - np.eye(3)[yy[has]]) ** 2, axis=1)))
    else:
        ll_m = ll_mod_same = br_m = np.nan
    return {"n": int(ok.sum()), "ll": ll, "brier": brier, "exact": exact, "mae_total": mae_t,
            "ll_pin": ll_m, "ll_mod_same": ll_mod_same, "brier_pin": br_m}


def base_scores(df, mask):
    """Frecuencias de la liga en temporadas previas (lo mínimo)."""
    y = np.select([df["FTHG"].to_numpy() > df["FTAG"].to_numpy(), df["FTHG"].to_numpy() == df["FTAG"].to_numpy()], [0, 1], 2)
    s = df["season"].to_numpy()
    lls = []
    for ss in np.unique(s[mask]):
        prev = s < ss
        f = np.bincount(y[prev], minlength=3) / max(prev.sum(), 1)
        cur = mask & (s == ss)
        lls.append(-np.log(f[y[cur]]))
    return float(np.mean(np.concatenate(lls)))


def run(data, params, verbose=True):
    res = {}
    for lg in LEAGUES:
        df, out = walk_league(data.filter(pl.col("league") == lg), params)
        s = df["season"].to_numpy()
        res[lg] = (df, out, s)
    return res


def summarize(res, period):
    lo, hi = period
    rows = {}
    for lg, (df, out, s) in res.items():
        m = (s >= lo) & (s <= hi)
        rows[lg] = {**scores(df, out, m), "ll_base": base_scores(df, m)}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()
    data = load_all(2010)
    params = DCParams()
    if args.grid:
        print("Búsqueda de ξ (descuento por día) y ridge, elegidos con 2019–2021 (log loss 1X2 promedio):")
        best = None
        for xi, rg in itertools.product((0.001, 0.002, 0.0035), (0.5, 2.0, 6.0)):
            t = time.time()
            res = run(data, DCParams(xi=xi, ridge=rg))
            sm = summarize(res, (2019, 2021))
            ll = np.average([v["ll"] for v in sm.values()], weights=[v["n"] for v in sm.values()])
            print(f"  ξ {xi:<7} ridge {rg:<4} → log loss {ll:.4f}  ({time.time() - t:.0f} s)")
            if best is None or ll < best[0]:
                best = (ll, DCParams(xi=xi, ridge=rg))
        params = best[1]
        print(f"Elegidos: {params}\n")
    res = run(data, params)
    for period in ((2019, 2021), (2022, 2026)):
        sm = summarize(res, period)
        print(f"\nTemporadas {period[0]}–{period[1]}  (log loss 1X2: menor es mejor)")
        print(f"{'Liga':16} {'n':>5} {'base':>7} {'modelo':>7} {'Pinnacle':>9} {'Brier':>7} {'Brier Pin':>10} {'marcador':>9} {'err. goles':>10}")
        tot = {"n": 0}
        for lg, v in sm.items():
            print(f"{lg:16} {v['n']:5} {v['ll_base']:7.4f} {v['ll']:7.4f} {v['ll_pin']:9.4f} {v['brier']:7.4f} {v['brier_pin']:10.4f} "
                  f"{v['exact']:9.3f} {v['mae_total']:10.3f}")
        w = [v["n"] for v in sm.values()]
        avg = lambda k: np.average([v[k] for v in sm.values()], weights=w)
        print(f"{'Todas':16} {sum(w):5} {avg('ll_base'):7.4f} {avg('ll'):7.4f} {avg('ll_pin'):9.4f} {avg('brier'):7.4f} "
              f"{avg('brier_pin'):10.4f} {avg('exact'):9.3f} {avg('mae_total'):10.3f}")


if __name__ == "__main__":
    main()
