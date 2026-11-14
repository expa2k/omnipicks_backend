"""
OmniPicks — Fútbol v2, fase 2: ¿los tiros a puerta mejoran a Dixon-Coles?

Los ratings se ajustan con una mezcla:  y = w · goles + (1 − w) · 0.289 · tiros a puerta.
w = 1 es la fase 1 (solo goles). El rho (marcadores bajos) sale del ajuste con goles.
Mismo backtest semana a semana; w se elige con 2019–2021 y se reporta 2022–2026.

Uso:
    python ml/scripts/soccer_shots_eval.py
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.soccer_dc import GOALS_PER_SOT, DCParams, fit_dc, fit_rates, markets, score_matrix
from app.services.soccer_history import load_all
from soccer_dc_eval import FIRST_EVAL, base_scores, scores

LEAGUES = ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"]
WEIGHTS = (1.0, 0.7, 0.5, 0.3, 0.0)


def walk(df: pl.DataFrame, params: DCParams):
    df = df.filter(pl.col("FTHG").is_not_null() & pl.col("date").is_not_null()).sort("date")
    dates = df["date"].to_numpy().astype("datetime64[D]")
    home, away = df["HomeTeam"].to_list(), df["AwayTeam"].to_list()
    hg, ag = df["FTHG"].to_numpy(), df["FTAG"].to_numpy()
    hst = df["HST"].fill_null(strategy="zero").to_numpy() if "HST" in df.columns else np.zeros(len(hg))
    ast = df["AST"].fill_null(strategy="zero").to_numpy() if "AST" in df.columns else np.zeros(len(ag))
    has = (df["HST"].is_not_null() & df["AST"].is_not_null()).to_numpy()
    season = df["season"].to_numpy()
    outs = {w: np.full((df.height, 5), np.nan) for w in WEIGHTS}
    week = (dates - np.datetime64("2000-01-03")).astype(int) // 7
    fit_g, fits = None, {w: None for w in WEIGHTS}
    for wk in np.unique(week):
        te = np.where(week == wk)[0]
        if season[te[0]] < FIRST_EVAL:
            continue
        tr = np.where(week < wk)[0]
        if len(tr) < 300:
            continue
        days = (dates[te[0]] - dates[tr]).astype(int).astype(float)
        H, A = [home[i] for i in tr], [away[i] for i in tr]
        fit_g = fit_dc(H, A, hg[tr], ag[tr], days, params, init=fit_g)
        for w in WEIGHTS:
            if w == 1.0:
                f = fit_g
            else:
                # Sin tiros en ese partido: solo goles
                yh = np.where(has[tr], w * hg[tr] + (1 - w) * GOALS_PER_SOT * hst[tr], hg[tr])
                ya = np.where(has[tr], w * ag[tr] + (1 - w) * GOALS_PER_SOT * ast[tr], ag[tr])
                f = fits[w] = fit_rates(H, A, yh, ya, days, params, init=fits[w], rho=fit_g.rho)
            for i in te:
                m = f.score_matrix(home[i], away[i])
                mk = markets(m)
                outs[w][i] = [mk["home"], mk["draw"], mk["away"], m[min(int(hg[i]), 10), min(int(ag[i]), 10)],
                              mk["exp_home"] + mk["exp_away"]]
    return df, outs


def main():
    data = load_all(2010)
    res = {lg: walk(data.filter(pl.col("league") == lg), DCParams()) for lg in LEAGUES}
    for lo, hi in ((2019, 2021), (2022, 2026)):
        print(f"\nTemporadas {lo}–{hi} · log loss 1X2 (y error de goles totales)")
        print(f"{'Liga':16} " + " ".join(f"{'w=' + str(w):>14}" for w in WEIGHTS) + f" {'Pinnacle':>9}")
        agg = {w: [] for w in WEIGHTS}
        pins, ns = [], []
        for lg, (df, outs) in res.items():
            s = df["season"].to_numpy()
            m = (s >= lo) & (s <= hi)
            cells, pin = [], None
            for w in WEIGHTS:
                sc = scores(df, outs[w], m)
                cells.append(f"{sc['ll']:.4f} ({sc['mae_total']:.2f})")
                agg[w].append((sc["ll"], sc["mae_total"], sc["n"]))
                pin = sc["ll_pin"]
            pins.append(pin); ns.append(agg[1.0][-1][2])
            print(f"{lg:16} " + " ".join(f"{c:>14}" for c in cells) + f" {pin:9.4f}")
        tot = []
        for w in WEIGHTS:
            ll = np.average([a[0] for a in agg[w]], weights=[a[2] for a in agg[w]])
            mt = np.average([a[1] for a in agg[w]], weights=[a[2] for a in agg[w]])
            tot.append(f"{ll:.4f} ({mt:.2f})")
        print(f"{'Todas':16} " + " ".join(f"{c:>14}" for c in tot) + f" {np.average(pins, weights=ns):9.4f}")


if __name__ == "__main__":
    main()
