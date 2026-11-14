"""
OmniPicks — MLB v2, fase 4: contra los momios de cierre de SBR (2015–2021).

Mismo recorrido de mlb_eval.py (cada 14 días, solo con lo anterior). Se compara el log loss del
ganador contra el del cierre sin margen, y se simulan apuestas de 1 unidad con los momios reales:
moneyline, run line (±1.5) y total, cuando la probabilidad del modelo da valor esperado > umbral,
y también por nivel de confianza del modelo (≥ 60%).

Uso:
    python ml/scripts/mlb_backtest.py
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.mlb_history import load_gamelogs, load_handedness, load_odds
from app.services.mlb_model import MLBParams, game_probs, team_rows
import mlb_eval

mlb_eval.FIRST, mlb_eval.LAST = 2015, 2021


def dec(a):
    return 1 + (a / 100 if a > 0 else 100 / -a)


def implied(a):
    return 1 / dec(a)


def main():
    gl = load_gamelogs(range(2012, 2022))
    rows = team_rows(gl, load_handedness(range(2012, 2022))).sort(["date", "game_id", "is_home"])
    lr, last = mlb_eval.walk(rows, MLBParams())
    r = last.dispersion
    gid, home, runs = rows["game_id"].to_list(), rows["is_home"].to_numpy() == 1, rows["runs"].to_numpy()
    ih = {g: i for i, (g, h) in enumerate(zip(gid, home)) if h}
    ia = {g: i for i, (g, h) in enumerate(zip(gid, home)) if not h}
    games = pl.DataFrame([{"game_id": g, "date": rows["date"][ih[g]], "home": rows["bat"][ih[g]], "away": rows["bat"][ia[g]],
                           "hs": runs[ih[g]], "as_": runs[ia[g]], "lh": float(np.exp(lr[ih[g]])), "la": float(np.exp(lr[ia[g]]))}
                          for g in ih if g in ia and not np.isnan(lr[ih[g]])])
    odds = load_odds(range(2015, 2022))
    j = games.join(odds, on=["date", "home", "away"], how="inner").filter(
        (pl.col("hs") == pl.col("final_home")) & (pl.col("as_") == pl.col("final_away"))).unique("game_id")
    print(f"Juegos con momios de cierre: {j.height:,}")
    ll_m, ll_k, bets = [], [], {k: [] for k in ("ml0", "ml3", "rl0", "tot0", "ml_conf", "tot_conf", "rl_conf")}
    for x in j.iter_rows(named=True):
        if x["ml_home"] is None or x["ml_away"] is None:
            continue
        y = 1.0 if x["hs"] > x["as_"] else 0.0
        g = game_probs(x["lh"], x["la"], r, run_line=x["rl_home"] if x["rl_home"] is not None else -1.5,
                       total=x["total"])
        p = min(max(g["home_win"], 0.02), 0.98)
        qh, qa = implied(x["ml_home"]), implied(x["ml_away"])
        q = qh / (qh + qa)
        ll_m.append(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
        ll_k.append(-(y * np.log(q) + (1 - y) * np.log(1 - q)))
        season = x["date"].year
        # Moneyline
        for side, pp, o, won in (("h", p, x["ml_home"], y == 1), ("a", 1 - p, x["ml_away"], y == 0)):
            ev = pp * dec(o) - 1
            pay = dec(o) - 1 if won else -1.0
            if ev > 0: bets["ml0"].append((season, pay))
            if ev > 0.03: bets["ml3"].append((season, pay))
            if pp >= 0.60: bets["ml_conf"].append((season, pay))
        # Run line
        if x["rl_home"] is not None and x["rl_home_odds"] is not None and x["rl_away_odds"] is not None:
            ph = g["home_cover"]
            m = x["hs"] - x["as_"]
            for pp, o, won in ((ph, x["rl_home_odds"], m + x["rl_home"] > 0), (1 - ph, x["rl_away_odds"], m + x["rl_home"] < 0)):
                pay = dec(o) - 1 if won else -1.0
                if pp * dec(o) - 1 > 0: bets["rl0"].append((season, pay))
                if pp >= 0.60: bets["rl_conf"].append((season, pay))
        # Total
        if x["total"] is not None and x["over_odds"] is not None and x["under_odds"] is not None and "over" in g:
            t = x["hs"] + x["as_"]
            if t != x["total"]:
                for pp, o, won in ((g["over"], x["over_odds"], t > x["total"]), (1 - g["over"], x["under_odds"], t < x["total"])):
                    pay = dec(o) - 1 if won else -1.0
                    if pp * dec(o) - 1 > 0: bets["tot0"].append((season, pay))
                    if pp >= 0.60: bets["tot_conf"].append((season, pay))
    print(f"Log loss del ganador: modelo {np.mean(ll_m):.4f} · cierre sin margen {np.mean(ll_k):.4f}  ({len(ll_m):,} juegos)")
    names = {"ml0": "Moneyline, valor > 0%", "ml3": "Moneyline, valor > 3%", "rl0": "Run line, valor > 0%",
             "tot0": "Total, valor > 0%", "ml_conf": "Moneyline, confianza ≥ 60%", "rl_conf": "Run line, confianza ≥ 60%",
             "tot_conf": "Total, confianza ≥ 60%"}
    print("\nApuestas de 1 unidad con momios de cierre:")
    for k, v in bets.items():
        if not v:
            print(f"  {names[k]:30} sin apuestas"); continue
        b = np.array(v)
        for lab, m in (("2015–2018", b[:, 0] <= 2018), ("2019–2021", b[:, 0] >= 2019)):
            g = b[m, 1]
            if len(g):
                se = g.std(ddof=1) / np.sqrt(len(g))
                print(f"  {names[k]:30} {lab}: {len(g):>5} · {g.sum():+8.1f} u · ROI {g.mean():+.1%} (±{1.96 * se:.1%})")


if __name__ == "__main__":
    main()
