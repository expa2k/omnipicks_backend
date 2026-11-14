"""
OmniPicks — Fútbol v2, fase 3: ¿el modelo conjunto (ligas + copas con fuerza por liga) predice bien
la Champions y la Europa League?

Semana a semana (solo con lo jugado antes) se predicen los partidos de copa con:
  · base: frecuencias de local / empate / visitante en copas de temporadas previas;
  · "solo ligas": el mismo modelo sin fuerza por liga y sin partidos de copa (cada equipo con su
    rating dentro de su liga, como si todas las ligas fueran iguales);
  · conjunto: ligas + copas, con fuerza por liga.
Y se revisa que en las ligas no empeore.

Uso:
    python ml/scripts/soccer_cups_eval.py
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.soccer_cups import load_cups, match_names
from app.services.soccer_dc import markets
from app.services.soccer_history import load_all
from app.services.soccer_joint import EU_LEAGUES, JointModel, JointParams, build_table

FIRST = 2019


def outcome(hg, ag):
    return 0 if hg > ag else 1 if hg == ag else 2


def main():
    leagues = load_all(2013)
    cups = load_cups(2016)
    fd = set(leagues.filter(pl.col("league").is_in(EU_LEAGUES))["HomeTeam"].to_list())
    names = set(cups["home"].drop_nulls().to_list()) | set(cups["away"].drop_nulls().to_list())
    nm = match_names(names, fd)
    table = build_table(leagues, cups, nm)
    model = JointModel(table, JointParams())
    no_cup = JointModel(table.filter(pl.col("comp").is_in(EU_LEAGUES)), JointParams(ridge_league=1e6))

    is_cup = ~table["comp"].is_in(EU_LEAGUES).to_numpy()
    season = table["season"].to_numpy()
    dates = table["date"].to_numpy().astype("datetime64[D]")
    week = (dates - np.datetime64("2000-01-03")).astype(int) // 7
    cup_weeks = np.unique(week[is_cup & (season >= FIRST)])
    res = {"base": [], "solo ligas": [], "conjunto": []}
    meta = []
    fj, fn = None, None
    for wk in cup_weeks:
        te = np.where((week == wk) & is_cup)[0]
        ref = dates[te[0]] - np.timedelta64(int(dates[te[0]].astype("datetime64[D]").view("int64") % 7), "D")
        ref = dates[te].min()
        fj = model.fit(ref, init=fj)
        fn = no_cup.fit(ref, init=fn)
        prev = is_cup & (dates < ref)
        y_prev = [outcome(h, a) for h, a in zip(table["hg"].to_numpy()[prev], table["ag"].to_numpy()[prev])]
        base = np.bincount(y_prev, minlength=3) / max(len(y_prev), 1)
        for i in te:
            r = table.row(int(i), named=True)
            y = outcome(r["hg"], r["ag"])
            mj = markets(fj.score_matrix(r["comp"], r["home"], r["away"], r["neutral"]))
            # "solo ligas": sin fuerza por liga y con la media/localía de una liga típica
            lh, la = fn.rates(fn.comps[0], r["home"], r["away"], r["neutral"])
            from app.services.soccer_dc import score_matrix
            mn = markets(score_matrix(lh, la, -0.05))
            res["base"].append(base[y])
            res["conjunto"].append([mj["home"], mj["draw"], mj["away"]][y])
            res["solo ligas"].append([mn["home"], mn["draw"], mn["away"]][y])
            known = (r["home"] in fj.team_league) + (r["away"] in fj.team_league)
            meta.append((r["season"], r["comp"], known))
    meta = np.array(meta, dtype=object)
    print(f"Partidos de copa evaluados: {len(meta)} (temporadas {FIRST}–2026)\n")
    print(f"{'':42} {'base':>7} {'solo ligas':>11} {'conjunto':>9}")
    for label, sel in (
        ("Todos", np.ones(len(meta), bool)),
        ("Champions", meta[:, 1] == "UEFA Champions League"),
        ("Europa League", meta[:, 1] == "UEFA Europa League"),
        ("Los dos equipos de nuestras ligas", meta[:, 2] == 2),
        ("Uno de nuestras ligas y otro de fuera", meta[:, 2] == 1),
        ("Los dos de otras ligas", meta[:, 2] == 0),
        ("2019–2021", meta[:, 0] <= 2021),
        ("2022–2026", meta[:, 0] >= 2022),
    ):
        ll = {k: float(-np.mean(np.log(np.clip(np.array(v)[sel], 1e-9, None)))) for k, v in res.items()}
        print(f"{label + f' ({int(sel.sum())})':42} {ll['base']:7.4f} {ll['solo ligas']:11.4f} {ll['conjunto']:9.4f}")
    # Fuerza por liga hoy
    f = model.fit(np.datetime64("2026-11-13"))
    print("\nFuerza por liga hoy (ataque − defensa rival media, en goles por partido contra un rival medio):")
    order = np.argsort(-(f.A + f.D))
    for k in order:
        print(f"  {f.leagues[k]:16} {f.A[k] + f.D[k]:+.2f}")


if __name__ == "__main__":
    main()
