"""
OmniPicks — NFL v3, fase 1: ¿la fuerza del equipo armada con el valor de sus jugadores (RAPM)
mejora el pronóstico de la v2?

  · RAPM de cada temporada entrenado SOLO con las 3 anteriores (sin ver la temporada que predice).
  · Fuerza del equipo = Σ valor del jugador × fracción de jugadas:
      - "esperada": su uso en los partidos previos de la temporada (lo que se sabe antes del partido);
      - "real": quién jugó de verdad ese partido (techo: como si se supieran todas las bajas).
  · Se agrega a las variables del margen de la v2 y se compara el error en 2018–2025
    (prueba limpia 2021–2025; la participación de nflverse llega hasta 2025).

Uso:
    python ml/scripts/nfl_players_eval.py
"""
import itertools
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services import nfl_v2
from app.services.nfl_players import (RapmParams, expected_usage, fit_rapm, load_participation, play_table,
                                      team_strength, usage_from_plays)

SEASONS = range(2018, 2026)
PLAYS_PER_GAME = 62.0


def walk(X, y, season, first=2018):
    pred = np.full(len(y), np.nan)
    for s in np.unique(season):
        tr = (season >= first) & (season < s)
        if tr.sum() < 200:
            continue
        c = nfl_v2._fit(X[tr], y[tr])
        te = season == s
        pred[te] = np.c_[np.ones(te.sum()), X[te]] @ c
    return pred


def strength_features(rows, plays, use, exp, params):
    values = {s: fit_rapm(plays, s, params) for s in SEASONS}
    actual = {}
    for r in use.iter_rows(named=True):
        actual.setdefault((r["game_id"], r["team"]), {"o": {}, "d": {}})[r["side"]][r["p"]] = r["share"]
    xa, xe = [], []
    for r in rows:
        v = values[r["season"]]
        h, a = r["home_team"], r["away_team"]
        ah, aa = actual.get((r["game_id"], h)), actual.get((r["game_id"], a))
        if ah and aa:
            oh, dh = team_strength(v, ah); oa, da = team_strength(v, aa)
            xa.append(((oh + dh) - (oa + da)) * PLAYS_PER_GAME)
        else:
            xa.append(np.nan)
        eh, ea = exp.get((r["season"], r["week"], h)), exp.get((r["season"], r["week"], a))
        if eh and ea and eh["o"] and ea["o"]:
            oh, dh = team_strength(v, eh); oa, da = team_strength(v, ea)
            xe.append(((oh + dh) - (oa + da)) * PLAYS_PER_GAME)
        else:
            xe.append(np.nan)
    return np.array(xa), np.array(xe)


def main():
    sched, pbp = nfl_v2.load_data(False, refresh_schedules=False)
    feats = nfl_v2.Features(sched, pbp)
    played = sched.filter(pl.col("result").is_not_null() & pl.col("season").is_in(list(SEASONS)))
    rows = list(played.iter_rows(named=True))
    M = np.array([[feats.row(r)[0][k] for k in nfl_v2.MARGIN_FEATURES] for r in rows])
    y = played["result"].to_numpy().astype(float)
    season = played["season"].to_numpy()
    week = played["week"].to_numpy()
    spread = played["spread_line"].to_numpy().astype(float)

    plays = play_table(pbp, load_participation(range(2015, 2026)))
    use = usage_from_plays(plays)
    exp = expected_usage(use)

    best = None
    print("Búsqueda del encogimiento del RAPM (elegido con 2019–2020, ajustando en 2018+):")
    for qb, other, hl in itertools.product((50, 150), (150, 400, 800), (2.0,)):
        prm = RapmParams(season_half_life=hl, ridge={"QB": qb, "default": other})
        xa, xe = strength_features(rows, plays, use, exp, prm)
        ok = ~np.isnan(xe)
        p = walk(np.c_[M, np.nan_to_num(xe)][ok], y[ok], season[ok])
        tune = (season[ok] >= 2019) & (season[ok] <= 2020) & ~np.isnan(p)
        mae = float(np.mean(np.abs(p[tune] - y[ok][tune])))
        print(f"  ridge QB {qb:>4} · resto {other:>5} · vida media {hl} temp. → MAE 19–20 {mae:.3f}")
        if best is None or mae < best[0]:
            best = (mae, prm, xa, xe)
    _, prm, xa, xe = best
    print(f"Elegido: {prm}\n")

    ok = ~np.isnan(xe) & ~np.isnan(xa)
    res = {
        "v2 (EPA, marcador, QB)": walk(M[ok], y[ok], season[ok]),
        "v2 + jugadores (uso esperado)": walk(np.c_[M, xe][ok], y[ok], season[ok]),
        "v2 + jugadores (quién jugó: techo)": walk(np.c_[M, xa][ok], y[ok], season[ok]),
        "v2 + esperado + efecto de las bajas": walk(np.c_[M, xe, xa - xe][ok], y[ok], season[ok]),
        "solo jugadores (uso esperado)": walk(xe[ok][:, None], y[ok], season[ok]),
        "mercado": spread[ok],
    }
    yy, ss = y[ok], season[ok]
    print(f"{'':38} {'2019–2020':>10} {'2021–2025':>10} {'semanas 2–6':>12}")
    for name, p in res.items():
        m = ~np.isnan(p)
        f = lambda sel: float(np.mean(np.abs(p[sel & m] - yy[sel & m])))
        early = (week[ok] >= 2) & (week[ok] <= 6) & (ss >= 2021)
        print(f"{name:38} {f((ss >= 2019) & (ss <= 2020)):10.3f} {f(ss >= 2021):10.3f} {f(early):12.3f}")
    d = (xa - xe)[ok]
    print(f"\nEfecto de las bajas (quién jugó − esperado), en EPA por partido: desviación {np.std(d):.2f}, "
          f"percentil 95 de |efecto| {np.percentile(np.abs(d), 95):.2f}")
    print(f"\nPartidos evaluados: {int(ok.sum()):,} (semana 2 en adelante, 2018–2025)")


if __name__ == "__main__":
    main()
