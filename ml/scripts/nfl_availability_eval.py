"""
OmniPicks — NFL v3, fase 3: ¿predecir quién juega con el reporte de lesiones (todas las posiciones)
mejora el pronóstico?

  · P(juega | estado, práctica) aprendida de 2016–2020 y aplicada a 2021–2025.
  · Fuerza del equipo = uso esperado − bajas previstas × (valor del jugador − el de un suplente).
  · Comparado contra: v2, + jugadores (uso esperado, sin lesiones) y el techo (quién jugó de verdad).

Uso:
    python ml/scripts/nfl_availability_eval.py
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services import nfl_v2
from app.services.nfl_players import (RapmParams, expected_usage, fit_rapm, injury_adjustment, learn_availability,
                                      load_injury_reports, load_participation, play_table, replacement_levels,
                                      team_strength, usage_from_plays)
from nfl_players_eval import PLAYS_PER_GAME, SEASONS, walk

PRM = RapmParams(season_half_life=2.0, ridge={"QB": 50, "default": 800})


def main():
    sched, pbp = nfl_v2.load_data(False, refresh_schedules=False)
    feats = nfl_v2.Features(sched, pbp)
    played = sched.filter(pl.col("result").is_not_null() & pl.col("season").is_in(list(SEASONS)))
    rows = list(played.iter_rows(named=True))
    M = np.array([[feats.row(r)[0][k] for k in nfl_v2.MARGIN_FEATURES] for r in rows])
    y = played["result"].to_numpy().astype(float)
    season = played["season"].to_numpy()

    plays = play_table(pbp, load_participation(range(2016, 2026)))
    use = usage_from_plays(plays)
    exp = expected_usage(use)
    inj = load_injury_reports(range(2016, 2026))
    table = learn_availability(inj.filter(pl.col("season") <= 2020), use, exp)
    print("P(juega) aprendida con 2016–2020 (titulares y rotación):")
    for st in ("questionable", "doubtful", "out", "none"):
        if st in table:
            print(f"  {st:12} {table[st]}")
    report = {}
    for r in inj.iter_rows(named=True):
        report.setdefault((r["season"], r["week"], r["team"]), {})[r["gsis_id"]] = (r["report_status"], r["practice_status"])

    values = {s: fit_rapm(plays, s, PRM) for s in SEASONS}
    repl = {s: replacement_levels(v) for s, v in values.items()}
    actual = {}
    for r in use.iter_rows(named=True):
        actual.setdefault((r["game_id"], r["team"]), {"o": {}, "d": {}})[r["side"]][r["p"]] = r["share"]

    xe, xi, xa, lost_h, lost_a = [], [], [], [], []
    for r in rows:
        v, rp = values[r["season"]], repl[r["season"]]
        h, a = r["home_team"], r["away_team"]
        eh, ea = exp.get((r["season"], r["week"], h)), exp.get((r["season"], r["week"], a))
        ah, aa = actual.get((r["game_id"], h)), actual.get((r["game_id"], a))
        if not (eh and ea and eh["o"] and ea["o"] and ah and aa):
            xe.append(np.nan); xi.append(np.nan); xa.append(np.nan); lost_h.append(0); lost_a.append(0)
            continue
        oh, dh = team_strength(v, eh); oa, da = team_strength(v, ea)
        base = ((oh + dh) - (oa + da)) * PLAYS_PER_GAME
        lh, _ = injury_adjustment(v, eh, report.get((r["season"], r["week"], h), {}), table, rp)
        la, _ = injury_adjustment(v, ea, report.get((r["season"], r["week"], a), {}), table, rp)
        xe.append(base)
        xi.append(base - (lh - la) * PLAYS_PER_GAME)
        lost_h.append(lh * PLAYS_PER_GAME); lost_a.append(la * PLAYS_PER_GAME)
        o1, d1 = team_strength(v, ah); o2, d2 = team_strength(v, aa)
        xa.append(((o1 + d1) - (o2 + d2)) * PLAYS_PER_GAME)
    xe, xi, xa = map(np.array, (xe, xi, xa))
    inj_diff = np.array(lost_h) - np.array(lost_a)

    ok = ~np.isnan(xe)
    yy, ss = y[ok], season[ok]
    res = {
        "v2 (EPA, marcador, QB)": walk(M[ok], yy, ss),
        "+ jugadores (uso esperado, sin lesiones)": walk(np.c_[M, xe][ok], yy, ss),
        "+ jugadores con bajas previstas (fase 3)": walk(np.c_[M, xe, inj_diff][ok], yy, ss),
        "+ jugadores (quién jugó: techo)": walk(np.c_[M, xe, xa - xe][ok], yy, ss),
        "mercado": played["spread_line"].to_numpy().astype(float)[ok],
    }
    big = np.abs(inj_diff[ok]) >= np.percentile(np.abs(inj_diff[ok]), 80)
    print(f"\n{'':44} {'2019–2020':>10} {'2021–2025':>10} {'2021–25, bajas grandes':>23}")
    for name, p in res.items():
        m = ~np.isnan(p)
        f = lambda sel: float(np.mean(np.abs(p[sel & m] - yy[sel & m])))
        print(f"{name:44} {f((ss >= 2019) & (ss <= 2020)):10.3f} {f(ss >= 2021):10.3f} {f((ss >= 2021) & big):23.3f}")
    print(f"\n'Bajas grandes' = el 20% de partidos con más diferencia de bajas previstas entre los equipos "
          f"(≥ {np.percentile(np.abs(inj_diff[ok]), 80):.1f} EPA por partido).")


if __name__ == "__main__":
    main()
