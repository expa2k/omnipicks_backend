"""
OmniPicks — Fútbol v2: backtest del modelo de selecciones (2018–2026).

Cada mes se ajusta con lo jugado antes y se predicen los partidos oficiales (no amistosos) de ese mes.
Se compara contra la base (frecuencias de local/empate/visita en partidos oficiales previos, separando
cancha neutral) y se reporta aparte la Nations League.

Uso:
    python ml/scripts/soccer_national_eval.py
"""
import itertools
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services import soccer_national as sn
from app.services.soccer_dc import DCParams, markets


def run(df, params, friendly_w):
    sn.PARAMS, sn.FRIENDLY_WEIGHT = params, friendly_w
    ev = df.filter((pl.col("date") >= pl.date(2018, 1, 1)) & pl.col("home_score").is_not_null()
                   & (pl.col("tournament") != "Friendly")).with_columns(month=pl.col("date").dt.strftime("%Y-%m"))
    rows = []
    for mo, g in ev.group_by("month", maintain_order=True):
        ref = np.datetime64(g["date"].min().isoformat())
        fit = sn.fit_national(df, ref)
        known = set(fit.teams)
        for r in g.iter_rows(named=True):
            if r["home_team"] not in known or r["away_team"] not in known:
                continue
            mk = markets(fit.score_matrix(r["home_team"], r["away_team"], bool(r["neutral"])))
            y = 0 if r["home_score"] > r["away_score"] else 1 if r["home_score"] == r["away_score"] else 2
            rows.append((r["date"].year, r["tournament"], bool(r["neutral"]), [mk["home"], mk["draw"], mk["away"]][y], y))
    return rows


def main():
    df = sn.load_results()
    print(f"Partidos internacionales en la base: {df.height:,} (último {df['date'].max()})")
    best = None
    for xi, rg, fw in itertools.product((0.0005, 0.0008, 0.0015), (1.0, 3.0), (0.5, 1.0)):
        rows = run(df, DCParams(xi=xi, ridge=rg, window_days=2900), fw)
        tune = [r for r in rows if r[0] <= 2021]
        ll = -np.mean(np.log(np.clip([r[3] for r in tune], 1e-9, None)))
        print(f"  ξ {xi} · ridge {rg} · amistosos {fw} → log loss 2018–21 {ll:.4f} ({len(tune)})")
        if best is None or ll < best[0]:
            best = (ll, xi, rg, fw, rows)
    _, xi, rg, fw, rows = best
    print(f"Elegidos: ξ {xi}, ridge {rg}, amistosos {fw}\n")
    # Base: frecuencias por tipo de sede en partidos oficiales previos
    base = {}
    for nz in (False, True):
        prev = [r[4] for r in rows if r[2] == nz and r[0] <= 2021]
        base[nz] = np.bincount(prev, minlength=3) / max(len(prev), 1)
    def show(label, sel):
        rr = [r for r in rows if sel(r)]
        if not rr:
            return
        ll = -np.mean(np.log(np.clip([r[3] for r in rr], 1e-9, None)))
        lb = -np.mean(np.log([base[r[2]][r[4]] for r in rr]))
        print(f"  {label:40} {len(rr):5} partidos · base {lb:.4f} · modelo {ll:.4f}")
    print("Log loss del 1X2 (menor es mejor):")
    show("2022–2026, todos los oficiales", lambda r: r[0] >= 2022)
    show("2022–2026, UEFA Nations League", lambda r: r[0] >= 2022 and r[1] == "UEFA Nations League")
    show("2022–2026, CONCACAF Nations League", lambda r: r[0] >= 2022 and r[1] == "CONCACAF Nations League")
    show("2022–2026, eliminatorias mundialistas", lambda r: r[0] >= 2022 and r[1] == "FIFA World Cup qualification")
    show("2022 y 2026, Mundial", lambda r: r[1] == "FIFA World Cup" and r[0] >= 2022)


if __name__ == "__main__":
    main()
