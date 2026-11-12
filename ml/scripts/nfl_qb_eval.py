"""
OmniPicks — NFL: ¿cuánto mejora el margen previsto al tener en cuenta al QB titular?

Mismo recorrido semana a semana que nfl_ratings_eval.py. A los ratings de equipo (EPA + marcador)
se les suma el ajuste por QB titular (nfl_qb). El titular es el que reporta el calendario de
nflverse, que en producción se sabe antes del partido (depth chart y reporte de lesiones).

Los parámetros del QB se eligen con 2015–2020 y se reportan aparte 2021–2026, que no se usaron
para elegirlos.

Uso:
    python ml/scripts/nfl_qb_eval.py
"""
import itertools
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.nfl_history import load_pbp, team_games
from app.services.nfl_qb import QBParams, QBRatingEngine, qb_games, qb_margin_features
from app.services.nfl_ratings import RatingParams
from nfl_ratings_eval import (elo_predictions, games_table, metrics, points_features, rating_features,
                              walk_forward)

TUNE = (2015, 2020)
HOLDOUT = (2021, 2026)


def qb_features(games: pl.DataFrame, qbg: pl.DataFrame, params: QBParams) -> np.ndarray:
    eng = QBRatingEngine(qbg, params)
    cache, x = {}, []
    for r in games.iter_rows(named=True):
        key = (r["season"], r["week"])
        if key not in cache:
            cache[key] = eng.at(*key)
        x.append(qb_margin_features(cache[key], r["home_team"], r["away_team"], r["home_qb_id"], r["away_qb_id"]))
    return np.array(x)


def report(title: str, rows: dict, y: np.ndarray, mask: np.ndarray):
    print(f"\n{title} ({int(mask.sum()):,} partidos)")
    print(f"{'':36} {'MAE':>6} {'RMSE':>6} {'Brier':>7} {'Acierto':>8}")
    for name, pred in rows.items():
        m = metrics(np.where(mask, pred, np.nan), y)
        print(f"{name:36} {m['MAE']:6.2f} {m['RMSE']:6.2f} {m['Brier']:7.4f} {m['Acierto']:8.1%}")


def main():
    games = games_table()
    pbp = load_pbp(range(2010, 2027))
    tg, qbg = team_games(pbp), qb_games(pbp)
    y = games["result"].to_numpy().astype(float)
    season = games["season"].to_numpy()
    tune = (season >= TUNE[0]) & (season <= TUNE[1])
    hold = (season >= HOLDOUT[0]) & (season <= HOLDOUT[1])

    x_epa = rating_features(games, tg, RatingParams(half_life=16, offseason_gap=12, ridge=300))
    x_pts = points_features(games, RatingParams(half_life=16, offseason_gap=12), ridge=8.0)
    base = np.c_[x_epa, x_pts]

    print("Búsqueda de parámetros del QB (elegidos con 2015–2020):")
    best = None
    for hl, pn, pd in itertools.product((32, 64, 128), (50, 100, 250), (-0.05, -0.10, -0.15)):
        prm = QBParams(half_life=hl, prior_n=pn, prior_delta=pd)
        xq = qb_features(games, qbg, prm)
        pred = walk_forward(np.c_[base, xq[:, 0]], y, season)
        mae = metrics(np.where(tune, pred, np.nan), y)["MAE"]
        print(f"  vida media {hl:>2} · prior {pn:>3} dropbacks · suplente {pd:+.2f} → MAE {mae:.3f}")
        if best is None or mae < best[0]:
            best = (mae, prm, xq)
    _, prm, xq = best
    print(f"\nMejores: {prm}")

    rows = {
        "Elo clásico": walk_forward(elo_predictions(games), y, season),
        "Fase 1: EPA + marcador": walk_forward(base, y, season),
        "Fase 2: + ajuste por QB titular": walk_forward(np.c_[base, xq[:, 0]], y, season),
        "Fase 2: + ajuste y calidad del QB": walk_forward(np.c_[base, xq], y, season),
        "Mercado (línea de cierre)": games["spread_line"].to_numpy().astype(float),
    }
    report("Temporadas 2015–2020 (con las que se eligieron los parámetros)", rows, y, tune)
    report("Temporadas 2021–2026 (no usadas para elegir)", rows, y, hold)

    # Partidos con cambio de QB: el ajuste de algún equipo vale más de 2 puntos de EPA
    change = (np.abs(xq[:, 0]) > 2.0) & (season >= TUNE[0])
    report("Partidos con cambio de QB relevante (2015–2026)", rows, y, change)

    # Cuánto vale el QB: los ajustes más grandes
    eng = QBRatingEngine(qbg, prm)
    w = eng.at(2026, 10)
    names = dict(zip(qbg["id"].to_list(), qbg["name"].to_list()))
    top = sorted(((v, q) for q, v in w.rating.items() if w.dropbacks.get(q, 0) > 150), reverse=True)
    print("\nQBs antes de la semana 10 de 2026 (EPA por dropback sobre la media):")
    for v, q in top[:8]:
        print(f"  {names.get(q, q):22} {v:+.3f}")
    print("  …")
    for v, q in top[-5:]:
        print(f"  {names.get(q, q):22} {v:+.3f}")


if __name__ == "__main__":
    main()
