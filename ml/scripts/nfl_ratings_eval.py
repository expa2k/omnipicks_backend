"""
OmniPicks — NFL: ¿qué tan bien predicen los ratings de equipo el margen de cada partido?

Camina semana por semana (solo con lo jugado antes de cada semana) y compara:
  · relleno   : lo que usa hoy la app (local por 2.5 siempre)
  · localía   : solo la ventaja de local media
  · Elo       : Elo clásico con margen de victoria (referencia pública habitual)
  · ratings   : ataque/defensa por EPA ajustados por rival (nfl_ratings)
  · mercado   : la línea de cierre del casino (lo que hay que igualar; no entra al modelo)

La conversión de EPA a puntos (margen = a + b·x) se ajusta solo con temporadas anteriores.

Uso:
    python ml/scripts/nfl_ratings_eval.py                 # parámetros por defecto
    python ml/scripts/nfl_ratings_eval.py --grid          # prueba varias vidas medias / pausas / ridge
"""
import argparse
import itertools
import logging
import math
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.nfl_history import load_pbp, load_schedules, normalize_teams, team_games
from app.services.nfl_ratings import NFLRatingEngine, PointsRatingEngine, RatingParams, expected_margin_epa

logging.basicConfig(level=logging.WARNING)
SIGMA = 13.5
EVAL_FROM = 2015


def games_table() -> pl.DataFrame:
    s = normalize_teams(load_schedules(), cols=("home_team", "away_team"))
    return s.filter((pl.col("season") >= 2010) & pl.col("result").is_not_null()).sort(["season", "week", "gameday"])


def elo_predictions(games: pl.DataFrame, k: float = 20.0, hfa: float = 48.0, revert: float = 1 / 3) -> np.ndarray:
    """Margen previsto por un Elo clásico (25 puntos de Elo ≈ 1 punto de margen)."""
    elo, last_season, out = {}, None, []
    for r in games.iter_rows(named=True):
        if r["season"] != last_season:
            elo = {t: 1505 + (1 - revert) * (v - 1505) for t, v in elo.items()}
            last_season = r["season"]
        h, a = elo.get(r["home_team"], 1505.0), elo.get(r["away_team"], 1505.0)
        neutral = r["location"] == "Neutral"
        diff = h - a + (0 if neutral else hfa)
        out.append(diff / 25.0)
        exp = 1 / (1 + 10 ** (-diff / 400))
        res = r["result"]
        actual = 1.0 if res > 0 else 0.0 if res < 0 else 0.5
        mult = math.log(abs(res) + 1) * 2.2 / ((abs(diff) * 0.001) + 2.2)
        delta = k * mult * (actual - exp)
        elo[r["home_team"]], elo[r["away_team"]] = h + delta, a - delta
    return np.array(out)


def rating_features(games: pl.DataFrame, tg: pl.DataFrame, params: RatingParams) -> np.ndarray:
    neutral = games.filter(pl.col("location") == "Neutral")["game_id"].to_list()
    eng = NFLRatingEngine(tg, params, neutral_games=neutral)
    cache, x = {}, []
    for r in games.iter_rows(named=True):
        key = (r["season"], r["week"])
        if key not in cache:
            cache[key] = eng.at(*key)
        x.append(expected_margin_epa(cache[key], r["home_team"], r["away_team"], r["location"] == "Neutral")[0])
    return np.array(x)


def split_features(games: pl.DataFrame, tg: pl.DataFrame, params: RatingParams) -> np.ndarray:
    """Dos columnas: diferencia de EPA esperada en pase y en carrera (local − visitante), por jugada × 35."""
    neutral = games.filter(pl.col("location") == "Neutral")["game_id"].to_list()
    eng = NFLRatingEngine(tg, params, neutral_games=neutral)
    cache, x = {}, []
    for r in games.iter_rows(named=True):
        key = (r["season"], r["week"])
        if key not in cache:
            cache[key] = eng.at(*key)
        w = cache[key]
        h, a = r["home_team"], r["away_team"]
        row = []
        for k in ("pass", "rush"):
            eh = w.off[k].get(h, 0.0) - w.deff[k].get(a, 0.0)
            ea = w.off[k].get(a, 0.0) - w.deff[k].get(h, 0.0)
            row.append((eh - ea) * 35.0)
        x.append(row)
    return np.array(x)


def points_features(games: pl.DataFrame, params: RatingParams, ridge: float = 2.0) -> np.ndarray:
    eng = PointsRatingEngine(games, params)
    cache, x = {}, []
    for r in games.iter_rows(named=True):
        key = (r["season"], r["week"])
        if key not in cache:
            cache[key] = eng.at(*key, ridge=ridge)
        hfa, st = cache[key]
        neutral = r["location"] == "Neutral"
        x.append((0.0 if neutral else hfa) + st.get(r["home_team"], 0.0) - st.get(r["away_team"], 0.0))
    return np.array(x)


def walk_forward(x: np.ndarray, y: np.ndarray, season: np.ndarray) -> np.ndarray:
    """margen = a + b·x (x puede tener varias columnas), ajustado con las temporadas anteriores
    (desde 2011: 2010 calienta los ratings)."""
    X = x.reshape(len(y), -1)
    pred = np.full(len(y), np.nan)
    for s in np.unique(season):
        if s < EVAL_FROM:
            continue
        tr = (season < s) & (season >= 2011)
        A = np.c_[np.ones(tr.sum()), X[tr]]
        coef = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        te = season == s
        pred[te] = np.c_[np.ones(te.sum()), X[te]] @ coef
    return pred


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    from scipy.stats import norm
    ok = ~np.isnan(pred)
    p, yy = pred[ok], y[ok]
    prob = norm.cdf(p / SIGMA)
    dec = yy != 0
    return {
        "MAE": float(np.mean(np.abs(p - yy))),
        "RMSE": float(np.sqrt(np.mean((p - yy) ** 2))),
        "Brier": float(np.mean((prob[dec] - (yy[dec] > 0)) ** 2)),
        "Acierto": float(np.mean((p[dec] > 0) == (yy[dec] > 0))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()

    games = games_table()
    tg = team_games(load_pbp(range(2010, 2027)))
    y = games["result"].to_numpy().astype(float)
    season = games["season"].to_numpy()
    ev = season >= EVAL_FROM
    market = games["spread_line"].to_numpy().astype(float)

    rows = {
        "Relleno actual (local +2.5)": np.where(games["location"].to_numpy() == "Neutral", 0.0, 2.5),
        "Solo localía": walk_forward(np.zeros(len(y)), y, season),
        "Elo clásico": walk_forward(elo_predictions(games), y, season),
    }

    if args.grid:
        print("Búsqueda de parámetros (2015–2026):")
        best = None
        for hl, gap, lam in itertools.product((6, 10, 16), (4, 12, 24), (300, 600, 1500)):
            prm = RatingParams(half_life=hl, offseason_gap=gap, ridge=lam)
            pr = walk_forward(rating_features(games, tg, prm), y, season)
            m = metrics(np.where(ev, pr, np.nan), y)
            print(f"  vida media {hl:>2} sem · pausa {gap:>2} · ridge {lam:>4} → MAE {m['MAE']:.3f}  Brier {m['Brier']:.4f}")
            if best is None or m["MAE"] < best[0]:
                best = (m["MAE"], prm)
        params = best[1]
        print(f"\nMejores: {params}\n")
    else:
        params = RatingParams(half_life=16, offseason_gap=12, ridge=300)

    x_epa = rating_features(games, tg, params)
    rows["Ratings EPA ajustados por rival"] = walk_forward(x_epa, y, season)
    best_pts = None
    for hl, gap, lam in itertools.product((6, 10, 16), (4, 12, 24), (1.0, 3.0, 8.0)):
        xp = points_features(games, RatingParams(half_life=hl, offseason_gap=gap), ridge=lam)
        m = metrics(np.where(ev, walk_forward(xp, y, season), np.nan), y)["MAE"]
        if best_pts is None or m < best_pts[0]:
            best_pts = (m, hl, gap, lam, xp)
    print(f"Rating por marcador: vida media {best_pts[1]} · pausa {best_pts[2]} · ridge {best_pts[3]}")
    x_pts = best_pts[4]
    rows["Rating por marcador"] = walk_forward(x_pts, y, season)
    rows["EPA + marcador"] = walk_forward(np.c_[x_epa, x_pts], y, season)
    x_split = split_features(games, tg, params)
    rows["EPA pase/carrera + marcador"] = walk_forward(np.c_[x_split, x_pts], y, season)
    rows["Mercado (línea de cierre)"] = market

    print(f"Partidos evaluados: {int(ev.sum()):,} (temporadas {EVAL_FROM}–{int(season.max())})\n")
    print(f"{'':34} {'MAE':>6} {'RMSE':>6} {'Brier':>7} {'Acierto':>8}")
    for name, pred in rows.items():
        m = metrics(np.where(ev, pred, np.nan), y)
        print(f"{name:34} {m['MAE']:6.2f} {m['RMSE']:6.2f} {m['Brier']:7.4f} {m['Acierto']:8.1%}")

    # Por temporada: ratings vs mercado
    pr = rows["EPA pase/carrera + marcador"]
    print(f"\n{'Temporada':>9} {'Ratings':>8} {'Mercado':>8} {'Elo':>6}")
    for s in range(EVAL_FROM, int(season.max()) + 1):
        m = season == s
        print(f"{s:>9} {np.mean(np.abs(pr[m] - y[m])):8.2f} {np.mean(np.abs(market[m] - y[m])):8.2f} "
              f"{np.mean(np.abs(rows['Elo clásico'][m] - y[m])):6.2f}")
    corr = np.corrcoef(pr[ev], market[ev])[0, 1]
    print(f"\nCorrelación ratings–mercado: {corr:.3f}")


if __name__ == "__main__":
    main()
