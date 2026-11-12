"""
OmniPicks — NFL: ¿cuánto agrega el contexto del partido (descanso, viaje, husos, división)?

Parte del modelo de la fase 2 (EPA + marcador + QB titular) y le suma las variables de
nfl_context. También prueba dar más peso a las temporadas recientes al ajustar la conversión a
puntos (la ventaja de local fue bajando con los años).

Mismo esquema que nfl_qb_eval.py: se elige con 2015–2020 y se reporta 2021–2026 aparte.

Uso:
    python ml/scripts/nfl_context_eval.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.nfl_context import CONTEXT_FEATURES, context
from app.services.nfl_history import load_pbp, team_games
from app.services.nfl_qb import QBParams, qb_games
from app.services.nfl_ratings import RatingParams
from nfl_qb_eval import HOLDOUT, TUNE, qb_features, report
from nfl_ratings_eval import EVAL_FROM, games_table, metrics, points_features, rating_features


def context_matrix(games) -> np.ndarray:
    rows = []
    for r in games.iter_rows(named=True):
        c = context(r["home_team"], r["away_team"], r["season"], neutral=r["location"] == "Neutral",
                    home_rest=r["home_rest"], away_rest=r["away_rest"], gametime_et=r["gametime"],
                    div_game=bool(r["div_game"]), playoff=r["game_type"] != "REG")
        rows.append([c[f] for f in CONTEXT_FEATURES])
    return np.array(rows)


def walk_forward_w(X: np.ndarray, y: np.ndarray, season: np.ndarray, season_half_life: float = None,
                   ridge: float = 0.0) -> np.ndarray:
    """Como walk_forward, con peso por antigüedad de la temporada y un ridge chico opcional
    (sin intercepto: la localía va en home_field)."""
    pred = np.full(len(y), np.nan)
    for s in np.unique(season):
        if s < EVAL_FROM:
            continue
        tr = (season < s) & (season >= 2011)
        w = np.ones(tr.sum()) if not season_half_life else 0.5 ** ((s - season[tr]) / season_half_life)
        A = np.c_[np.ones(tr.sum()), X[tr]]
        Aw = A * w[:, None]
        reg = ridge * np.eye(A.shape[1])
        reg[0, 0] = 0.0
        coef = np.linalg.solve(Aw.T @ A + reg + 1e-6 * np.eye(A.shape[1]), Aw.T @ y[tr])   # 1e-6: columnas vacías (sin 2020 aún)
        te = season == s
        pred[te] = np.c_[np.ones(te.sum()), X[te]] @ coef
    return pred


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
    xq = qb_features(games, qbg, QBParams())
    base = np.c_[x_epa, x_pts, xq]
    ctx = context_matrix(games)

    print("Peso de las temporadas recientes al convertir a puntos (con contexto):")
    best = None
    for hl in (None, 8, 4, 2):
        for lam in (0.0, 50.0, 200.0):
            pred = walk_forward_w(np.c_[base, ctx], y, season, hl, lam)
            m = metrics(np.where(tune, pred, np.nan), y)["MAE"]
            print(f"  vida media {hl or '∞':>2} temporadas · ridge {lam:>5} → MAE {m:.3f}")
            if best is None or m < best[0]:
                best = (m, hl, lam)
    _, hl, lam = best
    print(f"Mejor: vida media {hl or '∞'} · ridge {lam}")

    rows = {
        "Fase 2 (EPA + marcador + QB)": walk_forward_w(base, y, season),
        "Fase 2 con peso por temporada": walk_forward_w(base, y, season, hl, lam),
        "Fase 3: + contexto": walk_forward_w(np.c_[base, ctx], y, season, hl, lam),
        "Mercado (línea de cierre)": games["spread_line"].to_numpy().astype(float),
    }
    report("Temporadas 2015–2020 (con las que se eligió)", rows, y, tune)
    report("Temporadas 2021–2026 (no usadas para elegir)", rows, y, hold)

    # Coeficientes del contexto (ajuste con todo hasta 2025): cuántos puntos vale cada cosa
    tr = (season >= 2011) & (season <= 2025)
    X = np.c_[np.ones(tr.sum()), base[tr], ctx[tr]]
    w = np.ones(tr.sum()) if not hl else 0.5 ** ((2026 - season[tr]) / hl)
    reg = lam * np.eye(X.shape[1]); reg[0, 0] = 0.0
    coef = np.linalg.solve((X * w[:, None]).T @ X + reg, (X * w[:, None]).T @ y[tr])
    print("\nCuánto vale cada variable de contexto (puntos para el local):")
    for f, c in zip(CONTEXT_FEATURES, coef[1 + base.shape[1]:]):
        print(f"  {f:14} {c:+.2f}")
    print(f"  (intercepto {coef[0]:+.2f})")


if __name__ == "__main__":
    main()
