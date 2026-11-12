"""
OmniPicks — NFL v3, fase 5: todas las variables + regresión lineal contra LightGBM.

Variables (todas se saben antes del partido):
  · v2: EPA y marcador ajustados por rival, QB titular (ajuste y calidad)
  · métricas: EPA sin pérdidas, tasa de éxito, equipos especiales
  · contexto: localía, descanso, bye, semana corta, viaje, husos, temprano para el visitante, división,
    playoffs, horario estelar, altura (Denver), pasto natural, coach nuevo, semana de la temporada
  · total: anotación, EPA de los ataques, ritmo, techo, viento, frío, lluvia, QB, EPA sin pérdidas

Se camina temporada por temporada (cada una se predice con las anteriores) y se compara el error en
2015–2020 y 2021–2026 (prueba limpia).

Uso:
    python ml/scripts/nfl_gbm_eval.py
"""
import os
import sys
import warnings

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services import nfl_v2
from app.services.nfl_history import current_season, load_special_teams, team_games
from app.services.nfl_ratings import NFLRatingEngine, expected_margin_epa
from nfl_context_eval import context_matrix
from nfl_features_eval import st_snapshots


def extra_context(rows, sched) -> np.ndarray:
    coach = {}
    for r in sched.iter_rows(named=True):
        coach[(r["season"], r["home_team"])] = r["home_coach"]
        coach[(r["season"], r["away_team"])] = r["away_coach"]
    out = []
    for r in rows:
        neutral = r["location"] == "Neutral"
        try:
            hh = int((r["gametime"] or "13:00").split(":")[0])
        except ValueError:
            hh = 13
        prime = 1.0 if (r["weekday"] != "Sunday" or hh >= 19) else 0.0
        new_h = 1.0 if coach.get((r["season"] - 1, r["home_team"])) not in (None, r["home_coach"]) else 0.0
        new_a = 1.0 if coach.get((r["season"] - 1, r["away_team"])) not in (None, r["away_coach"]) else 0.0
        out.append([prime, 0.0 if neutral else float(r["home_team"] == "DEN"),
                    float((r["surface"] or "") == "grass"), new_h, new_a, float(r["week"])])
    return np.array(out)


def walk(X, y, season, model="ridge", first=2011):
    pred = np.full(len(y), np.nan)
    for s in np.unique(season):
        tr = (season >= first) & (season < s)
        if tr.sum() < 500:
            continue
        te = season == s
        if model == "ridge":
            c = nfl_v2._fit(X[tr], y[tr])
            pred[te] = np.c_[np.ones(te.sum()), X[te]] @ c
        else:
            import lightgbm as lgb
            m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=7, min_child_samples=60,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                  objective=model, verbose=-1)
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
    return pred


def main():
    sched, pbp = nfl_v2.load_data(False, refresh_schedules=False)
    feats = nfl_v2.Features(sched, pbp)
    played = sched.filter(pl.col("result").is_not_null())
    rows = list(played.iter_rows(named=True))
    M, T = [], []
    for r in rows:
        m, t, _ = feats.row(r)
        M.append([m[k] for k in nfl_v2.MARGIN_FEATURES]); T.append([t[k] for k in nfl_v2.TOTAL_FEATURES])
    M, T = np.array(M), np.array(T)
    neutral = sched.filter(pl.col("location") == "Neutral")["game_id"].to_list()
    tg = team_games(pbp)
    eng_sc = NFLRatingEngine(tg, nfl_v2.EPA_P, neutral_games=neutral, value="succ")
    eng_nt = NFLRatingEngine(tg, nfl_v2.EPA_P, neutral_games=neutral, value="epant")
    st = st_snapshots(load_special_teams(range(2010, current_season() + 1)), sched)
    cache, E = {}, []
    for r in rows:
        key = (r["season"], r["week"])
        if key not in cache:
            cache[key] = (eng_sc.at(*key), eng_nt.at(*key))
        ws, wn = cache[key]
        nz = r["location"] == "Neutral"
        sk = st.get(key, {})
        E.append([expected_margin_epa(ws, r["home_team"], r["away_team"], nz)[0],
                  expected_margin_epa(wn, r["home_team"], r["away_team"], nz)[0],
                  sk.get(r["home_team"], 0.0) - sk.get(r["away_team"], 0.0)])
    E = np.array(E)
    C = context_matrix(played)
    X2 = extra_context(rows, played)
    y = played["result"].to_numpy().astype(float)
    tot = played["total"].to_numpy().astype(float)
    season = played["season"].to_numpy()
    tune, hold = (season >= 2015) & (season <= 2020), season >= 2021
    mae = lambda p, yy, m: float(np.mean(np.abs(p[m] - yy[m])))

    full_m = np.c_[M, E, C, X2]
    full_t = np.c_[T, C[:, [0]], X2[:, [0, 1, 2, 5]]]
    print(f"{'':48} {'2015–20':>8} {'2021–26':>8}")
    for name, X, yy, mdl in (
        ("MARGEN v2 (lineal, 4 variables)", M, y, "ridge"),
        ("MARGEN todas las variables, lineal", full_m, y, "ridge"),
        ("MARGEN v2, LightGBM", M, y, "regression"),
        ("MARGEN todas, LightGBM (error cuadrático)", full_m, y, "regression"),
        ("MARGEN todas, LightGBM (error absoluto)", full_m, y, "regression_l1"),
        ("TOTAL v2 (lineal)", T, tot, "ridge"),
        ("TOTAL todas, lineal", full_t, tot, "ridge"),
        ("TOTAL todas, LightGBM", full_t, tot, "regression"),
        ("TOTAL todas, LightGBM (error absoluto)", full_t, tot, "regression_l1"),
    ):
        p = walk(X, yy, season, mdl)
        print(f"{name:48} {mae(p, yy, tune):8.3f} {mae(p, yy, hold):8.3f}")
    # Promedio de lineal y LightGBM (suele ser más estable que cada uno)
    pm = 0.5 * walk(M, y, season, "ridge") + 0.5 * walk(full_m, y, season, "regression")
    pt = 0.5 * walk(T, tot, season, "ridge") + 0.5 * walk(full_t, tot, season, "regression")
    print(f"{'MARGEN promedio v2 lineal + LightGBM todas':48} {mae(pm, y, tune):8.3f} {mae(pm, y, hold):8.3f}")
    print(f"{'TOTAL promedio v2 lineal + LightGBM todas':48} {mae(pt, tot, tune):8.3f} {mae(pt, tot, hold):8.3f}")
    sp, tl = played["spread_line"].to_numpy().astype(float), played["total_line"].to_numpy().astype(float)
    ok = ~np.isnan(tl)
    print(f"{'Mercado (margen / total)':48} {mae(sp, y, tune):8.3f} {mae(sp, y, hold):8.3f}  ·  "
          f"{mae(tl, tot, tune & ok):.3f} {mae(tl, tot, hold & ok):.3f}")


if __name__ == "__main__":
    main()
