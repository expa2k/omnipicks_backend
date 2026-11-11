"""
OmniPicks — NBA v2: proyección de minutos cuando hay bajas.

Compara tres formas de repartir los minutos de quien falta, caminando hacia adelante:
  · proporcional : todos los disponibles suben en proporción a lo que ya juegan (la actual)
  · reemplazos   : los minutos van a quienes de verdad entran por ese jugador en su equipo
                   (de las rotaciones), y de respaldo por posición y estatura
  · lgbm         : un LightGBM que predice los minutos de cada jugador con lo anterior como
                   variables; se entrena solo con temporadas anteriores a la que predice

Mide el error de minutos por jugador y el efecto en la predicción del margen.
Con --save entrena el LightGBM con todo y lo guarda para producción.

Uso:
    python ml/scripts/nba_minutes_model.py
    python ml/scripts/nba_minutes_model.py --save
"""
import argparse
import logging
import os
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

from sqlalchemy import create_engine, text

from app.config import get_settings
from app.services.nba_history import load_substitutions
from app.services.nba_lineups import (MINUTES_FEATURES, ROTATION_MIN, ReplacementContext, RosterTracker,
                                      expected_minutes, fill_minutes, minutes_features, MINUTES_MODEL_PATH)
from app.services.nba_ratings import Prepared, RatingConfig, RatingEngine, load_data
from nba_ratings_eval import metrics, pace_by_day, team_logs

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("nba_minutes_model")

FIRST_OUTPUT = date(2022, 12, 1)
REPORT_FROM = date(2023, 10, 1)


def player_info(engine, prep: Prepared):
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT player_id, position, height_in FROM nba_players"), conn)
    height, pos = {}, {}
    for r in df.itertuples():
        i = prep.pidx.get(int(r.player_id))
        if i is not None:
            if r.height_in == r.height_in and r.height_in:
                height[i] = float(r.height_in)
            if r.position:
                pos[i] = r.position
    return height, pos


def collect(prep: Prepared, cfg: RatingConfig, height, pos):
    actual, _ = team_logs(prep)
    pace = pace_by_day(prep)
    day_of = {d: i for i, d in enumerate(prep.days)}
    games_by_day = {d: g for d, g in prep.games.groupby("game_date")}
    subs = load_substitutions(prep.games.game_id.tolist())
    tracker = RosterTracker()
    last_team_game = {}
    fed = [0]
    sides, games = [], []

    def feed_until(i):
        while fed[0] < i:
            d = prep.days[fed[0]]
            for g in games_by_day[d].itertuples():
                for tid in (int(g.home_team_id), int(g.away_team_id)):
                    if (g.game_id, tid) in actual:
                        tracker.update(d, tid, *actual[(g.game_id, tid)])
                    pairs = [(prep.pidx[o], prep.pidx[n], w) for (t, o, n, w) in subs.get(g.game_id, [])
                             if t == tid and o in prep.pidx and n in prep.pidx]
                    tracker.add_substitutions(tid, pairs)
                    last_team_game[tid] = d
            fed[0] += 1

    def on_day(r):
        i = day_of[r.day]
        feed_until(i)
        total = r.o + r.d
        for g in games_by_day[r.day].itertuples():
            recs = {}
            for side, tid in (("h", int(g.home_team_id)), ("a", int(g.away_team_id))):
                cands = tracker.candidates(tid, r.day)
                if (g.game_id, tid) not in actual or len(cands) < 8:
                    recs = None
                    break
                act_p, act_min = actual[(g.game_id, tid)]
                act = dict(zip(act_p.tolist(), act_min.tolist()))
                ids = np.array([c.player for c in cands])
                outs = {c.player: 0.0 for c in cands if c.player not in act and c.rate >= 0.6 and c.base >= ROTATION_MIN}
                back = {c.player for c in cands if c.player in act and c.rate <= 0.2 and c.base >= ROTATION_MIN}
                ctx = ReplacementContext(tracker.replacements.get(tid, {}), height, pos)
                m_prop = expected_minutes(cands, outs, back)[0]
                m_rep = expected_minutes(cands, outs, back, ctx=ctx)[0]
                rest = (r.day - last_team_game[tid]).days if tid in last_team_game else 3
                X = minutes_features(tracker, cands, r.day, outs, back, m_prop, m_rep, total[ids], rest)
                unknown = [p for p in act if p not in set(ids.tolist())]
                recs[side] = {"game_id": g.game_id, "day": r.day, "season": g.season, "side": side, "ids": ids,
                              "total": total[ids], "m_prop": m_prop, "m_rep": m_rep, "X": X,
                              "out": np.array([c.player in outs for c in cands]),
                              "actual": np.array([act.get(p, 0.0) for p in ids.tolist()]),
                              "unknown_min": float(sum(act[p] for p in unknown)),
                              "unknown_net": float(sum(act[p] * total[p] for p in unknown)) / 48.0,
                              "n_out": len(outs)}
            if recs is None:
                continue
            sides.extend([recs["h"], recs["a"]])
            games.append({"game_id": g.game_id, "game_date": g.game_date, "season": g.season,
                          "margin": g.home_pts - g.away_pts, "pace": pace[i], "home_edge": r.home_edge,
                          "outs": recs["h"]["n_out"] + recs["a"]["n_out"]})

    RatingEngine(prep, cfg).run(first_output=FIRST_OUTPUT, on_day=on_day)
    return sides, pd.DataFrame(games)


def train_lgbm(sides):
    import lightgbm as lgb
    X = np.vstack([s["X"][~s["out"]] for s in sides])
    y = np.concatenate([s["actual"][~s["out"]] for s in sides])
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, verbose=-1)
    return model.fit(pd.DataFrame(X, columns=MINUTES_FEATURES), y)


def lgbm_minutes(model, s):
    pred = np.clip(model.predict(pd.DataFrame(s["X"], columns=MINUTES_FEATURES)), 0.0, 44.0)
    pred[s["out"]] = 0.0
    return fill_minutes(np.where(pred > 0.5, pred, 0.0), np.full(len(pred), 48.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    engine = create_engine(get_settings().mysql_url_sync)
    cfg = RatingConfig()
    prep = Prepared(load_data(engine), cfg)
    height, pos = player_info(engine, prep)
    sides, games = collect(prep, cfg, height, pos)
    logger.info(f"{len(games)} partidos, {len(sides)} equipos-partido en {time.time() - t0:.0f}s")

    # LightGBM caminando hacia adelante por temporada
    seasons = sorted({s["season"] for s in sides})
    for s in sides:
        s["m_lgbm"] = None
    for season in seasons:
        train = [s for s in sides if s["season"] < season]
        if len(train) < 1500:
            continue
        model = train_lgbm(train)
        for s in sides:
            if s["season"] == season:
                s["m_lgbm"] = lgbm_minutes(model, s)

    methods = ("m_prop", "m_rep", "m_lgbm")
    labels = {"m_prop": "proporcional (actual)", "m_rep": "reemplazos", "m_lgbm": "LightGBM"}
    evals = [s for s in sides if s["day"] >= REPORT_FROM and s["m_lgbm"] is not None]

    def minutes_error(subset):
        out = {}
        for m in methods:
            err = n = 0.0
            for s in subset:
                pred = s[m]
                mask = (pred > 0) | (s["actual"] > 0)
                err += np.abs(pred - s["actual"])[mask].sum() + s["unknown_min"]
                n += mask.sum() + (1 if s["unknown_min"] else 0)
            out[labels[m]] = err / max(n, 1)
        return out

    print("\n── Error medio de minutos por jugador ──")
    rows = {"todos": minutes_error(evals),
            "equipos con 1+ baja": minutes_error([s for s in evals if s["n_out"] >= 1]),
            "equipos con 2+ bajas": minutes_error([s for s in evals if s["n_out"] >= 2]),
            "solo jugadores que heredan minutos": None}
    # Error solo en quienes más cambian: compañeros de un ausente
    sub = {}
    for m in methods:
        err = n = 0.0
        for s in evals:
            if s["n_out"] == 0:
                continue
            mask = (~s["out"]) & ((s[m] > 0) | (s["actual"] > 0))
            err += np.abs(s[m] - s["actual"])[mask].sum()
            n += mask.sum()
        sub[labels[m]] = err / max(n, 1)
    rows["solo jugadores que heredan minutos"] = sub
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.3f}"))

    # Efecto en el margen
    by_game = {}
    for s in evals:
        by_game.setdefault(s["game_id"], {})[s["side"]] = s
    g = games.set_index("game_id")
    res = []
    for gid, d in by_game.items():
        if len(d) != 2:
            continue
        row = g.loc[gid]
        rec = {"game_id": gid, "margin": row.margin, "outs": row.outs}
        for m in methods:
            net = {k: (d[k][m] * d[k]["total"]).sum() / 48.0 for k in ("h", "a")}
            rec[labels[m]] = row.pace * (net["h"] - net["a"] + row.home_edge)
        res.append(rec)
    res = pd.DataFrame(res)
    for title, sub_df in (("todos", res), ("con bajas", res[res.outs > 0]), ("3+ bajas", res[res.outs >= 3])):
        t = pd.DataFrame({labels[m]: metrics(sub_df[labels[m]].to_numpy(float), sub_df.margin.to_numpy(float)) for m in methods}).T
        print(f"\n── Margen ({title}) ──\n{t.to_string(float_format=lambda v: f'{v:.4f}')}")

    if args.save:
        model = train_lgbm(sides)
        os.makedirs(os.path.dirname(MINUTES_MODEL_PATH), exist_ok=True)
        model.booster_.save_model(MINUTES_MODEL_PATH)
        imp = pd.Series(model.booster_.feature_importance("gain"), index=MINUTES_FEATURES).sort_values(ascending=False)
        print("\nImportancia (ganancia):\n" + (imp / imp.sum()).round(3).to_string())
        logger.info(f"Modelo de minutos guardado en {MINUTES_MODEL_PATH}")


if __name__ == "__main__":
    main()
