"""
OmniPicks — NBA v2 (N2): ¿cuánto mejora saber quién falta y repartir bien sus minutos?

Camina hacia adelante con los ratings de N1 y compara supuestos sobre la plantilla:
  · sin_info   : minutos habituales, sin saber quién falta hoy
  · con_bajas  : sabe lo que diría un reporte de lesiones (titulares y rotación que no
                 juegan, y quién regresa); NO sabe qué suplentes se quedan sin jugar
  · sabe_quien : sabe exactamente quién jugó (techo de la información de disponibilidad)
  · minutos    : minutos reales (techo absoluto)

En el histórico no hay reportes de lesiones, así que "baja" = jugador de rotación
(≥15 min, presente en ≥60% de los últimos partidos) que no jugó.

Uso:
    python ml/scripts/nba_minutes_eval.py --from 2023-10-01 --to 2024-07-01
"""
import argparse
import logging
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

from sqlalchemy import create_engine

from app.config import get_settings
from app.services.nba_lineups import ROTATION_MIN, RosterTracker, expected_minutes, team_net
from app.services.nba_ratings import Prepared, RatingConfig, RatingEngine, load_data
from nba_ratings_eval import metrics, pace_by_day, team_logs

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("nba_minutes_eval")

SETTINGS = ("sin_info", "con_bajas", "con_bajas_aprendido", "sabe_quien", "minutos")
# 3d: corrección por lo observado sin el jugador, con distintos pesos del modelo (k partidos)
WITHOUT_K = (5.0, 15.0, 40.0)
EXTRA = tuple(f"sin_jugador_k{k:.0f}" for k in WITHOUT_K) + ("equipo_k20",)


def evaluate(prep: Prepared, cfg: RatingConfig, first: date):
    actual, _ = team_logs(prep)
    pace = pace_by_day(prep)
    day_of = {d: i for i, d in enumerate(prep.days)}
    games_by_day = {d: g for d, g in prep.games.groupby("game_date")}
    tracker = RosterTracker()
    fed = [0]                                           # días ya entregados al tracker
    rows, min_rows = [], []
    wo_sum, wo_n = {}, {}                               # (equipo, jugador) → residuo acumulado sin él
    tm_sum, tm_n = {}, {}                               # equipo → residuo reciente (con descuento)
    pending = []                                        # residuos del día, se aplican al cerrar el día

    def feed_until(i):
        while fed[0] < i:
            d = prep.days[fed[0]]
            for g in games_by_day[d].itertuples():
                for tid in (int(g.home_team_id), int(g.away_team_id)):
                    if (g.game_id, tid) in actual:
                        tracker.update(d, tid, *actual[(g.game_id, tid)])
            fed[0] += 1
        for tid, outs, resid in pending:
            for p in outs:
                wo_sum[(tid, p)] = wo_sum.get((tid, p), 0.0) + resid
                wo_n[(tid, p)] = wo_n.get((tid, p), 0) + 1
            tm_sum[tid] = 0.97 * tm_sum.get(tid, 0.0) + resid
            tm_n[tid] = 0.97 * tm_n.get(tid, 0.0) + 1.0
        pending.clear()

    def on_day(r):
        i = day_of[r.day]
        feed_until(i)
        total = r.o + r.d
        for g in games_by_day[r.day].itertuples():
            rec = {"game_id": g.game_id, "game_date": g.game_date, "season": g.season,
                   "margin": g.home_pts - g.away_pts, "outs": 0, "miss_min": 0.0, "miss_off": 0.0}
            nets, out_sets = {}, {}
            for side, tid in (("h", int(g.home_team_id)), ("a", int(g.away_team_id))):
                cands = tracker.candidates(tid, r.day)
                if (g.game_id, tid) not in actual or len(cands) < 8:
                    nets = None
                    break
                act_p, act_min = actual[(g.game_id, tid)]
                act = dict(zip(act_p.tolist(), act_min.tolist()))
                ids = np.array([c.player for c in cands])
                rating = total[ids]
                outs = {c.player: 0.0 for c in cands if c.player not in act and c.rate >= 0.6 and c.base >= ROTATION_MIN}
                back = {c.player for c in cands if c.player in act and c.rate <= 0.2 and c.base >= ROTATION_MIN}
                absent = {c.player: 0.0 for c in cands if c.player not in act}
                present = {c.player: 1.0 for c in cands if c.player in act}
                proj = {
                    "sin_info": expected_minutes(cands)[0],
                    "con_bajas": expected_minutes(cands, outs, back)[0],
                    "con_bajas_aprendido": expected_minutes(cands, outs, back, tracker.learned_shift(tid, r.day, list(outs)))[0],
                    "sabe_quien": expected_minutes(cands, {**absent, **present})[0],
                }
                real = np.array([act.get(c.player, 0.0) for c in cands])
                unknown = [p for p in act if p not in set(ids.tolist())]      # debutantes o recién llegados
                real_net = team_net(real, rating) + float(sum(act[p] * total[p] for p in unknown)) / 48.0
                real_net *= 240.0 / max(sum(act.values()), 1.0)
                nets[side] = {**{k: team_net(v, rating) for k, v in proj.items()}, "minutos": real_net}
                rec["outs"] += len(outs)
                out_sets[side] = (tid, list(outs))
                sgn = 1.0 if side == "h" else -1.0
                for c in cands:
                    if c.player in outs:                # minutos y valor ofensivo que faltan (local − visita)
                        rec["miss_min"] += sgn * c.base
                        rec["miss_off"] += sgn * c.base * r.o[c.player] / 48.0
                for k, v in proj.items():
                    err = np.abs(v - real)
                    mask = (v > 0) | (real > 0)
                    min_rows.append({"setting": k, "outs": len(outs), "abs_err": float(err[mask].sum()) + sum(act[p] for p in unknown),
                                     "n": int(mask.sum()) + len(unknown)})
            if nets is None:
                continue
            for k in SETTINGS:
                rec[k] = pace[i] * (nets["h"][k] - nets["a"][k] + r.home_edge)
            for k in WITHOUT_K:
                adj = 0.0
                for sign, side in ((1.0, "h"), (-1.0, "a")):
                    tid, outs = out_sets[side]
                    adj += sign * sum(wo_sum.get((tid, p), 0.0) / (wo_n.get((tid, p), 0) + k) for p in outs)
                rec[f"sin_jugador_k{k:.0f}"] = rec["con_bajas"] + adj
            th, ta = out_sets["h"][0], out_sets["a"][0]
            rec["equipo_k20"] = rec["con_bajas"] + tm_sum.get(th, 0.0) / (tm_n.get(th, 0.0) + 20.0) \
                - tm_sum.get(ta, 0.0) / (tm_n.get(ta, 0.0) + 20.0)
            resid = rec["margin"] - rec["con_bajas"]
            pending.append((th, out_sets["h"][1], resid))
            pending.append((ta, out_sets["a"][1], -resid))
            rows.append(rec)

    RatingEngine(prep, cfg).run(first_output=first, on_day=on_day)
    return pd.DataFrame(rows), pd.DataFrame(min_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="first", default="2023-10-01")
    ap.add_argument("--to", dest="last", default=None)
    args = ap.parse_args()
    last = date.fromisoformat(args.last) if args.last else None
    cfg = RatingConfig()
    prep = Prepared(load_data(create_engine(get_settings().mysql_url_sync), until=last), cfg)
    df, mins = evaluate(prep, cfg, date.fromisoformat(args.first))
    df = df[df.game_date <= prep.days[prep.last_stint_day]]
    fmt = lambda v: f"{v:.4f}"

    def table(d, title):
        out = pd.DataFrame({k: metrics(d[k].to_numpy(float), d["margin"].to_numpy(float)) for k in SETTINGS + EXTRA}).T
        print(f"\n── {title} ──\n{out.to_string(float_format=fmt)}")

    print(f"Evaluación {df.game_date.min()} → {df.game_date.max()}")
    table(df, "Margen: todos los partidos")
    table(df[df.outs > 0], "Margen: partidos con al menos una baja de rotación")
    table(df[df.outs >= 3], "Margen: partidos con 3 o más bajas de rotación")
    table(df[df.outs == 0], "Margen: partidos sin bajas")
    # 3c: si al faltar ataque los compañeros rindieran peor de lo que supone el modelo, el
    # error de con_bajas tendría pendiente negativa contra lo que falta
    print("\n── 3c: ¿el error depende de lo que falta? (pendiente ± error estándar) ──")
    resid = (df["margin"] - df["con_bajas"]).to_numpy(float)
    for col, label in (("miss_min", "minutos de rotación ausentes (por cada 10)"), ("miss_off", "valor ofensivo ausente (por pt/100)")):
        x = df[col].to_numpy(float) / (10.0 if col == "miss_min" else 1.0)
        b = float((x * resid).sum() / (x * x).sum())
        se = float(np.sqrt(((resid - b * x) ** 2).sum() / (len(x) - 1) / (x * x).sum()))
        print(f"{label}: {b:+.3f} ± {se:.3f}")
    print("\n── Error medio de minutos por jugador ──")
    for label, m in (("todos", mins), ("equipos con bajas", mins[mins.outs > 0]), ("equipos con 2+ bajas", mins[mins.outs >= 2])):
        g = m.groupby("setting")[["abs_err", "n"]].sum()
        print(label, (g.abs_err / g.n).reindex(SETTINGS[:4]).round(3).to_dict())


if __name__ == "__main__":
    main()
