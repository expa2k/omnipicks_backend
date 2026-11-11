"""
OmniPicks — NBA v2 (N1): ¿los ratings de jugador predicen mejor que los de equipo?

Predice el margen de cada partido caminando hacia adelante (solo con información previa)
y compara contra dos referencias de equipo: Elo con margen de victoria y un ridge de
márgenes con decaimiento.

Como la proyección de minutos es trabajo de N2, aquí se prueban tres supuestos:
  · plantilla : minutos esperados de quienes jugaron en los últimos 5 partidos del equipo
                (no sabe quién falta hoy; comparación justa contra los ratings de equipo)
  · sabe_bajas: igual, pero solo entre quienes de verdad jugaron (techo de lo que puede dar N2)
  · minutos   : con los minutos reales del partido (techo absoluto, no alcanzable)

Uso:
    python ml/scripts/nba_ratings_eval.py --from 2023-10-01 --to 2024-07-01
    python ml/scripts/nba_ratings_eval.py --from 2023-10-01 --to 2024-07-01 --grid
    python ml/scripts/nba_ratings_eval.py --from 2023-10-01 --ablation
"""
import argparse
import logging
import os
import sys
import time
from dataclasses import replace
from datetime import date
from math import erf, sqrt

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy import create_engine

from app.config import get_settings
from app.services.nba_ratings import Prepared, RatingConfig, RatingEngine, load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("nba_ratings_eval")

SIGMA_MARGIN = 13.0          # para convertir margen en probabilidad
KEY_MPG = 28.0               # "baja importante": faltó alguien que promedia al menos estos minutos


# ══════════════════════════════════════════════════════════════════════
# Minutos por partido (reales y de los 5 partidos previos del equipo)
# ══════════════════════════════════════════════════════════════════════

def team_logs(prep: Prepared):
    """{(game_id, team_id): (jugadores, minutos)} y {(game_id, team_id): [logs de los 5 partidos previos]}."""
    box = prep.box
    actual = {}
    for (gid, tid), g in box.groupby(["game_id", "team_id"], sort=False):
        actual[(gid, int(tid))] = (g["player_id"].map(prep.pidx).to_numpy(), g["minutes"].to_numpy(float))
    previous, history = {}, {}
    for g in prep.games.itertuples():
        for tid in (int(g.home_team_id), int(g.away_team_id)):
            h = history.setdefault(tid, [])
            previous[(g.game_id, tid)] = [x for x in h[-5:] if (g.game_date - x[0]).days <= 200]
            if (g.game_id, tid) in actual:
                h.append((g.game_date,) + actual[(g.game_id, tid)])
    return actual, previous


def net_rating(players, weights, total) -> float:
    """Rating neto del equipo: Σ minutos × rating / 48, con los minutos escalados a 240."""
    s = weights.sum()
    return float((weights * total[players]).sum() / s * 5.0) if s > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════
# Referencias de equipo
# ══════════════════════════════════════════════════════════════════════

def elo_predictions(games: pd.DataFrame, k=20.0, hca=100.0, carry=0.75) -> np.ndarray:
    elo, season, pred = {}, None, []
    for g in games.itertuples():
        if g.season != season:
            season = g.season
            elo = {t: 1505 + carry * (e - 1505) for t, e in elo.items()}
        h, a = elo.get(g.home_team_id, 1500.0), elo.get(g.away_team_id, 1500.0)
        diff = h + hca - a
        pred.append(diff / 28.0)
        margin = g.home_pts - g.away_pts
        exp = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        win_diff = diff if margin > 0 else -diff
        mult = (abs(margin) + 3.0) ** 0.8 / (7.5 + 0.006 * win_diff)
        delta = k * mult * ((1.0 if margin > 0 else 0.0) - exp)
        elo[g.home_team_id], elo[g.away_team_id] = h + delta, a - delta
    return np.array(pred)


def team_ridge_predictions(prep: Prepared, half_life=60.0, ridge=1.0) -> np.ndarray:
    """Fuerza por equipo con ridge sobre márgenes, recalculada cada día con decaimiento."""
    games = prep.games
    teams = {t: i for i, t in enumerate(sorted(set(games.home_team_id) | set(games.away_team_id)))}
    T = len(teams)
    hi = games.home_team_id.map(teams).to_numpy()
    ai = games.away_team_id.map(teams).to_numpy()
    margin = (games.home_pts - games.away_pts).to_numpy(float)
    day = games.day.to_numpy()
    pred = np.zeros(len(games))
    A, b = np.zeros((T + 1, T + 1)), np.zeros(T + 1)
    pen = np.r_[np.full(T, ridge), 1e-3]
    prev_eff = 0.0
    for i in range(len(prep.days)):
        g = 0.5 ** ((prep.eff[i] - prev_eff) / half_life)
        A *= g
        b *= g
        prev_eff = prep.eff[i]
        idx = np.flatnonzero(day == i)
        theta = np.linalg.solve(A + np.diag(pen), b)
        pred[idx] = theta[hi[idx]] - theta[ai[idx]] + theta[T]
        for j in idx:
            x = np.zeros(T + 1)
            x[hi[j]], x[ai[j]], x[T] = 1.0, -1.0, 1.0
            A += np.outer(x, x)
            b += x * margin[j]
    return pred


# ══════════════════════════════════════════════════════════════════════
# Predicción con ratings de jugador
# ══════════════════════════════════════════════════════════════════════

def pace_by_day(prep: Prepared) -> np.ndarray:
    """Posesiones por equipo y partido / 100 (en unidades de stint), solo con días anteriores."""
    poss_day = np.zeros(len(prep.days))
    np.add.at(poss_day, prep.st_day, prep.st_hposs + prep.st_aposs)
    cum_poss = np.concatenate([[0.0], np.cumsum(poss_day)[:-1]])
    cum_games = np.concatenate([[0.0], np.cumsum(prep.st_games_per_day)[:-1]])
    return np.where(cum_games > 50, cum_poss / np.maximum(cum_games, 1.0) / 200.0, 0.96)


def player_predictions(prep: Prepared, cfg: RatingConfig, first: date, logs=None):
    actual, previous = logs or team_logs(prep)
    engine = RatingEngine(prep, cfg)
    games_by_day = {d: g for d, g in prep.games.groupby("game_date")}
    rows = []

    PACE = pace_by_day(prep)
    day_of = {d: i for i, d in enumerate(prep.days)}

    def on_day(r):
        i = day_of[r.day]
        pace = PACE[i]
        total = r.o + r.d
        for g in games_by_day[r.day].itertuples():
            rec = {"game_id": g.game_id, "game_date": g.game_date, "season": g.season, "season_type": g.season_type,
                   "margin": g.home_pts - g.away_pts}
            nets, missing = {}, False
            for side, tid in (("h", int(g.home_team_id)), ("a", int(g.away_team_id))):
                act_p, act_min = actual.get((g.game_id, tid), (np.array([], int), np.array([])))
                prev = previous.get((g.game_id, tid), [])
                if not prev or not len(act_p):
                    nets = None
                    break
                exp = {}
                for _, pl, mn in prev:
                    for p in pl:
                        exp[p] = exp.get(p, 0.0) + 1.0 / len(prev)
                cand = np.array(list(exp), dtype=int)
                w_roster = np.array([exp[p] for p in cand]) * r.mpg[cand]
                played = np.isin(cand, act_p)
                missing |= bool(((r.mpg[cand] >= KEY_MPG) & (np.array([exp[p] for p in cand]) >= 0.6) & ~played).any())
                nets[side] = (net_rating(cand, w_roster, total),
                              net_rating(act_p, r.mpg[act_p], total),
                              net_rating(act_p, act_min, total))
            if nets is None:
                continue
            for j, name in enumerate(("plantilla", "sabe_bajas", "minutos")):
                rec[name] = pace * (nets["h"][j] - nets["a"][j] + r.home_edge)
            rec["key_out"] = missing
            rows.append(rec)

    engine.run(first_output=first, on_day=on_day)
    return pd.DataFrame(rows), engine


# ══════════════════════════════════════════════════════════════════════
# Métricas
# ══════════════════════════════════════════════════════════════════════

def metrics(pred: np.ndarray, margin: np.ndarray) -> dict:
    p = np.clip(np.array([0.5 * (1 + erf(x / (SIGMA_MARGIN * sqrt(2)))) for x in pred]), 1e-6, 1 - 1e-6)
    win = (margin > 0).astype(float)
    slope = float(np.polyfit(pred, margin, 1)[0]) if np.std(pred) > 0 else float("nan")
    return {
        "n": len(pred),
        "MAE": float(np.abs(pred - margin).mean()),
        "RMSE": float(np.sqrt(((pred - margin) ** 2).mean())),
        "acierto": float(((pred > 0) == (margin > 0)).mean()),
        "logloss": float(-(win * np.log(p) + (1 - win) * np.log(1 - p)).mean()),
        "Brier": float(((p - win) ** 2).mean()),
        "pendiente": slope,
    }


def report(df: pd.DataFrame, cols, title: str) -> pd.DataFrame:
    out = pd.DataFrame({c: metrics(df[c].to_numpy(float), df["margin"].to_numpy(float)) for c in cols}).T
    print(f"\n── {title} ──")
    print(out.to_string(float_format=lambda v: f"{v:.4f}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="first", default="2023-10-01")
    ap.add_argument("--to", dest="last", default=None)
    ap.add_argument("--grid", action="store_true", help="probar varias combinaciones de hiperparámetros")
    ap.add_argument("--ablation", action="store_true", help="quitar una pieza a la vez")
    args = ap.parse_args()
    first = date.fromisoformat(args.first)
    last = date.fromisoformat(args.last) if args.last else None

    engine = create_engine(get_settings().mysql_url_sync)
    data = load_data(engine, until=last)
    base = RatingConfig()
    prep = Prepared(data, base)
    logger.info(f"{len(prep.games)} partidos, {prep.P} jugadores, stints de {prep.games_with_stints} partidos "
                f"({prep.games_bad_stints} descartados por no cuadrar con el marcador)")
    last_stint = prep.days[prep.last_stint_day]
    logs = team_logs(prep)

    def run(cfg, label=""):
        t = time.time()
        df, eng = player_predictions(prep, cfg, first, logs)
        df = df[df.game_date <= last_stint]
        logger.info(f"{label} listo en {time.time() - t:.0f}s")
        return df, eng

    if args.grid or args.ablation:
        variants = {}
        if args.grid:
            for hl in (250.0, 400.0, 700.0):
                for lam in (2000.0, 4000.0, 8000.0):
                    variants[f"hl={hl:.0f} λ={lam:.0f}"] = replace(base, half_life_days=hl, ridge_player=lam)
            for bh in (75.0, 300.0):
                variants[f"box_hl={bh:.0f}"] = replace(base, box_half_life_days=bh)
            for bp in (200.0, 1000.0):
                variants[f"box_prior={bp:.0f}"] = replace(base, box_prior_poss=bp)
            for rb in (20.0, 2000.0):
                variants[f"ridge_box={rb:.0f}"] = replace(base, ridge_box=rb)
        if args.ablation:
            variants["completo"] = base
            variants["sin box score (solo RAPM)"] = replace(base, use_box=False)
            variants["sin RAPM (solo box score)"] = replace(base, use_rapm=False)
            variants["sin edad"] = replace(base, use_age=False)
        res = []
        for label, cfg in variants.items():
            df, eng = run(cfg, label)
            for col in ("plantilla", "sabe_bajas"):
                m = metrics(df[col].to_numpy(float), df["margin"].to_numpy(float))
                res.append({"variante": label, "supuesto": col, **m, "σ_tramo": eng.out_of_sample_sigma()})
        out = pd.DataFrame(res)
        print(out.sort_values(["supuesto", "MAE"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        return

    df, eng = run(base, "modelo base")
    games = prep.games
    games["elo"] = elo_predictions(games)
    games["ridge_equipo"] = team_ridge_predictions(prep)
    df = df.merge(games[["game_id", "elo", "ridge_equipo"]], on="game_id")
    cols = ["elo", "ridge_equipo", "plantilla", "sabe_bajas", "minutos"]
    print(f"\nEvaluación {df.game_date.min()} → {df.game_date.max()}")
    report(df, cols, "Todos los partidos")
    for season, g in df.groupby("season"):
        report(g, cols, f"Temporada {season}")
    report(df[df.key_out], cols, f"Partidos con baja importante (≥{KEY_MPG:.0f} min/partido)")
    report(df[~df.key_out], cols, "Partidos sin baja importante")
    print(f"\nσ por posesión fuera de muestra: {eng.out_of_sample_sigma():.1f} pts/100")
    print("\nPesos del box score (por desviación estándar de cada variable):")
    print(eng.box_weights().to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    last_r = None
    eng2 = RatingEngine(prep, base)
    last_r = eng2.run(extra_day=prep.days[-1])[-1]
    top = pd.DataFrame({"jugador": prep.player_names, "O": last_r.o, "D": last_r.d, "total": last_r.o + last_r.d,
                        "±": last_r.unc, "mpg": last_r.mpg, "min": last_r.minutes})
    top = top[top["min"] > 300].sort_values("total", ascending=False)
    print(f"\nMejores 15 al {prep.days[-1]}:")
    print(top.head(15).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nPeores 5:")
    print(top.tail(5).to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
