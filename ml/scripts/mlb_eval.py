"""
OmniPicks — MLB v2, fases 1 a 3: backtest del modelo de carreras (2015–2025).

Cada 14 días se ajusta con lo jugado antes y se predicen los juegos de las siguientes dos semanas,
con la alineación y los abridores de cada juego (en producción se conocen antes del juego).

Compara:
  · base: el local gana el % histórico;
  · solo equipos: ataque y defensa por equipo y temporada (sin jugadores);
  · v2 sin contexto: bateadores + abridor + bullpen + parque;
  · v2 completo: + ventaja de mano y cansancio del bullpen.

Uso:
    python ml/scripts/mlb_eval.py
"""
import os
import sys
import time

from dataclasses import replace

import numpy as np
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.mlb_history import load_gamelogs, load_handedness
from app.services.mlb_model import MLBParams, fit, game_probs, team_rows

FIRST, LAST = 2015, 2025
STEP = 14


def walk(rows: pl.DataFrame, params: MLBParams, skills: bool = False):
    rows = rows.sort("date")
    dates = rows["date"].to_numpy().astype("datetime64[D]")
    season = rows["season"].to_numpy()
    out = np.full((rows.height, 1), np.nan)
    f = None
    for s in range(FIRST, LAST + 1):
        sm = season == s
        if not sm.any():
            continue
        start, end = dates[sm].min(), dates[sm].max()
        t = start
        while t <= end:
            te = np.where(sm & (dates >= t) & (dates < t + np.timedelta64(STEP, "D")))[0]
            if len(te):
                cur = rows
                if skills:
                    from app.services.mlb_model import SKILLS, add_skills, skill_params
                    sf = {k: fit(rows, t, replace(skill_params(params), target=k)) for k in SKILLS}
                    cur = add_skills(rows, sf)
                f = fit(cur, t, params, init=f)
                for i in te:
                    out[i, 0] = f.log_rate(cur.row(int(i), named=True))
            t = t + np.timedelta64(STEP, "D")
    return out[:, 0], f


def main():
    gl = load_gamelogs(range(2012, LAST + 1))
    hand = load_handedness(range(2012, LAST + 1))
    rows = team_rows(gl, hand).sort(["date", "game_id", "is_home"])
    print(f"{rows.height:,} renglones (equipo × juego)")
    res = {}
    rg = {"hit": 40.0, "sp": 150.0, "pen": 6.0, "park": 10.0, "team": 6.0}
    grid = [("solo equipos", MLBParams(use_players=False, use_context=False), False),
            ("equipo+jugadores", MLBParams(ridge=rg), False),
            ("+ K/BB/HR del abridor", MLBParams(ridge=rg, covariates=("sp_k", "sp_bb", "sp_hr")), True),
            ("+ K/BB/HR, abridor 400", MLBParams(ridge={**rg, "sp": 400.0}, covariates=("sp_k", "sp_bb", "sp_hr")), True)]
    for name, prm, sk in grid:
        t = time.time()
        lr, last = walk(rows, prm, sk)
        res[name] = (lr, last)
        print(f"  {name}: listo ({time.time() - t:.0f} s)", flush=True)

    # Pares local/visitante por juego
    gid = rows["game_id"].to_list()
    home = rows["is_home"].to_numpy() == 1
    runs = rows["runs"].to_numpy()
    season = rows["season"].to_numpy()
    idx_h = {g: i for i, (g, h) in enumerate(zip(gid, home)) if h}
    idx_a = {g: i for i, (g, h) in enumerate(zip(gid, home)) if not h}
    games = [g for g in idx_h if g in idx_a]
    hi = np.array([idx_h[g] for g in games]); ai = np.array([idx_a[g] for g in games])
    y = (runs[hi] > runs[ai]).astype(float)
    tot = runs[hi] + runs[ai]
    ss = season[hi]
    print(f"\n{'':24} {'log loss ganador':>17} {'Brier':>7} {'acierto':>8} {'error total':>12}   (2015–2019 | 2021–2025)")
    pb = {}
    for s in np.unique(ss):
        prev = (ss < s) & (ss >= s - 3)
        pb[s] = y[prev].mean() if prev.any() else 0.53
    base = np.array([pb[s] for s in ss])
    def metrics(p, et, m):
        ll = -np.mean(y[m] * np.log(p[m]) + (1 - y[m]) * np.log(1 - p[m]))
        return ll, np.mean((p[m] - y[m]) ** 2), np.mean((p[m] > 0.5) == (y[m] == 1)), (np.mean(np.abs(et[m] - tot[m])) if et is not None else np.nan)
    periods = (((ss >= 2015) & (ss <= 2019)), ((ss >= 2021) & (ss <= 2025)))
    line = lambda nm, p, et: print(f"{nm:24} " + " | ".join("{:.4f} {:.4f} {:.1%} {:.2f}".format(*metrics(p, et, m)) for m in periods))
    line("base (local %)", base, None)
    res_ll = {}
    for name, (lr, last) in res.items():
        r = last.dispersion
        lh, la = np.exp(lr[hi]), np.exp(lr[ai])
        ok = ~np.isnan(lh) & ~np.isnan(la)
        p = np.full(len(games), np.nan); et = np.full(len(games), np.nan)
        for k in np.where(ok)[0]:
            g = game_probs(lh[k], la[k], r, run_line=None)
            p[k] = g["home_win"]; et[k] = lh[k] + la[k]
        p = np.clip(np.nan_to_num(p, nan=0.53), 0.02, 0.98); et = np.nan_to_num(et, nan=8.8)
        line(name, p, et)
        res_ll[name] = metrics(p, et, periods[0])[0]
    best = min((k for k in res if k != "solo equipos"), key=lambda k: res_ll.get(k, 9))
    print(f"\nMejor (por 2015–2019): {best}")
    f = res[best][1]
    b = f.beta
    for c in f.params.covariates:
        print(f"  coeficiente {c}: {f.beta[f.index[('cov', c)]]:+.3f}")
    print(f"\nCoeficientes (mejor, último ajuste): localía {b[f.index[('home','')]]:+.3f} · noche {b[f.index[('night','')]]:+.3f} · "
          f"ventaja de mano {b[f.index[('platoon','')]]:+.3f} · cansancio del bullpen rival {b[f.index[('fatigue','')]]:+.4f} por lanzador")
    from app.services.mlb_history import load_people
    ppl = load_people(); nm = {r_: f"{a} {l}" for r_, a, l in zip(ppl["retro_id"].to_list(), ppl["first"].to_list(), ppl["last"].to_list())}
    sp = sorted(((b[k], key[1]) for key, k in f.index.items() if key[0] == "sp"), reverse=True)
    hit = sorted(((b[k], key[1]) for key, k in f.index.items() if key[0] == "hit"), reverse=True)
    print("Mejores abridores:", [(nm.get(x, x), round(v, 3)) for v, x in sp[:8]])
    print("Mejores bateadores:", [(nm.get(x, x), round(v, 3)) for v, x in hit[:8]])


if __name__ == "__main__":
    main()
