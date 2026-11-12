"""
OmniPicks — NFL: total de puntos y probabilidades con números clave (fase 4).

1) Total: rating de anotación (ataque/defensa en puntos) + EPA esperado de los dos ataques +
   ritmo (jugadas por partido) + clima (viento, frío, techo). Contra el total de cierre.
2) Margen → probabilidades: normal contra normal corregida con los números clave (3, 7…).
   Se mide con el margen exacto (log-loss) y con la probabilidad de cubrir la línea de cierre.

Mismo esquema: se elige con 2015–2020 y se reporta 2021–2026 aparte.

Uso:
    python ml/scripts/nfl_score_eval.py
"""
import itertools
import os
import sys
from collections import defaultdict

import numpy as np
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.nfl_history import load_pbp, normalize_teams, team_games
from app.services.nfl_qb import QBParams, qb_games
from app.services.nfl_ratings import (NFLRatingEngine, RatingParams, ScoringRatingEngine,
                                      expected_margin_epa)
from app.services.nfl_score import SUPPORT, MarginDistribution, _normal_mass
from nfl_context_eval import walk_forward_w
from nfl_qb_eval import HOLDOUT, TUNE, qb_features
from nfl_ratings_eval import EVAL_FROM, games_table, points_features, rating_features


def total_epa_features(games, tg, params) -> np.ndarray:
    neutral = games.filter(pl.col("location") == "Neutral")["game_id"].to_list()
    eng = NFLRatingEngine(tg, params, neutral_games=neutral)
    cache, x = {}, []
    for r in games.iter_rows(named=True):
        key = (r["season"], r["week"])
        if key not in cache:
            cache[key] = eng.at(*key)
        x.append(expected_margin_epa(cache[key], r["home_team"], r["away_team"], r["location"] == "Neutral")[1])
    return np.array(x)


def scoring_features(games, params, ridge) -> np.ndarray:
    eng = ScoringRatingEngine(games, params)
    cache, x = {}, []
    for r in games.iter_rows(named=True):
        key = (r["season"], r["week"])
        if key not in cache:
            cache[key] = eng.at(*key, ridge=ridge)
        mu, h, off, dfn = cache[key]
        hm, aw = r["home_team"], r["away_team"]
        home = 0.0 if r["location"] == "Neutral" else h
        x.append(2 * mu + home + off.get(hm, 0) - dfn.get(aw, 0) + off.get(aw, 0) - dfn.get(hm, 0))
    return np.array(x)


def pace_features(games, pbp, half_life_games: float = 8.0) -> np.ndarray:
    """Jugadas por partido de cada equipo (todas, con descuento), suma de los dos, antes de cada semana."""
    plays = normalize_teams(pbp).group_by(["season", "week", "posteam"]).agg(n=pl.len()).sort(["season", "week"])
    d = 0.5 ** (1.0 / half_life_games)
    num, den, snap = defaultdict(float), defaultdict(float), {}
    by_week = defaultdict(list)
    for r in plays.iter_rows(named=True):
        by_week[(r["season"], r["week"])].append((r["posteam"], r["n"]))
    keys = sorted(set(by_week) | set(zip(games["season"].to_list(), games["week"].to_list())))
    for k in keys:
        lg = sum(num.values()) / max(sum(den.values()), 1e-9) if den else 62.0
        snap[k] = ({t: num[t] / den[t] for t in den}, lg)
        for t, n in by_week.get(k, []):
            num[t] = d * num[t] + n
            den[t] = d * den[t] + 1.0
    x = []
    for r in games.iter_rows(named=True):
        pace, lg = snap[(r["season"], r["week"])]
        x.append(pace.get(r["home_team"], lg) + pace.get(r["away_team"], lg) - 2 * lg)
    return np.array(x)


def weather_features(games) -> np.ndarray:
    rows = []
    for r in games.iter_rows(named=True):
        indoor = r["roof"] in ("dome", "closed")
        wind = 0.0 if indoor or r["wind"] is None else float(r["wind"])
        temp = 65.0 if indoor or r["temp"] is None else float(r["temp"])
        rows.append([1.0 if indoor else 0.0, max(0.0, wind - 10.0), max(0.0, 40.0 - temp)])
    return np.array(rows)


def mae(p, y, m):
    return float(np.mean(np.abs(p[m] - y[m])))


def main():
    games = games_table()
    pbp = load_pbp(range(2010, 2027))
    tg, qbg = team_games(pbp), qb_games(pbp)
    season = games["season"].to_numpy()
    tune = (season >= TUNE[0]) & (season <= TUNE[1])
    hold = (season >= HOLDOUT[0]) & (season <= HOLDOUT[1])
    margin = games["result"].to_numpy().astype(float)
    total = games["total"].to_numpy().astype(float)
    mk_spread = games["spread_line"].to_numpy().astype(float)
    mk_total = games["total_line"].to_numpy().astype(float)
    ok_total = ~np.isnan(mk_total)

    # ── 1) Total ──────────────────────────────────────────────────────────
    epa_p = RatingParams(half_life=16, offseason_gap=12, ridge=300)
    x_te = total_epa_features(games, tg, epa_p)
    x_pace = pace_features(games, pbp)
    x_w = weather_features(games)
    print("Rating de anotación (elegido con 2015–2020):")
    best = None
    for hl, gap, lam in itertools.product((8, 16, 32), (4, 12), (2.0, 4.0, 8.0)):
        xs = scoring_features(games, RatingParams(half_life=hl, offseason_gap=gap), lam)
        p = walk_forward_w(np.c_[xs, x_te, x_pace, x_w], total, season)
        m = mae(p, total, tune & ok_total)
        print(f"  vida media {hl:>2} · pausa {gap:>2} · ridge {lam:>3} → MAE {m:.3f}")
        if best is None or m < best[0]:
            best = (m, hl, gap, lam, xs)
    x_sc = best[4]
    print(f"Mejor: vida media {best[1]} · pausa {best[2]} · ridge {best[3]}\n")

    league = walk_forward_w(np.zeros((len(total), 1)), total, season)
    rows = {
        "Media de la liga": league,
        "Rating de anotación": walk_forward_w(x_sc[:, None], total, season),
        "+ EPA esperado de los ataques": walk_forward_w(np.c_[x_sc, x_te], total, season),
        "+ ritmo": walk_forward_w(np.c_[x_sc, x_te, x_pace], total, season),
        "+ clima (modelo completo)": walk_forward_w(np.c_[x_sc, x_te, x_pace, x_w], total, season),
        "Mercado (total de cierre)": mk_total,
    }
    print("Total de puntos, error medio:")
    print(f"{'':32} {'2015–2020':>10} {'2021–2026':>10}")
    for name, p in rows.items():
        print(f"{name:32} {mae(p, total, tune & ok_total):10.2f} {mae(p, total, hold & ok_total):10.2f}")
    p_tot = rows["+ clima (modelo completo)"]
    tr = (season >= 2011) & (season <= 2025)
    X = np.c_[np.ones(tr.sum()), x_sc[tr], x_te[tr], x_pace[tr], x_w[tr]]
    coef = np.linalg.lstsq(X, total[tr], rcond=None)[0]
    print("  clima: techo {:+.2f} pts · por mph de viento sobre 10 {:+.2f} · por °F bajo 40 {:+.2f}".format(*coef[-3:]))
    res_t = total - p_tot
    print(f"  desviación del total alrededor del modelo: {np.nanstd(res_t[season >= EVAL_FROM]):.2f}")

    # ── 2) Margen → probabilidades ────────────────────────────────────────
    base = np.c_[rating_features(games, tg, epa_p),
                 points_features(games, RatingParams(half_life=16, offseason_gap=12), ridge=8.0),
                 qb_features(games, qbg, QBParams())]
    mu = walk_forward_w(base, margin, season)
    print(f"\nMargen: desviación alrededor del modelo {np.nanstd((margin - mu)[season >= EVAL_FROM]):.2f}")

    ll = {"normal": [], "números clave": []}
    cover = {"normal": [], "números clave": []}
    pushes = []
    for s in range(EVAL_FROM, int(season.max()) + 1):
        tr = (season < s) & ~np.isnan(mu)
        if tr.sum() < 500:            # hacen falta partidos previstos para aprender los números clave
            continue
        sigma = float(np.std(margin[tr] - mu[tr]))
        dists = {"normal": MarginDistribution(sigma=sigma),
                 "números clave": MarginDistribution.fit(mu[tr], margin[tr], sigma)}
        for i in np.where((season == s) & ~np.isnan(mu))[0]:
            k = int(round(margin[i])) - SUPPORT[0]
            line = -mk_spread[i]                    # spread del local al estilo casino (−3 = favorito)
            real = margin[i] + line
            for name, d in dists.items():
                p = d.pmf(mu[i])
                ll[name].append((season[i], -np.log(max(p[k], 1e-12))))
                pr = d.probs(mu[i], line)
                if real != 0:
                    cover[name].append((season[i], pr["home_cover"] / (1 - pr["push"]), 1.0 if real > 0 else 0.0))
                if name == "números clave" and float(line).is_integer():
                    pushes.append((abs(line), pr["push"], 1.0 if real == 0 else 0.0))

    # Calibración de "cubrir": P = Φ(k · (prob. del modelo en z)) con k aprendido de temporadas previas
    from scipy.optimize import minimize_scalar
    from scipy.stats import norm as _n
    c = np.array(cover["números clave"])
    z = _n.ppf(np.clip(c[:, 1], 1e-4, 1 - 1e-4))
    cal = np.full(len(c), np.nan)
    ks = {}
    for s in np.unique(c[:, 0]):
        tr = c[:, 0] < s
        if tr.sum() < 500:
            continue
        nll = lambda k: -np.mean(c[tr, 2] * np.log(_n.cdf(k * z[tr]) + 1e-12)
                                 + (1 - c[tr, 2]) * np.log(1 - _n.cdf(k * z[tr]) + 1e-12))
        k = minimize_scalar(nll, bounds=(0.0, 1.5), method="bounded").x
        ks[int(s)] = k
        cal[c[:, 0] == s] = _n.cdf(k * z[c[:, 0] == s])
    cover["calibrada"] = [(a, b, y) for (a, _, y), b in zip(c, cal) if not np.isnan(b)]

    print("\nMargen exacto (log-loss, menor es mejor):")
    for name in ll:
        a = np.array(ll[name])
        print(f"  {name:16} 2017–2020 {a[a[:, 0] <= 2020, 1].mean():.4f} · 2021–2026 {a[a[:, 0] >= 2021, 1].mean():.4f}")
    print("\nProbabilidad de que el local cubra la línea de cierre (Brier; 0.2500 = decir siempre 50%):")
    for name in cover:
        cc = np.array(cover[name])
        h = cc[:, 0] >= 2021
        print(f"  {name:16} 2021–2026 {np.mean((cc[h, 1] - cc[h, 2]) ** 2):.4f}")
    print("  factor de calibración k por temporada: " + " · ".join(f"{s}: {k:.2f}" for s, k in ks.items()))
    cc = np.array(cover["calibrada"]); h = cc[:, 0] >= 2021
    for lo in (0.53, 0.55, 0.58):
        sel = h & ((cc[:, 1] >= lo) | (cc[:, 1] <= 1 - lo))
        hit = np.mean((cc[sel, 1] >= 0.5) == (cc[sel, 2] == 1)) if sel.any() else float("nan")
        print(f"  partidos con prob. calibrada ≥ {lo:.0%} de un lado: {int(sel.sum())} · aciertos {hit:.1%}")
    pu = np.array(pushes)
    print("\nEmpate con la línea (push), previsto contra real:")
    for L in (3.0, 7.0):
        m = pu[:, 0] == L
        if m.any():
            print(f"  línea {L:.0f}: previsto {pu[m, 1].mean():.1%} · real {pu[m, 2].mean():.1%} ({int(m.sum())} partidos)")
    m = ~np.isin(pu[:, 0], (3.0, 7.0))
    print(f"  otras líneas enteras: previsto {pu[m, 1].mean():.1%} · real {pu[m, 2].mean():.1%} ({int(m.sum())} partidos)")
    d = MarginDistribution.fit(mu[~np.isnan(mu)], margin[~np.isnan(mu)], float(np.nanstd(margin - mu)))
    print("\nPeso de cada margen exacto (1 = lo que espera la normal):")
    print("  " + " · ".join(f"{k}: {d.weights[k]:.2f}" for k in (0, 1, 2, 3, 4, 6, 7, 8, 10, 14, 17, 21)))


if __name__ == "__main__":
    main()
