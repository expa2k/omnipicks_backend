"""
OmniPicks — NFL: backtest de apuestas (fase 5).

Semana a semana, solo con lo que se sabía antes de cada partido:
  · margen previsto (EPA + marcador + QB titular) y total previsto (anotación + EPA + ritmo + clima);
  · probabilidades de cubrir el spread, de over/under y de ganar, con números clave;
  · cada probabilidad se calibra con las temporadas anteriores (P = Φ(k·z)), porque la que sale
    directa del modelo está inflada.

Se apuesta 1 unidad con los momios reales de cierre (nflverse) cuando el valor esperado
(probabilidad × momio decimal − 1) pasa un umbral. También se prueba la regla "por puntos" (el
modelo difiere de la línea en 3 o más puntos), que se parece a lo que hace la app con la NBA.

El mercado no entra al modelo: solo define la apuesta (línea y momio) y el resultado.

Uso:
    python ml/scripts/nfl_backtest.py
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.nfl_history import load_pbp, team_games
from app.services.nfl_qb import QBParams, qb_games
from app.services.nfl_ratings import RatingParams
from app.services.nfl_score import MarginDistribution, total_probs
from nfl_context_eval import walk_forward_w
from nfl_qb_eval import qb_features
from nfl_ratings_eval import games_table, points_features, rating_features
from nfl_score_eval import pace_features, scoring_features, total_epa_features, weather_features

MIN_PRIOR = 500          # partidos previos para calibrar
REPORT_FROM = 2017


def decimal(american: float) -> float:
    a = float(american if american is not None and not np.isnan(american) else -110)
    return 1 + (a / 100 if a > 0 else 100 / -a)


def calibrate(z: np.ndarray, y: np.ndarray, season: np.ndarray, valid: np.ndarray) -> (np.ndarray, dict):
    """P = Φ(k·z), con k aprendido en las temporadas anteriores (solo eventos sin push)."""
    out, ks = np.full(len(z), np.nan), {}
    for s in np.unique(season):
        tr = valid & (season < s) & ~np.isnan(z)
        if tr.sum() < MIN_PRIOR:
            continue
        nll = lambda k: -np.mean(y[tr] * np.log(norm.cdf(k * z[tr]) + 1e-12)
                                 + (1 - y[tr]) * np.log(1 - norm.cdf(k * z[tr]) + 1e-12))
        k = minimize_scalar(nll, bounds=(0.0, 1.5), method="bounded").x
        ks[int(s)] = round(k, 2)
        te = season == s
        out[te] = norm.cdf(k * z[te])
    return out, ks


def settle(bets, label):
    """bets: lista de (temporada, ganancia en unidades). Imprime registro, unidades y ROI."""
    if not bets:
        print(f"  {label:42} sin apuestas")
        return
    b = np.array(bets)
    for name, m in (("2017–2020", b[:, 0] <= 2020), ("2021–2026", b[:, 0] >= 2021)):
        g = b[m, 1]
        if not len(g):
            continue
        w, l = int((g > 0).sum()), int((g < 0).sum())
        se = g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else np.nan
        print(f"  {label:42} {name}: {len(g):>4} apuestas · {w}-{l} ({w / max(w + l, 1):.1%}) · "
              f"{g.sum():+7.1f} u · ROI {g.mean():+.1%} (±{1.96 * se:.1%})")


def main():
    games = games_table()
    pbp = load_pbp(range(2010, 2027))
    tg, qbg = team_games(pbp), qb_games(pbp)
    season = games["season"].to_numpy()
    margin = games["result"].to_numpy().astype(float)
    total = games["total"].to_numpy().astype(float)
    col = lambda c: games[c].to_numpy().astype(float)
    spread, tline = col("spread_line"), col("total_line")
    hso, aso, oo, uo, hml, aml = (col(c) for c in ("home_spread_odds", "away_spread_odds", "over_odds",
                                                   "under_odds", "home_moneyline", "away_moneyline"))

    # 1) Predicciones semana a semana
    epa_p = RatingParams(half_life=16, offseason_gap=12, ridge=300)
    base = np.c_[rating_features(games, tg, epa_p),
                 points_features(games, RatingParams(half_life=16, offseason_gap=12), ridge=8.0),
                 qb_features(games, qbg, QBParams())]
    mu = walk_forward_w(base, margin, season)
    xt = np.c_[scoring_features(games, RatingParams(half_life=8, offseason_gap=12), 8.0),
               total_epa_features(games, tg, epa_p), pace_features(games, pbp), weather_features(games)]
    mt = walk_forward_w(xt, total, season)

    # 2) Probabilidades crudas (con números clave) y calibradas
    n = len(games)
    p_cover, p_over, p_win = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for s in np.unique(season):
        tr = (season < s) & ~np.isnan(mu)
        te = np.where((season == s) & ~np.isnan(mu))[0]
        if tr.sum() < MIN_PRIOR or not len(te):
            continue
        sig = float(np.std(margin[tr] - mu[tr]))
        sig_t = float(np.nanstd(total[tr] - mt[tr]))
        dist = MarginDistribution.fit(mu[tr], margin[tr], sig)
        for i in te:
            pr = dist.probs(mu[i], -spread[i])
            p_cover[i] = pr["home_cover"] / max(1 - pr["push"], 1e-9)
            p_win[i] = pr["home_win"]
            tp = total_probs(mt[i], sig_t, tline[i])
            p_over[i] = tp["over"] / max(1 - tp["push"], 1e-9)

    ats_real = margin - spread               # >0: cubrió el local
    ou_real = total - tline                  # >0: over
    z = lambda p: norm.ppf(np.clip(p, 1e-4, 1 - 1e-4))
    c_cover, k_cover = calibrate(z(p_cover), (ats_real > 0).astype(float), season, ats_real != 0)
    c_over, k_over = calibrate(z(p_over), (ou_real > 0).astype(float), season, ou_real != 0)
    c_win, k_win = calibrate(z(p_win), (margin > 0).astype(float), season, margin != 0)
    print("Factor de calibración k por temporada (1 = el modelo crudo está bien calibrado):")
    print("  spread:  " + " · ".join(f"{s}: {k}" for s, k in k_cover.items()))
    print("  total:   " + " · ".join(f"{s}: {k}" for s, k in k_over.items()))
    print("  ganador: " + " · ".join(f"{s}: {k}" for s, k in k_win.items()))

    # Brier de cada mercado
    print("\nBrier (0.2500 = decir siempre 50%):")
    for name, p, y, ok in (("spread", c_cover, ats_real > 0, ats_real != 0), ("total", c_over, ou_real > 0, ou_real != 0),
                           ("ganador", c_win, margin > 0, margin != 0)):
        h = ok & (season >= 2021) & ~np.isnan(p)
        imp = None
        if name == "ganador":
            dh, da = np.array([decimal(x) for x in hml]), np.array([decimal(x) for x in aml])
            imp = (1 / dh) / (1 / dh + 1 / da)
        line = f"  {name:8} modelo calibrado {np.mean((p[h] - y[h]) ** 2):.4f}"
        if imp is not None:
            line += f" · mercado sin margen {np.mean((imp[h] - y[h]) ** 2):.4f}"
        print(line)

    # 3) Apuestas
    def market_bets(p_home, home_odds, away_odds, real, thr):
        bets = []
        for i in np.where((season >= REPORT_FROM) & ~np.isnan(p_home))[0]:
            for side, p, odds, sign in (("home", p_home[i], home_odds[i], 1), ("away", 1 - p_home[i], away_odds[i], -1)):
                d = decimal(odds)
                if p * d - 1 > thr:
                    r = real[i] * sign
                    bets.append((season[i], 0.0 if r == 0 else (d - 1 if r > 0 else -1.0)))
        return bets

    print("\nApuestas de 1 unidad con momios reales de cierre:")
    for thr in (0.0, 0.02, 0.04, 0.06):
        print(f" Valor esperado > {thr:.0%}")
        settle(market_bets(c_cover, hso, aso, ats_real, thr), "spread")
        settle(market_bets(c_over, oo, uo, ou_real, thr), "total (over / under)")
        settle(market_bets(c_win, hml, aml, margin, thr), "ganador (moneyline)")

    print("\n Regla por puntos (sin calibrar): el modelo difiere de la línea en ≥ X puntos")
    for X in (1.5, 3.0, 5.0):
        sb, tb = [], []
        for i in np.where((season >= REPORT_FROM) & ~np.isnan(mu))[0]:
            e = mu[i] - spread[i]
            if abs(e) >= X:
                r = ats_real[i] * np.sign(e)
                d = decimal(hso[i] if e > 0 else aso[i])
                sb.append((season[i], 0.0 if r == 0 else (d - 1 if r > 0 else -1.0)))
            et = mt[i] - tline[i]
            if not np.isnan(et) and abs(et) >= X:
                r = ou_real[i] * np.sign(et)
                d = decimal(oo[i] if et > 0 else uo[i])
                tb.append((season[i], 0.0 if r == 0 else (d - 1 if r > 0 else -1.0)))
        settle(sb, f"spread, diferencia ≥ {X}")
        settle(tb, f"total, diferencia ≥ {X}")

    print("\nReferencia: apostar siempre al local con el spread")
    settle([(season[i], 0.0 if ats_real[i] == 0 else (decimal(hso[i]) - 1 if ats_real[i] > 0 else -1.0))
            for i in np.where(season >= REPORT_FROM)[0]], "siempre local (spread)")


if __name__ == "__main__":
    main()
