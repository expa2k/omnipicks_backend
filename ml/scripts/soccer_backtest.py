"""
OmniPicks — Fútbol v2, fase 4: mercados y backtest de apuestas con momios reales de cierre.

Modelo de la fase 2 (Dixon-Coles con 70% goles + 30% tiros a puerta en las 5 europeas; solo goles en
la Liga MX), semana a semana. De la matriz de marcadores salen:
  · 1X2 → contra el cierre de Pinnacle (la línea más dura) y el de Bet365;
  · más/menos 2.5 → contra el cierre promedio (desde 2019);
  · hándicap asiático → contra la línea y momios de cierre promedio (desde 2019; líneas de ¼ se
    reparten en dos medias apuestas).

Calibración: temperatura sobre las probabilidades 1X2 (p^(1/T) normalizado) y Φ(k·z) en los binarios,
aprendidas con las temporadas anteriores. Apuesta de 1 unidad cuando el valor esperado pasa un umbral.

Uso:
    python ml/scripts/soccer_backtest.py
"""
import os
import pickle
import sys

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from scipy.stats import norm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.soccer_dc import GOALS_PER_SOT, DCParams, fit_dc, fit_rates, markets
from app.services.soccer_history import CACHE_DIR, load_all

LEAGUES = ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1", "Liga MX"]
SHOTS = {"Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"}
FIRST = 2019
# Caché local de predicciones (la escribe este mismo script; nunca se carga un pickle de fuera).
CACHE = os.path.join(CACHE_DIR, "backtest_preds.pkl")


def predict_league(df: pl.DataFrame, use_shots: bool, params=DCParams()):
    df = df.filter(pl.col("FTHG").is_not_null() & pl.col("date").is_not_null()).sort("date")
    dates = df["date"].to_numpy().astype("datetime64[D]")
    home, away = df["HomeTeam"].to_list(), df["AwayTeam"].to_list()
    hg, ag = df["FTHG"].to_numpy(), df["FTAG"].to_numpy()
    has = (df["HST"].is_not_null() & df["AST"].is_not_null()).to_numpy() if "HST" in df.columns else np.zeros(len(hg), bool)
    hst = df["HST"].fill_null(0).to_numpy() if "HST" in df.columns else np.zeros(len(hg))
    ast = df["AST"].fill_null(0).to_numpy() if "AST" in df.columns else np.zeros(len(hg))
    season = df["season"].to_numpy()
    week = (dates - np.datetime64("2000-01-03")).astype(int) // 7
    preds = [None] * df.height
    fg, fs = None, None
    for wk in np.unique(week):
        te = np.where(week == wk)[0]
        if season[te[0]] < FIRST:
            continue
        tr = np.where(week < wk)[0]
        if len(tr) < 300:
            continue
        days = (dates[te[0]] - dates[tr]).astype(int).astype(float)
        H, A = [home[i] for i in tr], [away[i] for i in tr]
        fg = fit_dc(H, A, hg[tr], ag[tr], days, params, init=fg)
        f = fg
        if use_shots:
            yh = np.where(has[tr], 0.7 * hg[tr] + 0.3 * GOALS_PER_SOT * hst[tr], hg[tr])
            ya = np.where(has[tr], 0.7 * ag[tr] + 0.3 * GOALS_PER_SOT * ast[tr], ag[tr])
            f = fs = fit_rates(H, A, yh, ya, days, params, init=fs, rho=fg.rho)
        for i in te:
            preds[i] = f.score_matrix(home[i], away[i]).astype(np.float32)
    return df, preds


def margin_pmf(m):
    n = m.shape[0]
    out = np.zeros(2 * n - 1)
    for i in range(n):
        for j in range(n):
            out[i - j + n - 1] += m[i, j]
    return out          # índice n-1 = margen 0


def ah_ev(pmf, line, odds):
    """Valor esperado por unidad de apostar al local con hándicap `line` a momio decimal `odds`
    (líneas de ¼ = dos medias apuestas). Devuelve (ev, función de pago para un margen dado)."""
    n = (len(pmf) + 1) // 2
    margins = np.arange(-(n - 1), n)
    parts = [line] if (line * 4) % 2 == 0 else [line - 0.25, line + 0.25]
    def pay(mg):
        tot = 0.0
        for L in parts:
            r = mg + L
            tot += (odds - 1) if r > 0 else (0.0 if r == 0 else -1.0)
        return tot / len(parts)
    ev = float(sum(p * pay(mg) for p, mg in zip(pmf, margins)))
    return ev, pay


def main():
    if os.path.exists(CACHE):
        rows = pickle.load(open(CACHE, "rb"))
    else:
        data = load_all(2010)
        rows = []
        for lg in LEAGUES:
            df, preds = predict_league(data.filter(pl.col("league") == lg), lg in SHOTS)
            for r, m in zip(df.iter_rows(named=True), preds):
                if m is not None:
                    rows.append((r, m))
            print(f"  {lg}: listo", flush=True)
        pickle.dump(rows, open(CACHE, "wb"))
    R = [r for r, _ in rows]
    M = [m for _, m in rows]
    season = np.array([r["season"] for r in R])
    hg = np.array([r["FTHG"] for r in R]); ag = np.array([r["FTAG"] for r in R])
    y = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    P = np.array([[markets(m)["home"], markets(m)["draw"], markets(m)["away"]] for m in M])
    Pov = np.array([markets(m)["over_2.5"] for m in M])
    col = lambda k: np.array([np.nan if r.get(k) is None else float(r[k]) for r in R])

    # ── Calibración por temperatura (1X2) aprendida con temporadas anteriores ──
    Pc = P.copy(); Ts = {}
    for s in np.unique(season):
        tr = season < s
        if tr.sum() < 1000:
            continue
        def nll(T):
            q = P[tr] ** (1 / T); q /= q.sum(axis=1, keepdims=True)
            return -np.mean(np.log(q[np.arange(tr.sum()), y[tr]]))
        T = minimize_scalar(nll, bounds=(0.5, 2.0), method="bounded").x
        Ts[int(s)] = round(T, 3)
        q = P[season == s] ** (1 / T); Pc[season == s] = q / q.sum(axis=1, keepdims=True)
    print("Temperatura (1 = bien calibrado; > 1 = el modelo exagera):", Ts)
    ll = lambda Q, m: float(-np.mean(np.log(np.clip(Q[m][np.arange(m.sum()), y[m]], 1e-9, None))))
    ph, pd_, pa = col("PSCH"), col("PSCD"), col("PSCA")
    okp = ~np.isnan(ph) & ~np.isnan(pd_) & ~np.isnan(pa)
    inv = np.c_[1 / ph, 1 / pd_, 1 / pa]; Q = inv / inv.sum(axis=1, keepdims=True)
    for lo, hi in ((2020, 2021), (2022, 2026)):
        m = (season >= lo) & (season <= hi) & okp & ~np.isnan(Pc[:, 0])
        print(f"  log loss {lo}–{hi}: modelo {ll(P, m):.4f} · calibrado {ll(Pc, m):.4f} · Pinnacle {ll(Q, m):.4f}  ({m.sum()} partidos)")

    # Fiabilidad: probabilidad prevista contra frecuencia real (todas las salidas)
    print("\nCalibración del 1X2 (calibrado, 2022–2026): prob. prevista → frecuencia real")
    h = (season >= 2022) & ~np.isnan(Pc[:, 0])
    pp, oo = Pc[h].ravel(), np.eye(3)[y[h]].ravel()
    for lo_, hi_ in ((0, .2), (.2, .35), (.35, .5), (.5, .65), (.65, 1.0)):
        b = (pp >= lo_) & (pp < hi_)
        print(f"  {lo_:.2f}–{hi_:.2f}: {pp[b].mean():.3f} → {oo[b].mean():.3f}  ({b.sum()})")

    def settle(bets, label):
        if not bets:
            print(f"  {label:44} sin apuestas"); return
        b = np.array(bets)
        for nm, m in (("2020–21", b[:, 0] <= 2021), ("2022–26", b[:, 0] >= 2022)):
            g = b[m, 1]
            if len(g) == 0:
                continue
            se = g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else np.nan
            w = int((g > 0).sum()); l = int((g < 0).sum())
            print(f"  {label:44} {nm}: {len(g):>5} apuestas · {w}-{l} · {g.sum():+8.1f} u · ROI {g.mean():+.1%} (±{1.96 * se:.1%})")

    print("\n1X2, 1 unidad cuando valor esperado > umbral (prob. calibrada):")
    for book, cols in (("Pinnacle", ("PSCH", "PSCD", "PSCA")), ("Bet365", ("B365CH", "B365CD", "B365CA"))):
        O = np.c_[col(cols[0]), col(cols[1]), col(cols[2])]
        for thr in (0.0, 0.05, 0.10):
            bets = []
            for i in np.where(~np.isnan(Pc[:, 0]) & ~np.isnan(O).any(axis=1))[0]:
                ev = Pc[i] * O[i] - 1
                k = int(np.argmax(ev))
                if ev[k] > thr:
                    bets.append((season[i], (O[i, k] - 1) if y[i] == k else -1.0))
            settle(bets, f"{book}, valor > {thr:.0%}")

    print("\nMás/menos 2.5 (cierre promedio, desde 2019), prob. calibrada Φ(k·z):")
    oov, oun = col("AvgC>2.5"), col("AvgC<2.5")
    tot = hg + ag
    z = norm.ppf(np.clip(Pov, 1e-4, 1 - 1e-4)); Pov_c = np.full(len(z), np.nan)
    for s in np.unique(season):
        tr = (season < s) & ~np.isnan(z)
        if tr.sum() < 1000:
            continue
        yy = (tot[tr] > 2.5).astype(float)
        nll = lambda k: -np.mean(yy * np.log(norm.cdf(k * z[tr]) + 1e-12) + (1 - yy) * np.log(1 - norm.cdf(k * z[tr]) + 1e-12))
        k = minimize_scalar(nll, bounds=(0.2, 2.0), method="bounded").x
        Pov_c[season == s] = norm.cdf(k * z[season == s])
    for thr in (0.0, 0.05, 0.10):
        bets = []
        for i in np.where(~np.isnan(Pov_c) & ~np.isnan(oov) & ~np.isnan(oun))[0]:
            e_o, e_u = Pov_c[i] * oov[i] - 1, (1 - Pov_c[i]) * oun[i] - 1
            if max(e_o, e_u) > thr:
                win = (tot[i] > 2.5) if e_o >= e_u else (tot[i] < 2.5)
                bets.append((season[i], ((oov[i] if e_o >= e_u else oun[i]) - 1) if win else -1.0))
        settle(bets, f"Más/menos 2.5, valor > {thr:.0%}")

    print("\nHándicap asiático (línea y momios de cierre promedio, desde 2019):")
    line, oh, oa = col("AHCh"), col("AvgCAHH"), col("AvgCAHA")
    for thr in (0.0, 0.05, 0.10):
        bets = []
        for i in np.where(~np.isnan(line) & ~np.isnan(oh) & ~np.isnan(oa))[0]:
            pmf = margin_pmf(M[i].astype(float))
            ev_h, pay_h = ah_ev(pmf, line[i], oh[i])
            ev_a, pay_a = ah_ev(pmf[::-1], -line[i], oa[i])
            mg = hg[i] - ag[i]
            if max(ev_h, ev_a) > thr:
                bets.append((season[i], pay_h(mg) if ev_h >= ev_a else pay_a(-mg)))
        settle(bets, f"Hándicap asiático, valor > {thr:.0%}")


if __name__ == "__main__":
    main()
