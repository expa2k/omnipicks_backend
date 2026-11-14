"""
OmniPicks — Fútbol v2, fase 1: Dixon-Coles.

Goles de cada equipo con Poisson:

    λ_local     = exp(μ + localía + ataque[local]  − defensa[visitante])
    λ_visitante = exp(μ +           ataque[visitante] − defensa[local])

con la corrección de Dixon y Coles para los marcadores bajos (0-0, 1-0, 0-1, 1-1), que en el fútbol
salen más (o menos) de lo que diría una Poisson independiente. Se ajusta por máxima verosimilitud
con cada partido pesando según su antigüedad (exp(−ξ·días)) y un encogimiento (ridge) que evita
ratings extremos en equipos con pocos partidos (ascendidos, inicio de temporada).

De la matriz de marcadores salen 1X2, goles totales, más/menos, ambos anotan y hándicaps.
"""
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


@dataclass
class DCParams:
    xi: float = 0.002          # descuento por día (vida media ≈ ln2/ξ ≈ 1 año)
    ridge: float = 2.0         # encogimiento de ataque y defensa hacia 0 (en "partidos" equivalentes)
    window_days: int = 1100    # se usan los partidos de los últimos ~3 años


@dataclass
class DCFit:
    teams: List[str]
    attack: np.ndarray
    defense: np.ndarray
    mu: float
    home: float
    rho: float

    def rates(self, home: str, away: str, neutral: bool = False) -> Tuple[float, float]:
        i = {t: k for k, t in enumerate(self.teams)}
        ah = self.attack[i[home]] if home in i else 0.0
        dh = self.defense[i[home]] if home in i else 0.0
        aa = self.attack[i[away]] if away in i else 0.0
        da = self.defense[i[away]] if away in i else 0.0
        lh = np.exp(self.mu + (0.0 if neutral else self.home) + ah - da)
        la = np.exp(self.mu + aa - dh)
        return float(lh), float(la)

    def score_matrix(self, home: str, away: str, neutral: bool = False) -> np.ndarray:
        lh, la = self.rates(home, away, neutral)
        return score_matrix(lh, la, self.rho)


def _tau(x, y, lh, la, rho):
    t = np.ones_like(lh)
    t = np.where((x == 0) & (y == 0), 1 - lh * la * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lh * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + la * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


def score_matrix(lh: float, la: float, rho: float) -> np.ndarray:
    """P(local = i, visitante = j) para i, j = 0..MAX_GOALS, con la corrección de Dixon-Coles."""
    g = np.arange(MAX_GOALS + 1)
    m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    m[0, 0] *= 1 - lh * la * rho
    m[0, 1] *= 1 + lh * rho
    m[1, 0] *= 1 + la * rho
    m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


def fit_dc(home: List[str], away: List[str], hg: np.ndarray, ag: np.ndarray, days_ago: np.ndarray,
           params: DCParams = DCParams(), init: Optional[DCFit] = None) -> DCFit:
    """Ajuste por máxima verosimilitud ponderada. `days_ago`: antigüedad de cada partido en días."""
    m = days_ago <= params.window_days
    home = [h for h, k in zip(home, m) if k]
    away = [a for a, k in zip(away, m) if k]
    hg, ag, w = hg[m], ag[m], np.exp(-params.xi * days_ago[m])
    teams = sorted(set(home) | set(away))
    idx = {t: k for k, t in enumerate(teams)}
    hi = np.array([idx[t] for t in home])
    ai = np.array([idx[t] for t in away])
    n = len(teams)

    def unpack(p):
        return p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1], p[2 * n + 2]

    def nll(p):
        att, dfn, mu, hfa, rho = unpack(p)
        lh = np.exp(mu + hfa + att[hi] - dfn[ai])
        la = np.exp(mu + att[ai] - dfn[hi])
        ll = poisson.logpmf(hg, lh) + poisson.logpmf(ag, la) + np.log(np.clip(_tau(hg, ag, lh, la, rho), 1e-10, None))
        return -np.sum(w * ll) + params.ridge * (np.sum(att ** 2) + np.sum(dfn ** 2))

    p0 = np.zeros(2 * n + 3)
    p0[2 * n] = np.log(max(np.average(np.r_[hg, ag], weights=np.r_[w, w]), 0.3))
    p0[2 * n + 1] = 0.25
    p0[2 * n + 2] = -0.05
    if init is not None:                      # arranque con el ajuste anterior (mucho más rápido)
        prev = {t: k for k, t in enumerate(init.teams)}
        for t, k in idx.items():
            if t in prev:
                p0[k] = init.attack[prev[t]]
                p0[n + k] = init.defense[prev[t]]
        p0[2 * n], p0[2 * n + 1], p0[2 * n + 2] = init.mu, init.home, init.rho
    bounds = [(None, None)] * (2 * n + 2) + [(-0.3, 0.3)]
    res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
    att, dfn, mu, hfa, rho = unpack(res.x)
    return DCFit(teams=teams, attack=att, defense=dfn, mu=float(mu), home=float(hfa), rho=float(rho))


def markets(m: np.ndarray) -> Dict[str, float]:
    """Probabilidades de los mercados principales a partir de la matriz de marcadores."""
    g = np.arange(m.shape[0])
    tot = g[:, None] + g[None, :]
    return {
        "home": float(np.tril(m, -1).sum()), "draw": float(np.trace(m)), "away": float(np.triu(m, 1).sum()),
        "over_2.5": float(m[tot > 2.5].sum()), "btts": float(m[1:, 1:].sum()),
        "exp_home": float((m.sum(axis=1) * g).sum()), "exp_away": float((m.sum(axis=0) * g).sum()),
    }


# ══════════════════════════════════════════════════════════════════════
# Fase 2: ratings con tiros (goles esperados aproximados)
# ══════════════════════════════════════════════════════════════════════

GOALS_PER_SOT = 0.289      # calibrado con 2010–2018 en las 5 grandes (los tiros desviados ≈ 0)


def fit_rates(home: List[str], away: List[str], yh: np.ndarray, ya: np.ndarray, days_ago: np.ndarray,
              params: DCParams = DCParams(), init: Optional[DCFit] = None, rho: float = 0.0) -> DCFit:
    """Ataque, defensa, media y localía con objetivos continuos (cuasi-Poisson: y·log λ − λ), para
    mezclar goles reales con goles esperados por tiros. Sin corrección de marcadores bajos: el `rho`
    se toma del ajuste con goles."""
    m = days_ago <= params.window_days
    home = [h for h, k in zip(home, m) if k]
    away = [a for a, k in zip(away, m) if k]
    yh, ya, w = yh[m], ya[m], np.exp(-params.xi * days_ago[m])
    teams = sorted(set(home) | set(away))
    idx = {t: k for k, t in enumerate(teams)}
    hi = np.array([idx[t] for t in home])
    ai = np.array([idx[t] for t in away])
    n = len(teams)

    def nll_grad(p):
        att, dfn, mu, hfa = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]
        lh = np.exp(mu + hfa + att[hi] - dfn[ai])
        la = np.exp(mu + att[ai] - dfn[hi])
        f = -np.sum(w * (yh * np.log(lh) - lh + ya * np.log(la) - la)) + params.ridge * (att @ att + dfn @ dfn)
        rh, ra = w * (lh - yh), w * (la - ya)          # derivada respecto a log λ
        g = np.zeros_like(p)
        g[:n] = np.bincount(hi, rh, n) + np.bincount(ai, ra, n) + 2 * params.ridge * att
        g[n:2 * n] = -np.bincount(ai, rh, n) - np.bincount(hi, ra, n) + 2 * params.ridge * dfn
        g[2 * n] = rh.sum() + ra.sum()
        g[2 * n + 1] = rh.sum()
        return f, g

    p0 = np.zeros(2 * n + 2)
    p0[2 * n] = np.log(max(np.average(np.r_[yh, ya], weights=np.r_[w, w]), 0.3))
    p0[2 * n + 1] = 0.25
    if init is not None:
        prev = {t: k for k, t in enumerate(init.teams)}
        for t, k in idx.items():
            if t in prev:
                p0[k], p0[n + k] = init.attack[prev[t]], init.defense[prev[t]]
        p0[2 * n], p0[2 * n + 1] = init.mu, init.home
    res = minimize(nll_grad, p0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    p = res.x
    return DCFit(teams=teams, attack=p[:n], defense=p[n:2 * n], mu=float(p[2 * n]), home=float(p[2 * n + 1]), rho=rho)
