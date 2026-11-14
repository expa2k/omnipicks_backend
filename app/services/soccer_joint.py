"""
OmniPicks — Fútbol v2, fase 3: un solo modelo para ligas y copas europeas.

    λ = exp(μ_competición + localía_competición + ataque[eq] + A[liga del eq] − defensa[rival] − D[liga del rival])

  · ataque/defensa de cada equipo: se aprenden sobre todo en su liga;
  · A y D de cada liga (y de un grupo "otras ligas" para los clubes de ligas que no tenemos):
    se aprenden de los cruces en Champions y Europa League. Así un 2º lugar de la Eredivisie no
    vale lo mismo que un 2º de la Premier;
  · dentro de una liga, A − D se confunde con μ de esa liga, así que no cambia las predicciones
    de liga: solo ordena a las ligas entre sí.

Se ajusta con cuasi-verosimilitud Poisson (objetivo = 70% goles + 30% goles por tiros a puerta en las
ligas que tienen tiros; solo goles en copas) y peso por antigüedad.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy.optimize import minimize

from app.services.soccer_dc import GOALS_PER_SOT, score_matrix

EU_LEAGUES = ["Premier League", "Championship", "La Liga", "Serie A", "Bundesliga", "Ligue 1", "Eredivisie",
              "Primeira Liga"]
OTHER = "Otras ligas"


@dataclass
class JointParams:
    xi: float = 0.002
    ridge_team: float = 2.0
    ridge_league: float = 5.0
    window_days: int = 1100
    shots_weight: float = 0.3


@dataclass
class JointFit:
    teams: List[str]
    att: np.ndarray
    dfn: np.ndarray
    leagues: List[str]
    A: np.ndarray
    D: np.ndarray
    comps: List[str]
    mu: np.ndarray
    home: np.ndarray
    team_league: Dict[str, str]
    rho: float = -0.05

    def rates(self, comp: str, home: str, away: str, neutral: bool = False) -> Tuple[float, float]:
        ti = {t: k for k, t in enumerate(self.teams)}
        li = {l: k for k, l in enumerate(self.leagues)}
        ci = {c: k for k, c in enumerate(self.comps)}
        def strength(t):
            k = ti.get(t)
            lg = self.team_league.get(t, OTHER)
            j = li.get(lg, li.get(OTHER))
            a = (self.att[k] if k is not None else 0.0) + (self.A[j] if j is not None else 0.0)
            d = (self.dfn[k] if k is not None else 0.0) + (self.D[j] if j is not None else 0.0)
            return a, d
        ah, dh = strength(home)
        aa, da = strength(away)
        c = ci[comp]
        lh = np.exp(self.mu[c] + (0.0 if neutral else self.home[c]) + ah - da)
        la = np.exp(self.mu[c] + aa - dh)
        return float(lh), float(la)

    def score_matrix(self, comp, home, away, neutral=False):
        return score_matrix(*self.rates(comp, home, away, neutral), self.rho)


def build_table(leagues_df: pl.DataFrame, cups_df: pl.DataFrame, name_map: Dict[str, str],
                shots_weight: float = 0.3) -> pl.DataFrame:
    """Una tabla de partidos (ligas + copas) con objetivo de goles para el ajuste y los goles reales."""
    lg = leagues_df.filter(pl.col("league").is_in(EU_LEAGUES) & pl.col("FTHG").is_not_null() & pl.col("date").is_not_null())
    w = 1 - shots_weight
    has = pl.col("HST").is_not_null() & pl.col("AST").is_not_null()
    lg = lg.select(
        pl.col("date"), pl.col("league").alias("comp"), pl.col("HomeTeam").alias("home"), pl.col("AwayTeam").alias("away"),
        pl.col("FTHG").alias("hg"), pl.col("FTAG").alias("ag"), pl.lit(False).alias("neutral"),
        pl.when(has).then(w * pl.col("FTHG") + (1 - w) * GOALS_PER_SOT * pl.col("HST")).otherwise(pl.col("FTHG")).alias("yh"),
        pl.when(has).then(w * pl.col("FTAG") + (1 - w) * GOALS_PER_SOT * pl.col("AST")).otherwise(pl.col("FTAG")).alias("ya"),
        pl.col("season"))
    cu = cups_df.filter(pl.col("completed") & pl.col("hg").is_not_null()).select(
        pl.col("date"), pl.col("competition").alias("comp"),
        pl.col("home").replace_strict(name_map, default=None).fill_null(pl.col("home") + " ·").alias("home"),
        pl.col("away").replace_strict(name_map, default=None).fill_null(pl.col("away") + " ·").alias("away"),
        pl.col("hg"), pl.col("ag"), pl.col("neutral"), pl.col("hg").alias("yh"), pl.col("ag").alias("ya"), pl.col("season"))
    return pl.concat([lg, cu], how="diagonal_relaxed").sort("date")


class JointModel:
    """Arreglos listos para reajustar en cualquier fecha (solo con partidos anteriores)."""

    def __init__(self, table: pl.DataFrame, params: JointParams = JointParams()):
        self.p = params
        self.t = table
        self.dates = table["date"].to_numpy().astype("datetime64[D]")
        self.comp = table["comp"].to_list()
        self.home = table["home"].to_list()
        self.away = table["away"].to_list()
        self.yh = table["yh"].to_numpy().astype(float)
        self.ya = table["ya"].to_numpy().astype(float)
        self.neutral = table["neutral"].to_numpy().astype(bool)

    def team_leagues(self, before: np.datetime64) -> Dict[str, str]:
        """Liga de cada equipo: la última liga en la que jugó antes de la fecha (sino: otras ligas)."""
        out = {}
        for c, h, a, d in zip(self.comp, self.home, self.away, self.dates):
            if d >= before:
                break
            if c in EU_LEAGUES:
                out[h] = c
                out[a] = c
        return out

    def fit(self, ref: np.datetime64, init: Optional[JointFit] = None) -> JointFit:
        p = self.p
        days = (ref - self.dates).astype(int)
        m = (days > 0) & (days <= p.window_days)
        idx = np.where(m)[0]
        tl = self.team_leagues(ref)
        teams = sorted(set(self.home[i] for i in idx) | set(self.away[i] for i in idx))
        leagues = EU_LEAGUES + [OTHER]
        comps = sorted(set(self.comp[i] for i in idx))
        ti = {t: k for k, t in enumerate(teams)}
        li = {l: k for k, l in enumerate(leagues)}
        ci = {c: k for k, c in enumerate(comps)}
        hi = np.array([ti[self.home[i]] for i in idx]); ai = np.array([ti[self.away[i]] for i in idx])
        hl = np.array([li[tl.get(self.home[i], OTHER)] for i in idx]); al = np.array([li[tl.get(self.away[i], OTHER)] for i in idx])
        cc = np.array([ci[self.comp[i]] for i in idx])
        hn = (~self.neutral[idx]).astype(float)
        yh, ya = self.yh[idx], self.ya[idx]
        w = np.exp(-p.xi * days[idx])
        nT, nL, nC = len(teams), len(leagues), len(comps)
        sl = [slice(0, nT), slice(nT, 2 * nT), slice(2 * nT, 2 * nT + nL), slice(2 * nT + nL, 2 * nT + 2 * nL),
              slice(2 * nT + 2 * nL, 2 * nT + 2 * nL + nC), slice(2 * nT + 2 * nL + nC, 2 * nT + 2 * nL + 2 * nC)]

        def f(x):
            att, dfn, A, D, mu, hm = (x[s] for s in sl)
            eh = mu[cc] + hm[cc] * hn + att[hi] + A[hl] - dfn[ai] - D[al]
            ea = mu[cc] + att[ai] + A[al] - dfn[hi] - D[hl]
            lh, la = np.exp(eh), np.exp(ea)
            val = -np.sum(w * (yh * eh - lh + ya * ea - la)) + p.ridge_team * (att @ att + dfn @ dfn) \
                + p.ridge_league * (A @ A + D @ D)
            rh, ra = w * (lh - yh), w * (la - ya)
            g = np.zeros_like(x)
            g[sl[0]] = np.bincount(hi, rh, nT) + np.bincount(ai, ra, nT) + 2 * p.ridge_team * att
            g[sl[1]] = -np.bincount(ai, rh, nT) - np.bincount(hi, ra, nT) + 2 * p.ridge_team * dfn
            g[sl[2]] = np.bincount(hl, rh, nL) + np.bincount(al, ra, nL) + 2 * p.ridge_league * A
            g[sl[3]] = -np.bincount(al, rh, nL) - np.bincount(hl, ra, nL) + 2 * p.ridge_league * D
            g[sl[4]] = np.bincount(cc, rh + ra, nC)
            g[sl[5]] = np.bincount(cc, rh * hn, nC)
            return val, g

        x0 = np.zeros(2 * nT + 2 * nL + 2 * nC)
        x0[sl[4]] = np.log(1.35)
        x0[sl[5]] = 0.25
        if init is not None:
            pt = {t: k for k, t in enumerate(init.teams)}
            for t, k in ti.items():
                if t in pt:
                    x0[k], x0[nT + k] = init.att[pt[t]], init.dfn[pt[t]]
            x0[sl[2]], x0[sl[3]] = init.A, init.D
            pc = {c: k for k, c in enumerate(init.comps)}
            for c, k in ci.items():
                if c in pc:
                    x0[2 * nT + 2 * nL + k] = init.mu[pc[c]]
                    x0[2 * nT + 2 * nL + nC + k] = init.home[pc[c]]
        res = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 800})
        x = res.x
        return JointFit(teams=teams, att=x[sl[0]], dfn=x[sl[1]], leagues=leagues, A=x[sl[2]], D=x[sl[3]],
                        comps=comps, mu=x[sl[4]], home=x[sl[5]], team_league=tl)
