"""
OmniPicks — NFL: rating de cada equipo (ataque y defensa, en pase y en carrera).

Para cada tipo de jugada se ajusta, con todo lo jugado antes de la semana a predecir:

    EPA por jugada del ataque = media de la liga + localía + ataque[equipo] − defensa[rival]

con mínimos cuadrados ponderados y encogidos hacia la media (ridge):
  · cada partido pesa por su número de jugadas y por su antigüedad (vida media en semanas);
  · la pausa entre temporadas cuenta como semanas extra, así lo del año pasado pesa menos
    (en la NFL los equipos cambian mucho de un año a otro);
  · el ridge evita que un equipo con pocas jugadas quede con un rating extremo.

Así el rating ya viene ajustado por rival: no vale lo mismo sumar EPA contra la mejor defensa
que contra la peor.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import polars as pl

KINDS = ("pass", "rush")
WEEKS_PER_SEASON = 22          # 18 de temporada regular + playoffs


@dataclass
class RatingParams:
    half_life: float = 10.0     # semanas
    offseason_gap: float = 12.0 # semanas extra entre temporadas
    ridge: float = 600.0        # jugadas "virtuales" de nivel de liga por equipo
    window: float = 4.0         # se usan partidos hasta window × vida media hacia atrás


@dataclass
class WeekRatings:
    """Ratings vigentes antes de una semana. EPA por jugada, positivo = mejor (también en defensa)."""
    mu: Dict[str, float]
    home: Dict[str, float]
    off: Dict[str, Dict[str, float]]
    deff: Dict[str, Dict[str, float]]
    pass_rate: Dict[str, float] = field(default_factory=dict)   # % de jugadas de pase del equipo

    def expected_epa(self, offense: str, defense: str, at_home: bool) -> float:
        """EPA por jugada esperado del ataque de `offense` contra la defensa de `defense`."""
        pr = self.pass_rate.get(offense, 0.58)
        out = 0.0
        for k, share in (("pass", pr), ("rush", 1.0 - pr)):
            e = self.mu[k] + (self.home[k] if at_home else 0.0) \
                + self.off[k].get(offense, 0.0) - self.deff[k].get(defense, 0.0)
            out += share * e
        return out

    def net(self, team: str) -> float:
        """Rating total del equipo (EPA por jugada a favor menos en contra, contra un rival medio)."""
        pr = self.pass_rate.get(team, 0.58)
        return sum(s * (self.off[k].get(team, 0.0) + self.deff[k].get(team, 0.0))
                   for k, s in (("pass", pr), ("rush", 1.0 - pr)))


def time_index(season: np.ndarray, week: np.ndarray, gap: float) -> np.ndarray:
    return (season - 2000) * (WEEKS_PER_SEASON + gap) + week


class NFLRatingEngine:
    """Ratings point-in-time a partir de la tabla de team_games (un renglón por equipo y partido)."""

    def __init__(self, team_games: pl.DataFrame, params: RatingParams = RatingParams(), neutral_games=(),
                 value: str = "epa"):
        """value: qué se mide por jugada. "epa" (EPA), "epant" (EPA sin pérdidas de balón) o
        "succ" (tasa de éxito)."""
        self.p = params
        tg = team_games
        self.teams: List[str] = sorted(set(tg["posteam"].to_list()) | set(tg["defteam"].to_list()))
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.season = tg["season"].to_numpy().astype(float)
        self.week = tg["week"].to_numpy().astype(float)
        self.t = time_index(self.season, self.week, params.offseason_gap)
        self.off_i = np.array([self.idx[t] for t in tg["posteam"].to_list()])
        self.def_i = np.array([self.idx[t] for t in tg["defteam"].to_list()])
        neutral = set(neutral_games)
        self.home = np.array([0.0 if g in neutral else float(h)
                              for g, h in zip(tg["game_id"].to_list(), tg["is_home"].to_list())])
        self.epa = {k: tg[f"{value}_{k}"].to_numpy() for k in KINDS}
        self.n = {k: tg[f"n_{k}"].to_numpy() for k in KINDS}

    def at(self, season: int, week: int) -> WeekRatings:
        """Ratings con todo lo jugado antes de (season, week)."""
        p = self.p
        t_now = time_index(np.array([season]), np.array([week]), p.offseason_gap)[0]
        age = t_now - self.t
        mask = (age > 0) & (age <= p.window * p.half_life)
        decay = 0.5 ** (age[mask] / p.half_life)
        nt = len(self.teams)
        mu, home, off, deff = {}, {}, {}, {}
        for k in KINDS:
            n = self.n[k][mask]
            ok = n > 0
            w = (n * decay)[ok]
            y = self.epa[k][mask][ok] / n[ok]
            oi, di, hh = self.off_i[mask][ok], self.def_i[mask][ok], self.home[mask][ok]
            # Columnas: mu, localía, ataque (nt), defensa (nt)
            m = 2 + 2 * nt
            X = np.zeros((len(y), m))
            X[:, 0] = 1.0
            X[:, 1] = hh
            X[np.arange(len(y)), 2 + oi] = 1.0
            X[np.arange(len(y)), 2 + nt + di] = -1.0
            A = (X * w[:, None]).T @ X
            A[np.arange(2, m), np.arange(2, m)] += p.ridge
            b = (X * w[:, None]).T @ y
            beta = np.linalg.solve(A + 1e-9 * np.eye(m), b)
            mu[k], home[k] = float(beta[0]), float(beta[1])
            off[k] = {t: float(beta[2 + i]) for t, i in self.idx.items()}
            deff[k] = {t: float(beta[2 + nt + i]) for t, i in self.idx.items()}
        # Mezcla de pase del equipo (reciente)
        npass, nrush = self.n["pass"][mask] * decay, self.n["rush"][mask] * decay
        tot_p = np.bincount(self.off_i[mask], weights=npass, minlength=nt)
        tot_r = np.bincount(self.off_i[mask], weights=nrush, minlength=nt)
        league = tot_p.sum() / max(tot_p.sum() + tot_r.sum(), 1e-9)
        pr = (tot_p + 100 * league) / (tot_p + tot_r + 100)
        pass_rate = {t: float(pr[i]) for t, i in self.idx.items()}
        return WeekRatings(mu=mu, home=home, off=off, deff=deff, pass_rate=pass_rate)


def expected_margin_epa(r: WeekRatings, home: str, away: str, neutral: bool = False,
                        plays: float = 62.0) -> Tuple[float, float]:
    """Diferencia de EPA esperada del partido (local − visitante) y suma de EPA esperada (para el total),
    escaladas a un partido de `plays` jugadas por equipo. Se convierten a puntos con una regresión."""
    eh = r.expected_epa(home, away, at_home=not neutral)
    ea = r.expected_epa(away, home, at_home=False)
    return (eh - ea) * plays, (eh + ea) * plays


class PointsRatingEngine:
    """Rating por marcador ajustado por rival (tipo SRS), con la misma ponderación por antigüedad:

        margen del partido (local − visitante) = localía + fuerza[local] − fuerza[visitante]

    Complementa al EPA: el marcador incluye equipos especiales, pérdidas de balón y zona roja.
    El margen se recorta a ±`cap` puntos para que una paliza no pese de más."""

    def __init__(self, games: pl.DataFrame, params: RatingParams = RatingParams(), cap: float = 24.0):
        self.p = params
        self.teams = sorted(set(games["home_team"].to_list()) | set(games["away_team"].to_list()))
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.t = time_index(games["season"].to_numpy().astype(float), games["week"].to_numpy().astype(float),
                            params.offseason_gap)
        self.h = np.array([self.idx[t] for t in games["home_team"].to_list()])
        self.a = np.array([self.idx[t] for t in games["away_team"].to_list()])
        self.neutral = (games["location"].to_numpy() == "Neutral").astype(float)
        self.y = np.clip(games["result"].to_numpy().astype(float), -cap, cap)

    def at(self, season: int, week: int, ridge: float = 2.0) -> Tuple[float, Dict[str, float]]:
        """(localía, fuerza por equipo en puntos) con los partidos antes de (season, week).
        `ridge` en partidos virtuales de nivel de liga."""
        p = self.p
        t_now = time_index(np.array([season]), np.array([week]), p.offseason_gap)[0]
        age = t_now - self.t
        mask = (age > 0) & (age <= p.window * p.half_life)
        w = 0.5 ** (age[mask] / p.half_life)
        nt = len(self.teams)
        n = int(mask.sum())
        X = np.zeros((n, 1 + nt))
        X[:, 0] = 1.0 - self.neutral[mask]
        X[np.arange(n), 1 + self.h[mask]] = 1.0
        X[np.arange(n), 1 + self.a[mask]] = -1.0
        A = (X * w[:, None]).T @ X
        A[np.arange(1, 1 + nt), np.arange(1, 1 + nt)] += ridge
        beta = np.linalg.solve(A + 1e-9 * np.eye(1 + nt), (X * w[:, None]).T @ self.y[mask])
        return float(beta[0]), {t: float(beta[1 + i]) for t, i in self.idx.items()}


class ScoringRatingEngine:
    """Rating de anotación (para el total): por equipo y partido,

        puntos anotados = media + localía + ataque[equipo] − defensa[rival]

    con la misma ponderación por antigüedad y ridge. La media se mueve con los años, así que sigue
    los cambios de anotación de la liga."""

    def __init__(self, games: pl.DataFrame, params: RatingParams = RatingParams()):
        self.p = params
        self.teams = sorted(set(games["home_team"].to_list()) | set(games["away_team"].to_list()))
        idx = {t: i for i, t in enumerate(self.teams)}
        self.idx = idx
        t = time_index(games["season"].to_numpy().astype(float), games["week"].to_numpy().astype(float),
                       params.offseason_gap)
        h = np.array([idx[x] for x in games["home_team"].to_list()])
        a = np.array([idx[x] for x in games["away_team"].to_list()])
        home = (games["location"].to_numpy() != "Neutral").astype(float)
        # Dos renglones por partido: el local anotando y el visitante anotando
        self.t = np.r_[t, t]
        self.off = np.r_[h, a]
        self.dfn = np.r_[a, h]
        self.home = np.r_[home, np.zeros(len(home))]
        self.y = np.r_[games["home_score"].to_numpy().astype(float), games["away_score"].to_numpy().astype(float)]

    def at(self, season: int, week: int, ridge: float = 4.0):
        p = self.p
        t_now = time_index(np.array([season]), np.array([week]), p.offseason_gap)[0]
        age = t_now - self.t
        mask = (age > 0) & (age <= p.window * p.half_life)
        w = 0.5 ** (age[mask] / p.half_life)
        nt = len(self.teams)
        n = int(mask.sum())
        X = np.zeros((n, 2 + 2 * nt))
        X[:, 0] = 1.0
        X[:, 1] = self.home[mask]
        X[np.arange(n), 2 + self.off[mask]] = 1.0
        X[np.arange(n), 2 + nt + self.dfn[mask]] = -1.0
        A = (X * w[:, None]).T @ X
        A[np.arange(2, 2 + 2 * nt), np.arange(2, 2 + 2 * nt)] += ridge
        beta = np.linalg.solve(A + 1e-9 * np.eye(A.shape[0]), (X * w[:, None]).T @ self.y[mask])
        off = {t: float(beta[2 + i]) for t, i in self.idx.items()}
        dfn = {t: float(beta[2 + nt + i]) for t, i in self.idx.items()}
        return float(beta[0]), float(beta[1]), off, dfn
