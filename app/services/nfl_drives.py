"""
OmniPicks — NFL v3, fase 4: el marcador serie por serie.

Cada equipo tiene ~9 series por partido y cada una termina en touchdown, gol de campo o nada
(despeje, pérdida de balón, fin de mitad…). Con lo jugado antes de cada semana se estima, ajustado
por rival:
  · probabilidad de TD y de gol de campo por serie (ataque contra defensa);
  · series por partido (ritmo de los dos equipos);
  · puntos que no salen del ataque (TD de defensa o de equipos especiales, safeties).

Con eso se arma la distribución completa del marcador de cada equipo (convolución serie por serie),
y de ahí la del margen y la del total. Los números clave (3, 7, 10, 14…) salen solos.
"""
import logging
import os
from typing import Dict, Iterable, Tuple

import numpy as np
import polars as pl

from app.services.nfl_history import CACHE_DIR, normalize_teams
from app.services.nfl_ratings import RatingParams, time_index

logger = logging.getLogger(__name__)

MAX_PTS = 90


def _drive_path(season: int) -> str:
    return os.path.join(CACHE_DIR, f"drives_{season}.parquet")


def load_drives(seasons: Iterable[int], refresh_current: bool = False) -> pl.DataFrame:
    """Una fila por equipo y partido: series, TD, goles de campo y puntos cedidos por su ataque
    (TD rival tras pérdida de balón, safety)."""
    import nflreadpy as nfl
    from app.services.nfl_history import current_season
    os.makedirs(CACHE_DIR, exist_ok=True)
    cur = current_season()
    frames = []
    for s in seasons:
        path = _drive_path(s)
        if os.path.exists(path) and not (refresh_current and s == cur):
            frames.append(pl.read_parquet(path))
            continue
        p = None
        for attempt in range(3):                       # GitHub a veces corta la descarga
            try:
                p = nfl.load_pbp([s])
                break
            except Exception as e:
                logger.warning(f"nflverse: falló la descarga de {s} (intento {attempt + 1}: {type(e).__name__})")
        if p is None:
            continue
        p = p.filter(pl.col("fixed_drive").is_not_null() & pl.col("posteam").is_not_null())
        d = p.group_by(["game_id", "season", "week", "posteam", "defteam", "home_team", "fixed_drive"]).agg(
            res=pl.col("fixed_drive_result").drop_nulls().last())
        g = d.group_by(["game_id", "season", "week", "posteam", "defteam", "home_team"]).agg(
            drives=pl.len(), td=(pl.col("res") == "Touchdown").sum(), fg=(pl.col("res") == "Field goal").sum(),
            opp_td=(pl.col("res") == "Opp touchdown").sum(), safety=(pl.col("res") == "Safety").sum())
        g = g.with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
        g.write_parquet(path)
        frames.append(g)
    return normalize_teams(pl.concat(frames, how="diagonal_relaxed"), cols=("posteam", "defteam", "home_team"))


class DriveRates:
    """Probabilidad por serie (TD y FG) ajustada por rival, y series por partido de cada equipo,
    antes de cada semana. Mismo esquema de los otros ratings: descuento por antigüedad + ridge."""

    def __init__(self, drives: pl.DataFrame, params: RatingParams = RatingParams(half_life=16, offseason_gap=12),
                 ridge: float = 60.0):
        self.p, self.ridge = params, ridge
        self.teams = sorted(set(drives["posteam"].to_list()) | set(drives["defteam"].to_list()))
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.t = time_index(drives["season"].to_numpy().astype(float), drives["week"].to_numpy().astype(float),
                            params.offseason_gap)
        self.off = np.array([self.idx[t] for t in drives["posteam"].to_list()])
        self.dfn = np.array([self.idx[t] for t in drives["defteam"].to_list()])
        self.home = (drives["posteam"] == drives["home_team"]).to_numpy().astype(float)
        self.n = drives["drives"].to_numpy().astype(float)
        self.td = drives["td"].to_numpy().astype(float)
        self.fg = drives["fg"].to_numpy().astype(float)
        self.extra = (drives["opp_td"].to_numpy() * 7.0 + drives["safety"].to_numpy() * 2.0).astype(float)

    def _fit(self, y_num, mask, w):
        nt = len(self.teams)
        n = self.n[mask]
        y = y_num[mask] / np.maximum(n, 1)
        ww = w * n
        X = np.zeros((mask.sum(), 2 + 2 * nt))
        X[:, 0] = 1.0
        X[:, 1] = self.home[mask]
        X[np.arange(mask.sum()), 2 + self.off[mask]] = 1.0
        X[np.arange(mask.sum()), 2 + nt + self.dfn[mask]] = -1.0
        A = (X * ww[:, None]).T @ X
        A[np.arange(2, A.shape[0]), np.arange(2, A.shape[0])] += self.ridge
        b = np.linalg.solve(A + 1e-9 * np.eye(A.shape[0]), (X * ww[:, None]).T @ y)
        return b

    def at(self, season: int, week: int) -> Dict:
        p = self.p
        now = time_index(np.array([season]), np.array([week]), p.offseason_gap)[0]
        age = now - self.t
        mask = (age > 0) & (age <= p.window * p.half_life)
        w = 0.5 ** (age[mask] / p.half_life)
        nt = len(self.teams)
        bt, bf = self._fit(self.td, mask, w), self._fit(self.fg, mask, w)
        # Series por partido: media con descuento de las series del equipo y de sus rivales
        tot_d = np.bincount(self.off[mask], weights=w * self.n[mask], minlength=nt) + \
            np.bincount(self.dfn[mask], weights=w * self.n[mask], minlength=nt)
        cnt = np.bincount(self.off[mask], weights=w, minlength=nt) + np.bincount(self.dfn[mask], weights=w, minlength=nt)
        league_d = float(np.sum(w * self.n[mask]) / max(np.sum(w), 1e-9))
        pace = (tot_d + 4 * league_d) / (cnt + 4)
        extra = float(np.sum(w * self.extra[mask]) / max(np.sum(w), 1e-9))     # puntos por partido para el rival
        return {"td": bt, "fg": bf, "pace": {t: float(pace[i]) for t, i in self.idx.items()},
                "league_drives": league_d, "extra": extra}

    def game(self, r: Dict, home: str, away: str, neutral: bool = False) -> Dict:
        """Probabilidades por serie de cada equipo y series esperadas del partido."""
        nt = len(self.teams)
        i = self.idx
        out = {}
        for side, a, b in (("home", home, away), ("away", away, home)):
            at_home = 0.0 if (neutral or side == "away") else 1.0
            ptd = r["td"][0] + r["td"][1] * at_home + r["td"][2 + i.get(a, 0)] - r["td"][2 + nt + i.get(b, 0)]
            pfg = r["fg"][0] + r["fg"][1] * at_home + r["fg"][2 + i.get(a, 0)] - r["fg"][2 + nt + i.get(b, 0)]
            out[side] = (float(np.clip(ptd, 0.05, 0.6)), float(np.clip(pfg, 0.05, 0.4)))
        out["drives"] = (r["pace"].get(home, r["league_drives"]) + r["pace"].get(away, r["league_drives"])) / 2.0
        out["extra"] = r["extra"]
        return out


# ══════════════════════════════════════════════════════════════════════
# Distribución del marcador
# ══════════════════════════════════════════════════════════════════════

def team_points_pmf(ptd: float, pfg: float, drives: float, extra_pts: float, drive_sd: float = 1.1) -> np.ndarray:
    """Distribución de los puntos de un equipo (0..MAX_PTS): series con TD (7; 6 u 8 a veces) o FG (3),
    número de series alrededor de `drives`, y puntos que no salen del ataque (TD de defensa ~7)."""
    per = np.zeros(9)
    per[0] = 1 - ptd - pfg
    per[3] = pfg
    per[7] = ptd * 0.94
    per[6] = ptd * 0.03
    per[8] = ptd * 0.03
    lo, hi = max(1, int(np.floor(drives - 3 * drive_sd))), int(np.ceil(drives + 3 * drive_sd))
    ns = np.arange(lo, hi + 1)
    wn = np.exp(-0.5 * ((ns - drives) / drive_sd) ** 2)
    wn /= wn.sum()
    out = np.zeros(MAX_PTS + 1)
    cur = np.zeros(MAX_PTS + 1); cur[0] = 1.0
    k = 0
    for n, wgt in zip(ns, wn):
        while k < n:
            cur = np.convolve(cur, per)[: MAX_PTS + 1]
            k += 1
        out += wgt * cur
    # Puntos extra (TD de defensa/equipos especiales): Poisson de "touchdowns" de 7
    lam = max(extra_pts, 0.0) / 7.0
    ex = np.zeros(MAX_PTS + 1)
    pk = np.exp(-lam)
    for j in range(0, 4):
        ex[7 * j] += pk
        pk *= lam / (j + 1)
    out = np.convolve(out, ex)[: MAX_PTS + 1]
    return out / out.sum()


def game_distributions(home_pmf: np.ndarray, away_pmf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(margen del local de −MAX..MAX, total 0..2·MAX), suponiendo independencia entre equipos."""
    margin = np.convolve(home_pmf, away_pmf[::-1])          # índice MAX_PTS = margen 0
    total = np.convolve(home_pmf, away_pmf)
    return margin, total


def scale_to_points(ptd: float, pfg: float, drives: float, target_pts: float, extra: float) -> Tuple[float, float]:
    """Escala las probabilidades por serie para que los puntos esperados sean `target_pts`
    (para usar la forma de la simulación con las medias de otro modelo)."""
    base = drives * (6.97 * ptd + 3 * pfg) + extra
    f = max(0.2, (target_pts - extra) / max(base - extra, 1e-6))
    return float(np.clip(ptd * f, 0.02, 0.75)), float(np.clip(pfg * f ** 0.5, 0.02, 0.45))
