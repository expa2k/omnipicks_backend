"""
OmniPicks — NFL: rating de cada QB y ajuste cuando cambia el titular.

Rating del QB = su EPA por dropback (pases, sacks y scrambles) por encima de la media de la liga,
con los partidos recientes pesando más y encogido hacia un nivel de suplente (prior_delta) según
cuántos dropbacks tenga: un QB nuevo o con pocas jugadas empieza como suplente.

El rating de pase del equipo ya trae dentro al QB que jugó esas semanas. Por eso el ajuste del
partido es la diferencia entre el titular de hoy y el QB "promedio" con el que el equipo sumó
sus números recientes:

    ajuste = (rating del titular − rating medio de los QB recientes del equipo) × dropbacks

Si el titular es el de siempre, el ajuste es ~0; si entra el suplente, el ajuste es negativo.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import polars as pl

from app.services.nfl_history import normalize_teams
from app.services.nfl_ratings import time_index


@dataclass
class QBParams:
    half_life: float = 64.0       # semanas (la calidad de un QB cambia más despacio que la del equipo)
    offseason_gap: float = 8.0
    prior_n: float = 100.0        # dropbacks "virtuales" de nivel suplente
    prior_delta: float = -0.10    # EPA por dropback de un suplente respecto a la media
    team_half_life: float = 16.0  # para el QB medio con el que el equipo sumó sus números
    window: float = 4.0


def qb_games(pbp: pl.DataFrame, wp_lo: float = 0.05, wp_hi: float = 0.95) -> pl.DataFrame:
    """Un renglón por QB y partido: dropbacks y EPA (sin garbage time, igual que team_games)."""
    pbp = normalize_teams(pbp)
    d = pbp.filter((pl.col("qb_dropback") == 1) & pl.col("id").is_not_null() & pl.col("qb_epa").is_not_null()
                   & pl.col("wp").is_between(wp_lo, wp_hi) & (pl.col("half_seconds_remaining") > 120))
    return d.group_by(["game_id", "season", "week", "posteam", "defteam", "id"]).agg(
        epa=pl.col("qb_epa").sum(), n=pl.len(), name=pl.col("name").first()).sort(["season", "week", "game_id"])


@dataclass
class QBWeek:
    rating: Dict[str, float]        # QB → EPA por dropback sobre la media
    dropbacks: Dict[str, float]     # QB → dropbacks recientes (con descuento)
    team_qb: Dict[str, float]       # equipo → rating medio de sus QB recientes
    prior_delta: float

    def of(self, qb: Optional[str]) -> float:
        return self.rating.get(qb, self.prior_delta) if qb else self.prior_delta

    def adjustment(self, team: str, starter: Optional[str]) -> float:
        """EPA por dropback que gana (o pierde) el equipo hoy por su titular."""
        return self.of(starter) - self.team_qb.get(team, self.of(starter))


class QBRatingEngine:
    def __init__(self, qbg: pl.DataFrame, params: QBParams = QBParams()):
        self.p = params
        self.qb = qbg["id"].to_list()
        self.team = qbg["posteam"].to_list()
        s, w = qbg["season"].to_numpy().astype(float), qbg["week"].to_numpy().astype(float)
        self.t_q = time_index(s, w, params.offseason_gap)
        self.t_t = time_index(s, w, 12.0)
        self.epa = qbg["epa"].to_numpy()
        self.n = qbg["n"].to_numpy().astype(float)
        self.qbs = sorted(set(self.qb))
        self.qi = np.array([{q: i for i, q in enumerate(self.qbs)}[q] for q in self.qb])
        self.teams = sorted(set(self.team))
        self.ti = np.array([{t: i for i, t in enumerate(self.teams)}[t] for t in self.team])

    def at(self, season: int, week: int) -> QBWeek:
        p = self.p
        now_q = time_index(np.array([season]), np.array([week]), p.offseason_gap)[0]
        age = now_q - self.t_q
        m = (age > 0) & (age <= p.window * p.half_life)
        dec = 0.5 ** (age[m] / p.half_life)
        nw, ew = self.n[m] * dec, self.epa[m] * dec
        league = ew.sum() / max(nw.sum(), 1e-9)
        nq = len(self.qbs)
        S_n = np.bincount(self.qi[m], weights=nw, minlength=nq)
        S_e = np.bincount(self.qi[m], weights=ew, minlength=nq)
        r = (S_e - league * S_n + p.prior_n * p.prior_delta) / (S_n + p.prior_n)
        seen = S_n > 0
        rating = {q: float(r[i]) for i, q in enumerate(self.qbs) if seen[i]}
        drop = {q: float(S_n[i]) for i, q in enumerate(self.qbs) if seen[i]}

        # QB medio de cada equipo en sus semanas recientes (vida media del rating de equipo)
        now_t = time_index(np.array([season]), np.array([week]), 12.0)[0]
        age_t = now_t - self.t_t
        mt = (age_t > 0) & (age_t <= p.window * p.team_half_life)
        wt = self.n[mt] * 0.5 ** (age_t[mt] / p.team_half_life)
        qr = np.where(seen[self.qi[mt]], r[self.qi[mt]], p.prior_delta)
        num = np.bincount(self.ti[mt], weights=wt * qr, minlength=len(self.teams))
        den = np.bincount(self.ti[mt], weights=wt, minlength=len(self.teams))
        team_qb = {t: float(num[i] / den[i]) for i, t in enumerate(self.teams) if den[i] > 0}
        return QBWeek(rating=rating, dropbacks=drop, team_qb=team_qb, prior_delta=p.prior_delta)


def qb_margin_features(w: QBWeek, home: str, away: str, home_qb: Optional[str], away_qb: Optional[str],
                       dropbacks: float = 38.0) -> Tuple[float, float]:
    """(ajuste por titulares, diferencia de calidad de los titulares), local − visitante, en EPA por partido."""
    adj = (w.adjustment(home, home_qb) - w.adjustment(away, away_qb)) * dropbacks
    raw = (w.of(home_qb) - w.of(away_qb)) * dropbacks
    return adj, raw
