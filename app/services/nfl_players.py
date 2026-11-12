"""
OmniPicks — NFL v3: valor de cada jugador (RAPM) y fuerza del equipo con sus jugadores.

RAPM (plus-minus ajustado con ridge), jugada por jugada, con la participación de nflverse (2016–):

    EPA de la jugada = Σ ataque[jugadores de ataque en el campo] − Σ defensa[jugadores de defensa]

Cada jugador recibe un valor en EPA por jugada que estuvo en el campo, ajustado por sus
compañeros y rivales. El ridge lo encoge hacia 0 (nivel de reemplazo medio) según cuántas jugadas
tenga, con un encogimiento propio por grupo de posición.

La fuerza del equipo en un partido es la suma del valor de sus jugadores ponderada por la
fracción de jugadas que se espera que estén en el campo.
"""
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse.linalg import cg

from app.services.nfl_history import CACHE_DIR, normalize_teams

logger = logging.getLogger(__name__)

FIRST_PARTICIPATION = 2016
POS_GROUP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
             "T": "OL", "G": "OL", "C": "OL", "OL": "OL", "OT": "OL", "OG": "OL",
             "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL",
             "OLB": "LB", "ILB": "LB", "MLB": "LB", "LB": "LB",
             "CB": "DB", "FS": "DB", "SS": "DB", "S": "DB", "DB": "DB"}


def _part_path(season: int) -> str:
    return os.path.join(CACHE_DIR, f"participation_{season}.parquet")


def load_participation(seasons: Iterable[int]) -> pl.DataFrame:
    """Jugadores en el campo por jugada (caché por temporada; las cerradas no cambian)."""
    import nflreadpy as nfl
    os.makedirs(CACHE_DIR, exist_ok=True)
    frames = []
    for s in seasons:
        path = _part_path(s)
        if os.path.exists(path):
            frames.append(pl.read_parquet(path))
            continue
        try:
            d = nfl.load_participation([s])
        except Exception as e:
            logger.warning(f"nflverse: sin participación de {s} ({type(e).__name__}: {e})")
            continue
        for c in ("offense_positions", "defense_positions"):     # las temporadas viejas no las traen
            if c not in d.columns:
                d = d.with_columns(pl.lit(None, dtype=pl.Utf8).alias(c))
        d = d.select(pl.col("nflverse_game_id").alias("game_id"),
                     pl.col("play_id").cast(pl.Float64).cast(pl.Int64),
                     "offense_players", "defense_players", "offense_positions", "defense_positions")
        d.write_parquet(path)
        frames.append(d)
    return pl.concat(frames) if frames else pl.DataFrame()


def play_table(pbp: pl.DataFrame, part: pl.DataFrame, wp_lo: float = 0.05, wp_hi: float = 0.95) -> pl.DataFrame:
    """Jugadas de pase o carrera con su EPA y los 22 jugadores (sin garbage time, como team_games)."""
    pbp = normalize_teams(pbp)
    plays = pbp.filter(pl.col("wp").is_between(wp_lo, wp_hi) & (pl.col("half_seconds_remaining") > 120)) \
               .select("game_id", pl.col("play_id").cast(pl.Int64), "season", "week", "posteam", "defteam", "epa")
    j = plays.join(part, on=["game_id", "play_id"], how="inner")
    j = j.with_columns(off=pl.col("offense_players").str.split(";"), dfn=pl.col("defense_players").str.split(";"),
                       offp=pl.col("offense_positions").fill_null("").str.split(";"),
                       defp=pl.col("defense_positions").fill_null("").str.split(";"))
    return j.filter((pl.col("off").list.len() == 11) & (pl.col("dfn").list.len() == 11)) \
            .select("game_id", "play_id", "season", "week", "posteam", "defteam", "epa", "off", "dfn", "offp", "defp")


@dataclass
class RapmParams:
    season_half_life: float = 1.0        # temporadas
    seasons_back: int = 3
    ridge: Dict[str, float] = field(default_factory=lambda: {"QB": 300.0, "default": 1500.0})


@dataclass
class PlayerValues:
    off: Dict[str, float]                # EPA por jugada a favor de su ataque cuando está en el campo
    dfn: Dict[str, float]                # EPA por jugada que evita su defensa (positivo = bueno)
    plays_off: Dict[str, float]
    plays_def: Dict[str, float]
    position: Dict[str, str]


def fit_rapm(plays: pl.DataFrame, as_of_season: int, params: RapmParams = RapmParams()) -> PlayerValues:
    """RAPM con las temporadas anteriores a `as_of_season` (la más reciente pesa más)."""
    sub = plays.filter((pl.col("season") < as_of_season) & (pl.col("season") >= as_of_season - params.seasons_back))
    if sub.height == 0:
        return PlayerValues({}, {}, {}, {}, {})
    w = (0.5 ** ((as_of_season - 1 - sub["season"].to_numpy()) / params.season_half_life)).astype(float)
    y = sub["epa"].to_numpy().astype(float)
    y = y - np.average(y, weights=w)
    off_lists, def_lists = sub["off"].to_list(), sub["dfn"].to_list()
    offp, defp = sub["offp"].to_list(), sub["defp"].to_list()
    idx: Dict[Tuple[str, str], int] = {}
    pos: Dict[str, str] = {}
    try:
        from app.services.nfl_qb_status import espn_to_gsis  # noqa: F401  (asegura la caché de jugadores)
        espn_to_gsis()
        pt = pl.read_parquet(os.path.join(CACHE_DIR, "players.parquet"))
        base_pos = {g: p for g, p in zip(pt["gsis_id"].to_list(), pt["position"].to_list()) if g and p}
    except Exception:
        base_pos = {}
    rows, cols, vals = [], [], []
    for i, (ol, dl, op, dp) in enumerate(zip(off_lists, def_lists, offp, defp)):
        op = op if len(op) == len(ol) else [""] * len(ol)
        dp = dp if len(dp) == len(dl) else [""] * len(dl)
        for p, ps in zip(ol, op):
            k = idx.setdefault(("o", p), len(idx)); rows.append(i); cols.append(k); vals.append(1.0)
            if ps:
                pos[p] = ps
        for p, ps in zip(dl, dp):
            k = idx.setdefault(("d", p), len(idx)); rows.append(i); cols.append(k); vals.append(-1.0)
            if ps:
                pos[p] = ps
    for (_, p) in idx:
        if p not in pos and p in base_pos:
            pos[p] = base_pos[p]
    n = len(idx)
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(y), n))
    Xw = X.multiply(w[:, None]).tocsr()
    A = (X.T @ Xw).tocsr()
    lam = np.array([params.ridge.get(POS_GROUP.get(pos.get(p, ""), ""), params.ridge["default"])
                    for (_, p), _ in sorted(idx.items(), key=lambda kv: kv[1])])
    A = A + sparse.diags(lam)
    b = Xw.T @ y
    beta, info = cg(A, b, maxiter=2000, rtol=1e-8)
    if info != 0:
        logger.warning(f"RAPM {as_of_season}: el solver no convergió del todo (info={info})")
    counts = np.asarray(abs(X).multiply(w[:, None]).sum(axis=0)).ravel()
    off, dfn, po, pd_ = {}, {}, {}, {}
    for (side, p), k in idx.items():
        if side == "o":
            off[p], po[p] = float(beta[k]), float(counts[k])
        else:
            dfn[p], pd_[p] = float(beta[k]), float(counts[k])
    return PlayerValues(off, dfn, po, pd_, pos)


def usage_from_plays(plays: pl.DataFrame) -> pl.DataFrame:
    """Fracción de las jugadas de su equipo en que estuvo cada jugador, por partido (0–1)."""
    o = plays.select("game_id", "season", "week", pl.col("posteam").alias("team"), pl.col("off").alias("p")) \
             .explode("p").with_columns(side=pl.lit("o"))
    d = plays.select("game_id", "season", "week", pl.col("defteam").alias("team"), pl.col("dfn").alias("p")) \
             .explode("p").with_columns(side=pl.lit("d"))
    both = pl.concat([o, d])
    tot = both.group_by(["game_id", "team", "side"]).agg(n_team=pl.col("p").count() / 11.0)
    cnt = both.group_by(["game_id", "season", "week", "team", "side", "p"]).agg(n=pl.len())
    return cnt.join(tot, on=["game_id", "team", "side"]).with_columns(share=pl.col("n") / pl.col("n_team"))


def team_strength(values: PlayerValues, usage: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
    """(ataque, defensa) del equipo en EPA por jugada: Σ valor × fracción de jugadas esperada."""
    off = sum(values.off.get(p, 0.0) * s for p, s in usage.get("o", {}).items())
    dfn = sum(values.dfn.get(p, 0.0) * s for p, s in usage.get("d", {}).items())
    return off, dfn


def expected_usage(use: pl.DataFrame, half_life_games: float = 3.0) -> Dict[Tuple[int, int, str], Dict[str, Dict[str, float]]]:
    """Uso esperado de cada jugador ANTES de cada partido: media con descuento de su fracción de
    jugadas en los partidos previos de la temporada con ese equipo (si no jugó un partido, cuenta 0).
    (temporada, semana, equipo) → {"o": {jugador: fracción}, "d": {...}}"""
    out = {}
    d = 0.5 ** (1.0 / half_life_games)
    for (season, team), g in use.group_by(["season", "team"]):
        weeks = sorted(g["week"].unique().to_list())
        acc: Dict[str, Dict[str, List[float]]] = {"o": {}, "d": {}}
        for w in weeks:
            out[(season, w, team)] = {side: {p: num / den for p, (num, den) in acc[side].items() if den > 0}
                                      for side in ("o", "d")}
            gw = g.filter(pl.col("week") == w)
            for side in ("o", "d"):
                for a in acc[side].values():
                    a[0] *= d
                    a[1] = a[1] * d + 1.0
                for p, sh in zip(gw.filter(pl.col("side") == side)["p"].to_list(),
                                 gw.filter(pl.col("side") == side)["share"].to_list()):
                    if p in acc[side]:
                        acc[side][p][0] += sh
                    else:
                        acc[side][p] = [sh, 1.0]
    return out
