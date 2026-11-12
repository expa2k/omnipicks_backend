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


def values_to_dict(v: "PlayerValues") -> Dict:
    return {"off": v.off, "dfn": v.dfn, "plays_off": v.plays_off, "plays_def": v.plays_def, "position": v.position}


def values_from_dict(d: Dict) -> "PlayerValues":
    return PlayerValues(d["off"], d["dfn"], d["plays_off"], d["plays_def"], d["position"])


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
    # Además de cada semana jugada, la clave (temporada, -1, equipo) guarda el uso esperado
    # después del último partido: es el que se usa para el próximo.
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
        out[(season, -1, team)] = {side: {p: num / den for p, (num, den) in acc[side].items() if den > 0}
                                   for side in ("o", "d")}
    return out


def snaps_usage(season: int) -> pl.DataFrame:
    """Uso por partido a partir de los snaps (para temporadas sin participación, como 2026): mismo
    formato que usage_from_plays. Los ids de PFR se pasan a los de nflverse con la tabla de jugadores."""
    import nflreadpy as nfl
    from app.services.nfl_qb_status import espn_to_gsis
    espn_to_gsis()
    pt = pl.read_parquet(os.path.join(CACHE_DIR, "players.parquet"))
    pfr = {a: g for a, g in zip(pt["pfr_id"].to_list(), pt["gsis_id"].to_list()) if a and g}
    sn = nfl.load_snap_counts([season]).with_columns(
        p=pl.col("pfr_player_id").replace_strict(pfr, default=None),
        team=pl.col("team").replace({"OAK": "LV", "SD": "LAC", "STL": "LA"}))
    sn = sn.filter(pl.col("p").is_not_null())
    o = sn.filter(pl.col("offense_pct") > 0).select("game_id", "season", "week", "team", "p",
                                                   pl.col("offense_pct").alias("share")).with_columns(side=pl.lit("o"))
    dd = sn.filter(pl.col("defense_pct") > 0).select("game_id", "season", "week", "team", "p",
                                                    pl.col("defense_pct").alias("share")).with_columns(side=pl.lit("d"))
    return pl.concat([o, dd]).select("game_id", "season", "week", "team", "side", "p", "share")


# ══════════════════════════════════════════════════════════════════════
# Fase 3: quién juega (lesión + práctica) y fuerza del equipo con las bajas previstas
# ══════════════════════════════════════════════════════════════════════

def load_injury_reports(seasons: Iterable[int]) -> pl.DataFrame:
    """Reporte final de lesiones por jugador y semana (nflverse), con su práctica."""
    import nflreadpy as nfl
    path = os.path.join(CACHE_DIR, "injuries_hist.parquet")
    seasons = list(seasons)
    if os.path.exists(path):
        cached = pl.read_parquet(path)
        have = set(cached["season"].unique().to_list())
        missing = [s for s in seasons if s not in have]
    else:
        cached, missing = None, seasons
    frames = [cached] if cached is not None else []
    if missing:
        d = nfl.load_injuries(missing).select(
            pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32), "team", "gsis_id", "position",
            "report_status", "practice_status")
        frames.append(d)
        out = pl.concat(frames, how="diagonal_relaxed")
        out.write_parquet(path)
    else:
        out = cached
    out = out.with_columns(pl.col("team").replace({"OAK": "LV", "SD": "LAC", "STL": "LA"}))
    return out.filter(pl.col("season").is_in(seasons) & pl.col("gsis_id").is_not_null())


def _practice_key(p: Optional[str]) -> str:
    p = (p or "").lower()
    return "full" if p.startswith("full") else "limited" if p.startswith("limited") else "dnp" if p.startswith("did not") else "none"


def _status_key(s: Optional[str]) -> str:
    s = (s or "").lower()
    return s if s in ("out", "doubtful", "questionable") else "none"


def learn_availability(inj: pl.DataFrame, use: pl.DataFrame, exp: Dict, min_share: float = 0.25,
                       smooth: float = 10.0) -> Dict[str, Dict[str, float]]:
    """P(juega | estado, práctica) aprendida: un jugador con uso esperado ≥ min_share cuenta como
    "jugó" si estuvo en al menos un tercio de su uso esperado. Encogido hacia la tasa del estado."""
    played = {}
    for r in use.iter_rows(named=True):
        played[(r["season"], r["week"], r["team"], r["p"])] = max(played.get((r["season"], r["week"], r["team"], r["p"]), 0.0), r["share"])
    rows = []
    for r in inj.iter_rows(named=True):
        e = exp.get((r["season"], r["week"], r["team"]))
        if not e:
            continue
        sh = max(e["o"].get(r["gsis_id"], 0.0), e["d"].get(r["gsis_id"], 0.0))
        if sh < min_share:
            continue
        got = played.get((r["season"], r["week"], r["team"], r["gsis_id"]), 0.0)
        rows.append((_status_key(r["report_status"]), _practice_key(r["practice_status"]), 1.0 if got >= sh / 3 else 0.0))
    df = pl.DataFrame(rows, schema=["status", "practice", "played"], orient="row")
    out = {}
    for st, g in df.group_by("status"):
        st = st[0] if isinstance(st, tuple) else st
        base = float(g["played"].mean())
        cell = {"any": round(base, 3), "n": g.height}
        for pr, gg in g.group_by("practice"):
            pr = pr[0] if isinstance(pr, tuple) else pr
            cell[pr] = round((float(gg["played"].sum()) + smooth * base) / (gg.height + smooth), 3)
        out[st] = cell
    return out


def availability_prob(table: Dict, status: Optional[str], practice: Optional[str]) -> float:
    """P(juega) relativa a un jugador sano: la tasa aprendida se divide entre la de un jugador sin
    estado que entrenó completo (~93%), porque hasta los sanos a veces juegan menos por rotación."""
    st = _status_key(status)
    if st == "none" and not practice:
        return 1.0
    ref = float((table.get("none") or {}).get("full", 1.0)) or 1.0
    cell = table.get(st) or {}
    return min(1.0, float(cell.get(_practice_key(practice), cell.get("any", 1.0))) / ref)


def replacement_levels(values: "PlayerValues", min_plays: float = 50.0) -> Dict[str, float]:
    """Valor de un suplente típico por grupo de posición: media del RAPM de los jugadores con pocas jugadas."""
    out = {}
    for side, vals, cnt in (("o", values.off, values.plays_off), ("d", values.dfn, values.plays_def)):
        by = defaultdict(list)
        for p, v in vals.items():
            if cnt.get(p, 0) < 400 and cnt.get(p, 0) >= min_plays:
                by[POS_GROUP.get(values.position.get(p, ""), "OTHER")].append(v)
        for g, l in by.items():
            out[(side, g)] = float(np.mean(l))
    return out


def injury_adjustment(values: "PlayerValues", usage: Dict[str, Dict[str, float]], report: Dict[str, Tuple[str, str]],
                      table: Dict, repl: Dict, skip_groups=()) -> Tuple[float, List[Dict]]:
    """Cuánto pierde el equipo (EPA por jugada) por sus bajas previstas: para cada jugador del reporte,
    (1 − P(juega)) × su uso esperado × (su valor − el de un suplente de su posición).
    Incluye al QB: el ajuste final se aprende encima de la v2 (que ya usa al titular), así que la
    regresión solo le da el peso que agrega. Sacarlo empeoraba 2019–2020 (ver PLAN_NFL_V3.md)."""
    lost, detail = 0.0, []
    for side, vals in (("o", values.off), ("d", values.dfn)):
        for p, sh in usage.get(side, {}).items():
            if p not in report or sh <= 0:
                continue
            prob = availability_prob(table, *report[p])
            if prob >= 1.0:
                continue
            g = POS_GROUP.get(values.position.get(p, ""), "OTHER")
            if g in skip_groups:
                continue
            delta = (1 - prob) * sh * (vals.get(p, 0.0) - repl.get((side, g), 0.0))
            lost += delta
            detail.append({"player": p, "side": side, "group": g, "share": round(sh, 2), "play_prob": round(prob, 2),
                           "value": round(vals.get(p, 0.0), 4), "lost": round(delta, 4)})
    return lost, detail
