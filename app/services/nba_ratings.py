"""
OmniPicks — NBA v2 (N1): rating ofensivo y defensivo de cada jugador, por fecha.

El rating de un jugador en la fecha D usa solo lo ocurrido ANTES de D (point-in-time),
así que sirve tal cual para el backtest. Unidades: puntos por 100 posesiones sobre el
promedio de la liga; en defensa, positivo = defiende mejor.

Modelo (una sola regresión sobre los stints, recalculada cada día de partidos):

    eficiencia del ataque en un tramo (pts/100 pos) =
        media de liga ± localía
        + Σ atacantes  ( z_p · γ_O + δO_p )
        − Σ defensores ( z_q · γ_D + δD_q )

  · z_p (2a + 2c) es el perfil de box score del jugador ANTES del partido: sus tasas por
    100 posesiones, minutos por partido y edad. Las tasas salen de un filtro bayesiano
    con descuento (gamma-Poisson): cada partido pesa según las posesiones jugadas, lo
    viejo se descuenta con el tiempo y con pocos datos manda el nivel de un suplente.
    Como z es anterior al tramo que explica, γ aprende a PREDECIR (la edad, por ejemplo,
    recoge la mejora esperada de los jóvenes).
  · δ (2b) es el plus-minus ajustado propio del jugador (RAPM): lo que sus equipos
    rinden con él en cancha y el box score no explica. Va con penalización ridge, así
    que con pocos minutos se queda cerca de cero y manda el box score.
  · Todo con decaimiento temporal: los tramos recientes pesan más.

Rating final:  O_p = z_p·γ_O + δO_p   ·   D_p = z_p·γ_D + δD_p
"""
import json
import logging
import os
from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.linalg import solve
from sqlalchemy import text

logger = logging.getLogger(__name__)

MODEL_VERSION = "n1"
BOX_STATS = ["pts", "fga", "fta", "fg3a", "fg3m", "ast", "tov", "oreb", "dreb", "stl", "blk", "pf"]
FEATURES = BOX_STATS + ["mpg", "age"]
K = len(FEATURES)

# Posiciones dentro del vector de parámetros
COL_MEAN, COL_HOME = 0, 1
COL_BOX_O = 2
COL_BOX_D = 2 + K
COL_PLAYERS = 2 + 2 * K


@dataclass
class RatingConfig:
    half_life_days: float = 400.0        # decaimiento de los stints (días efectivos)
    ridge_player: float = 4000.0         # penalización de δ, en posesiones
    ridge_box: float = 200.0             # penalización de γ
    box_half_life_days: float = 150.0    # descuento del filtro de box score
    box_prior_poss: float = 500.0        # peso del nivel de suplente, en posesiones
    mpg_prior_games: float = 5.0
    offseason_cap_days: float = 45.0     # el verano cuenta como máximo estos días
    use_box: bool = True                 # ablación: sin 2a/2c
    use_age: bool = False                # sin edad: no mejoró el backtest (0.6048 vs 0.6050)
    use_rapm: bool = True                # ablación: sin 2b
    sigma: float = 118.0                 # desviación de pts/100 en UNA posesión
    max_stint_gap: int = 6               # puntos de diferencia tolerados entre stints y marcador


# ══════════════════════════════════════════════════════════════════════
# Datos
# ══════════════════════════════════════════════════════════════════════

def load_data(engine, until: Optional[date] = None) -> Dict[str, pd.DataFrame]:
    """Lee de MySQL lo necesario. `until` (exclusivo) recorta para pruebas."""
    cut = "" if until is None else f" AND game_date < '{until.isoformat()}'"
    with engine.connect() as conn:
        games = pd.read_sql(text(
            "SELECT game_id, season, season_type, game_date, home_team_id, away_team_id, home_pts, away_pts "
            f"FROM nba_games WHERE home_pts IS NOT NULL{cut} ORDER BY game_date, game_id"), conn)
        box = pd.read_sql(text(
            "SELECT game_id, team_id, player_id, player_name, minutes, " + ", ".join(BOX_STATS) +
            f" FROM nba_player_games WHERE minutes > 0{cut}"), conn)
        stints = pd.read_sql(text(
            "SELECT game_id, home_players, away_players, home_pts, away_pts, home_poss, away_poss FROM nba_stints"), conn)
        bio = pd.read_sql(text("SELECT season, player_id, age FROM nba_player_seasons WHERE age IS NOT NULL"), conn)
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
    return {"games": games, "box": box, "stints": stints, "bio": bio}


class Prepared:
    """Datos ordenados por día y pasados a arreglos de numpy."""

    def __init__(self, data: Dict[str, pd.DataFrame], cfg: RatingConfig):
        games = data["games"].reset_index(drop=True)
        self.games = games
        self.days: List[date] = sorted(games["game_date"].unique())
        day_idx = {d: i for i, d in enumerate(self.days)}
        games["day"] = games["game_date"].map(day_idx)
        game_day = dict(zip(games["game_id"], games["day"]))
        game_home = dict(zip(games["game_id"], games["home_team_id"]))

        # Días efectivos: el verano se comprime para que no borre la temporada anterior
        eff = [0.0]
        for a, b in zip(self.days, self.days[1:]):
            eff.append(eff[-1] + min((b - a).days, cfg.offseason_cap_days))
        self.eff = np.array(eff)
        self.day_season = games.groupby("day")["season"].first().sort_index().to_numpy()

        # ── Box scores ────────────────────────────────────────────────
        box = data["box"][data["box"]["game_id"].isin(game_day)].copy()
        box["day"] = box["game_id"].map(game_day)
        team = box.groupby(["game_id", "team_id"]).agg(
            fga=("fga", "sum"), fta=("fta", "sum"), tov=("tov", "sum"), oreb=("oreb", "sum"), tmin=("minutes", "sum"))
        team["tposs"] = team.fga + 0.44 * team.fta + team.tov - team.oreb
        box = box.join(team[["tposs", "tmin"]], on=["game_id", "team_id"])
        box["poss"] = box.tposs * box.minutes * 5.0 / box.tmin.clip(lower=1.0)
        box = box.sort_values(["day", "game_id", "player_id"]).reset_index(drop=True)

        self.player_ids = np.sort(box["player_id"].unique())
        self.pidx = {int(p): i for i, p in enumerate(self.player_ids)}
        self.P = len(self.player_ids)
        names = box.drop_duplicates("player_id", keep="last").set_index("player_id")["player_name"]
        self.player_names = [names.get(p, "") for p in self.player_ids]

        self.box = box
        self.box_p = box["player_id"].map(self.pidx).to_numpy()
        self.box_poss = box["poss"].to_numpy(float)
        self.box_min = box["minutes"].to_numpy(float)
        self.box_stats = box[BOX_STATS].to_numpy(float)
        self.box_day = box["day"].to_numpy()
        self.box_slices = np.searchsorted(self.box_day, np.arange(len(self.days) + 1))

        # ── Stints ────────────────────────────────────────────────────
        st = data["stints"][data["stints"]["game_id"].isin(game_day)].copy()
        tot = st.groupby("game_id")[["home_pts", "away_pts"]].sum().join(
            games.set_index("game_id")[["home_pts", "away_pts"]], rsuffix="_real")
        gap = (tot.home_pts - tot.home_pts_real).abs() + (tot.away_pts - tot.away_pts_real).abs()
        bad = set(gap[gap > cfg.max_stint_gap].index)
        self.games_with_stints = len(tot) - len(bad)
        self.games_bad_stints = len(bad)
        st = st[~st["game_id"].isin(bad)]
        st["day"] = st["game_id"].map(game_day)
        st = st.sort_values("day").reset_index(drop=True)
        self.st_games_per_day = (st.groupby("day")["game_id"].nunique()
                                 .reindex(range(len(self.days)), fill_value=0).to_numpy(float))

        def to_idx(col):
            arr = np.array([[self.pidx.get(int(x), -1) for x in s.split(",")] for s in st[col]], dtype=int)
            return arr.reshape(len(st), 5) if len(st) else arr.reshape(0, 5)

        home, away = to_idx("home_players"), to_idx("away_players")
        ok = (home >= 0).all(1) & (away >= 0).all(1)
        self.st_home, self.st_away = home[ok], away[ok]
        self.st_hp = st["home_pts"].to_numpy(float)[ok]
        self.st_ap = st["away_pts"].to_numpy(float)[ok]
        self.st_hposs = st["home_poss"].to_numpy(float)[ok]
        self.st_aposs = st["away_poss"].to_numpy(float)[ok]
        self.st_day = st["day"].to_numpy()[ok]
        self.st_slices = np.searchsorted(self.st_day, np.arange(len(self.days) + 1))
        self.last_stint_day = int(self.st_day.max()) if len(self.st_day) else -1
        self.game_home = game_home

        # ── Edad por temporada (si falta, se deduce de otra temporada; si no hay, 26) ──
        seasons = sorted(set(self.day_season))
        bio = data["bio"]
        known: Dict[int, Dict[int, float]] = {}
        for r in bio.itertuples():
            if int(r.player_id) in self.pidx:
                known.setdefault(self.pidx[int(r.player_id)], {})[int(r.season[:4])] = float(r.age)
        self.age = {}
        for s in seasons:
            y = int(s[:4])
            arr = np.full(self.P, 26.0)
            for p, d in known.items():
                ref = min(d, key=lambda yy: abs(yy - y))
                arr[p] = d[ref] + (y - ref)
            self.age[s] = arr

        # ── Constantes fijas, tomadas de la primera temporada (la de calentamiento) ──
        first = box["game_id"].map(dict(zip(games["game_id"], games["season"]))) == seasons[0]
        b0 = box[first]
        low = b0[b0.minutes < 15]
        self.prior_rate = low[BOX_STATS].sum().to_numpy(float) / max(low["poss"].sum(), 1.0)      # por posesión
        self.prior_mpg = float(low["minutes"].mean()) if len(low) else 10.0
        agg = b0.groupby("player_id").agg({**{s: "sum" for s in BOX_STATS}, "poss": "sum", "minutes": "mean"})
        w = agg["poss"].to_numpy(float)
        rates = 100.0 * agg[BOX_STATS].to_numpy(float) / np.maximum(w[:, None], 1.0)
        feats = np.column_stack([rates, agg["minutes"].to_numpy(float), np.full(len(agg), 26.5)])
        mean = (feats * w[:, None]).sum(0) / w.sum()
        sd = np.sqrt((((feats - mean) ** 2) * w[:, None]).sum(0) / w.sum())
        mean[-1], sd[-1] = 26.5, 4.0
        self.feat_mean, self.feat_sd = mean, np.where(sd > 0, sd, 1.0)


# ══════════════════════════════════════════════════════════════════════
# Motor
# ══════════════════════════════════════════════════════════════════════

@dataclass
class DayRatings:
    day: date
    o: np.ndarray                 # (P,) rating ofensivo
    d: np.ndarray                 # (P,) rating defensivo
    unc: np.ndarray               # (P,) incertidumbre aproximada de O+D
    mpg: np.ndarray               # (P,) minutos por partido esperados (filtro)
    minutes: np.ndarray           # (P,) minutos recientes con descuento (tamaño de muestra)
    seen: np.ndarray              # (P,) el jugador ya debutó
    home_edge: float              # ventaja de local en pts/100 (margen)
    league_eff: float             # eficiencia media de la liga, pts/100


class RatingEngine:
    def __init__(self, prep: Prepared, cfg: RatingConfig):
        self.prep, self.cfg = prep, cfg
        P = prep.P
        self.n = COL_PLAYERS + 2 * P
        self.A = np.zeros((self.n, self.n))
        self.b = np.zeros(self.n)
        self.alpha = np.zeros((P, len(BOX_STATS)))      # cuentas con descuento
        self.beta = np.zeros(P)                         # posesiones con descuento
        self.min_sum = np.zeros(P)
        self.min_games = np.zeros(P)
        self.seen = np.zeros(P, dtype=bool)
        self.theta: Optional[np.ndarray] = None
        self.resid = []                                 # (posesiones, error²) fuera de muestra, por tramo

    # ── 2a + 2c: perfil de box score antes del partido ────────────────
    def features(self, season: str) -> np.ndarray:
        prep, cfg = self.prep, self.cfg
        m = cfg.box_prior_poss
        rates = 100.0 * (self.alpha + m * prep.prior_rate) / (self.beta + m)[:, None]
        mpg = (self.min_sum + cfg.mpg_prior_games * prep.prior_mpg) / (self.min_games + cfg.mpg_prior_games)
        age = np.clip(prep.age[season], 19.0, 38.0) if cfg.use_age else np.full(prep.P, 26.5)
        z = (np.column_stack([rates, mpg, age]) - prep.feat_mean) / prep.feat_sd
        return z

    def mpg(self) -> np.ndarray:
        c = self.cfg
        return (self.min_sum + c.mpg_prior_games * self.prep.prior_mpg) / (self.min_games + c.mpg_prior_games)

    # ── 2b: regresión ridge sobre lo acumulado ────────────────────────
    def solve(self) -> np.ndarray:
        cfg, P = self.cfg, self.prep.P
        seen = np.flatnonzero(self.seen)
        cols = [np.array([COL_MEAN, COL_HOME])]
        pen = [np.array([1e-6, 1.0])]
        if cfg.use_box:
            cols.append(np.arange(COL_BOX_O, COL_PLAYERS))
            pen.append(np.full(2 * K, cfg.ridge_box))
        if cfg.use_rapm:
            cols += [COL_PLAYERS + seen, COL_PLAYERS + P + seen]
            pen.append(np.full(2 * len(seen), cfg.ridge_player))
        cols, pen = np.concatenate(cols), np.concatenate(pen)
        M = self.A[np.ix_(cols, cols)]
        M[np.diag_indices_from(M)] += pen
        theta = np.zeros(self.n)
        theta[cols] = solve(M, self.b[cols], assume_a="pos")
        return theta

    def ratings(self, z: np.ndarray, day: date) -> DayRatings:
        cfg, P = self.cfg, self.prep.P
        th = self.theta
        o = z @ th[COL_BOX_O:COL_BOX_D] + th[COL_PLAYERS:COL_PLAYERS + P]
        d = z @ th[COL_BOX_D:COL_PLAYERS] + th[COL_PLAYERS + P:]
        diag = np.diag(self.A)
        var = cfg.sigma ** 2 * (1.0 / (diag[COL_PLAYERS:COL_PLAYERS + P] + cfg.ridge_player)
                                + 1.0 / (diag[COL_PLAYERS + P:] + cfg.ridge_player))
        return DayRatings(day=day, o=o, d=d, unc=np.sqrt(var), mpg=self.mpg(), minutes=self.min_sum.copy(),
                          seen=self.seen.copy(), home_edge=2.0 * th[COL_HOME], league_eff=th[COL_MEAN])

    # ── Filas de la regresión para los tramos de un día ───────────────
    def _rows(self, i: int, z: np.ndarray):
        prep, P = self.prep, self.prep.P
        a, b = prep.st_slices[i], prep.st_slices[i + 1]
        if a == b:
            return None
        home, away = prep.st_home[a:b], prep.st_away[a:b]
        blocks = []
        for off, dfn, pts, poss, sign in ((home, away, prep.st_hp[a:b], prep.st_hposs[a:b], 1.0),
                                          (away, home, prep.st_ap[a:b], prep.st_aposs[a:b], -1.0)):
            keep = poss > 0
            off, dfn, pts, poss = off[keep], dfn[keep], pts[keep], poss[keep]
            n = len(poss)
            X = np.zeros((n, self.n))
            X[:, COL_MEAN] = 1.0
            X[:, COL_HOME] = sign
            X[:, COL_BOX_O:COL_BOX_D] = z[off].sum(1)
            X[:, COL_BOX_D:COL_PLAYERS] = -z[dfn].sum(1)
            r = np.arange(n)[:, None]
            X[r, COL_PLAYERS + off] = 1.0
            X[r, COL_PLAYERS + P + dfn] = -1.0
            blocks.append((X, 100.0 * pts / poss, poss))
        X = np.vstack([blk[0] for blk in blocks])
        y = np.concatenate([blk[1] for blk in blocks])
        w = np.concatenate([blk[2] for blk in blocks])
        return X, y, w

    # ── Recorrido cronológico ─────────────────────────────────────────
    def run(self, first_output: Optional[date] = None,
            on_day: Optional[Callable[[DayRatings], None]] = None, extra_day: Optional[date] = None) -> List[DayRatings]:
        """Recorre todos los días. Devuelve los ratings previos a cada día desde `first_output`.

        `extra_day` agrega al final los ratings para una fecha futura (p. ej. hoy), ya con
        todos los partidos incluidos.
        """
        prep, cfg = self.prep, self.cfg
        out: List[DayRatings] = []
        prev_eff = 0.0
        for i, day in enumerate(prep.days):
            self._decay(prep.eff[i] - prev_eff)
            prev_eff = prep.eff[i]
            z = self.features(prep.day_season[i])
            want = first_output is not None and day >= first_output and i > 0
            rows = self._rows(i, z)
            if want:
                self.theta = self.solve()
                r = self.ratings(z, day)
                out.append(r)
                if on_day:
                    on_day(r)
                if rows is not None:                       # error de predicción fuera de muestra
                    X, y, w = rows
                    self.resid.append((w.sum(), float((w * (y - X @ self.theta) ** 2).sum()), len(w)))
            if rows is not None:
                X, y, w = rows
                self.A += X.T @ (X * w[:, None])
                self.b += X.T @ (w * y)
            self._update_box(i)
        if extra_day is not None:
            gap = min((extra_day - prep.days[-1]).days, cfg.offseason_cap_days)
            self._decay(max(gap, 0))
            season = prep.day_season[-1]
            self.theta = self.solve()
            r = self.ratings(self.features(season), extra_day)
            out.append(r)
            if on_day:
                on_day(r)
        return out

    def _decay(self, days: float):
        if days <= 0:
            return
        cfg = self.cfg
        g = 0.5 ** (days / cfg.half_life_days)
        self.A *= g
        self.b *= g
        gb = 0.5 ** (days / cfg.box_half_life_days)
        self.alpha *= gb
        self.beta *= gb
        self.min_sum *= gb
        self.min_games *= gb

    def _update_box(self, i: int):
        prep = self.prep
        a, b = prep.box_slices[i], prep.box_slices[i + 1]
        p = prep.box_p[a:b]
        np.add.at(self.alpha, p, prep.box_stats[a:b])
        np.add.at(self.beta, p, prep.box_poss[a:b])
        np.add.at(self.min_sum, p, prep.box_min[a:b])
        np.add.at(self.min_games, p, 1.0)
        self.seen[p] = True

    def out_of_sample_sigma(self) -> Optional[float]:
        """Desviación por posesión (pts/100) del error al predecir tramos que aún no se veían."""
        if not self.resid:
            return None
        sse = sum(r[1] for r in self.resid)
        n = sum(r[2] for r in self.resid)
        return float(np.sqrt(sse / max(n, 1)))

    def box_weights(self) -> pd.DataFrame:
        th = self.theta
        return pd.DataFrame({"feature": FEATURES, "ataque": th[COL_BOX_O:COL_BOX_D], "defensa": th[COL_BOX_D:COL_PLAYERS]})


# ══════════════════════════════════════════════════════════════════════
# Guardado
# ══════════════════════════════════════════════════════════════════════

def rating_rows(prep: Prepared, r: DayRatings, active_minutes: float = 1.0) -> List[Dict]:
    """Filas para nba_player_ratings: jugadores con actividad reciente."""
    keep = np.flatnonzero(r.seen & (r.minutes >= active_minutes))
    return [{
        "as_of_date": r.day, "player_id": int(prep.player_ids[p]), "player_name": prep.player_names[p],
        "o_rating": round(float(r.o[p]), 3), "d_rating": round(float(r.d[p]), 3),
        "uncertainty": round(float(r.unc[p]), 3), "minutes_sample": round(float(r.minutes[p]), 1),
        "model_version": MODEL_VERSION,
    } for p in keep]


def build_and_store(engine, first_output: Optional[date] = None, today: Optional[date] = None,
                    cfg: Optional[RatingConfig] = None) -> Dict:
    """Recalcula los ratings desde cero y guarda los de `first_output` en adelante.

    Con `today` agrega los ratings vigentes hoy (con todos los partidos ya jugados);
    con `first_output=None` guarda solo esos. Reemplaza lo que hubiera de esas fechas.
    Recalcular todo tarda cerca de un minuto, así que no se guarda estado intermedio.
    """
    from app.db_models import Base, NBAPlayerRating

    cfg = cfg or RatingConfig()
    Base.metadata.create_all(engine)
    prep = Prepared(load_data(engine), cfg)
    if prep.games_bad_stints:
        logger.warning(f"Ratings NBA: {prep.games_bad_stints} partidos sin stints confiables (se omiten en el RAPM)")
    eng = RatingEngine(prep, cfg)
    history = eng.run(first_output=first_output, extra_day=today)
    if not history:
        return {"dates": 0, "rows": 0}

    rows = [row for r in history for row in rating_rows(prep, r, active_minutes=20.0)]
    since = min(r.day for r in history)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM nba_player_ratings WHERE model_version = :v AND as_of_date >= :d"),
                     {"v": MODEL_VERSION, "d": since})
        for i in range(0, len(rows), 5000):
            conn.execute(NBAPlayerRating.__table__.insert(), rows[i:i + 5000])
    _save_meta(history)
    result = {"dates": len(history), "rows": len(rows), "from": str(since), "to": str(history[-1].day),
              "players": prep.P, "games_with_stints": prep.games_with_stints, "games_skipped": prep.games_bad_stints,
              "home_edge": round(history[-1].home_edge, 2), "league_eff": round(history[-1].league_eff, 1)}
    logger.info(f"Ratings NBA guardados: {result}")
    return result


META_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ml", "saved_models", "nba_ratings_meta.json"))


def _save_meta(history: List[DayRatings]):
    """Localía y eficiencia de liga de cada fecha (las usa el modelo del partido)."""
    meta = load_meta()
    for r in history:
        meta[r.day.isoformat()] = {"home_edge": round(float(r.home_edge), 3), "league_eff": round(float(r.league_eff), 3)}
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(meta, f)


def load_meta() -> Dict[str, Dict[str, float]]:
    if not os.path.exists(META_PATH):
        return {}
    with open(META_PATH) as f:
        return json.load(f)


def meta_for(day: date) -> Dict[str, float]:
    """Valores vigentes en `day` (los de la fecha guardada más reciente que no la pase)."""
    meta = load_meta()
    keys = [k for k in meta if k <= day.isoformat()]
    return meta[max(keys)] if keys else {"home_edge": 2.2, "league_eff": 120.0}
