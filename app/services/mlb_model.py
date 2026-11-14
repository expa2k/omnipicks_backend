"""
OmniPicks — MLB v2: modelo de carreras por jugador.

Por cada equipo y partido (dos renglones por juego):

    log E[carreras] = μ + localía + noche + parque
                      + (1/9) Σ bateo[cada bateador de la alineación del día]
                      − abridor[abridor rival] − bullpen[bullpen/defensa del rival en esa temporada]
                      + plataforma · (fracción de la alineación con ventaja de mano contra el abridor)
                      + cansancio · (lanzadores que usó el bullpen rival en sus 2 juegos previos)

Ajuste por cuasi-verosimilitud Poisson con descuento por antigüedad y ridge por grupo (los jugadores
con pocos juegos quedan cerca del promedio). Las carreras de cada equipo siguen una binomial negativa
(más dispersión que Poisson); de ahí salen moneyline (con entradas extra), run line y total.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy import sparse
from scipy.optimize import minimize
from scipy.stats import nbinom

MAX_RUNS = 30
HOME_TIE_SHARE = 0.52        # el local gana ~52% de los juegos que se van a entradas extra


@dataclass
class MLBParams:
    xi: float = 0.002                 # por día (vida media ≈ 1 año)
    window_days: int = 1100
    # Elegidos con 2015–2019 (ml/scripts/mlb_eval.py): el abridor necesita mucho encogimiento (las
    # carreras de un juego tienen mucho azar); los bateadores casi no cambian con el valor.
    ridge: Dict[str, float] = field(default_factory=lambda: {"hit": 40.0, "sp": 150.0, "pen": 6.0, "park": 10.0, "team": 6.0})
    use_players: bool = True          # bateadores, abridor y bullpen
    use_team: bool = True             # nivel de ataque y defensa del equipo en la temporada (los jugadores se suman encima)
    use_context: bool = True          # plataforma y cansancio del bullpen
    covariates: Tuple[str, ...] = ()  # columnas numéricas extra (p. ej. habilidad K/BB/HR del abridor rival)
    target: str = "runs"


def team_rows(gl: pl.DataFrame, hand: Dict[str, Dict[str, str]]) -> pl.DataFrame:
    """Dos renglones por juego: el equipo que batea, el que lanza y lo que se sabe antes del juego."""
    g = gl.filter(pl.col("home_score").is_not_null()).sort(["date", "game_num"])
    # Cansancio: lanzadores usados por cada equipo en sus 2 juegos anteriores (menos los 2 abridores)
    fatigue = {}
    hist = defaultdict(list)
    for r in g.select("date", "home", "away", "home_pitchers", "away_pitchers").iter_rows(named=True):
        for t, k in ((r["home"], "home_pitchers"), (r["away"], "away_pitchers")):
            prev = [x for d, x in hist[t] if (r["date"] - d).days <= 2]
            fatigue[(r["date"], t)] = float(sum(max(0.0, x - 1) for x in prev))
            hist[t] = (hist[t] + [(r["date"], r[k] or 4.0)])[-3:]

    def adv(lineup: str, sp: str) -> float:
        th = (hand.get(sp) or {}).get("throws")
        if th not in ("L", "R"):
            return 0.0
        n = 0
        for b in lineup.split(";"):
            bt = (hand.get(b) or {}).get("bats")
            n += 1 if bt == "S" or (bt and bt != th) else 0
        return n / 9.0

    rows = []
    for r in g.iter_rows(named=True):
        base = {"date": r["date"], "season": r["season"], "park": r["park"], "night": 1.0 if r["day_night"] == "N" else 0.0,
                "game_id": f"{r['date']}_{r['home']}_{r['game_num']}"}
        for side, opp in (("home", "away"), ("away", "home")):
            rows.append({**base, "bat": r[side], "pit": r[opp], "is_home": 1.0 if side == "home" else 0.0,
                         "lineup": r[f"{side}_lineup"], "sp": r[f"{opp}_sp_id"], "runs": r[f"{side}_score"],
                         "k": r[f"{side}_k"], "bb": r[f"{side}_bb"], "hr": r[f"{side}_hr"],
                         "platoon": adv(r[f"{side}_lineup"], r[f"{opp}_sp_id"]),
                         "fatigue": fatigue.get((r["date"], r[opp]), 0.0)})
    return pl.DataFrame(rows)


@dataclass
class MLBFit:
    index: Dict[Tuple[str, str], int]
    beta: np.ndarray
    dispersion: float
    params: MLBParams

    def log_rate(self, row: Dict) -> float:
        b, ix = self.beta, self.index
        v = b[ix[("mu", "")]] + b[ix[("home", "")]] * row["is_home"] + b[ix[("night", "")]] * row["night"]
        v += b[ix[("park", row["park"])]] if ("park", row["park"]) in ix else 0.0
        if self.params.use_players:
            v += sum(b[ix[("hit", h)]] for h in row["lineup"].split(";") if ("hit", h) in ix) / 9.0
            v -= b[ix[("sp", row["sp"])]] if ("sp", row["sp"]) in ix else 0.0
            pen = ("pen", f"{row['pit']}_{row['season']}")
            v -= b[ix[pen]] if (pen in ix and not self.params.use_team) else 0.0
        if self.params.use_team or not self.params.use_players:
            for k, t in (("off", row["bat"]), ("def", row["pit"])):
                key = ("team", f"{k}_{t}_{row['season']}")
                if key in ix:
                    v += b[ix[key]] if k == "off" else -b[ix[key]]
        if self.params.use_context:
            v += b[ix[("platoon", "")]] * row["platoon"] + b[ix[("fatigue", "")]] * row["fatigue"]
        for c in self.params.covariates:
            v += b[ix[("cov", c)]] * float(row.get(c) or 0.0)
        return float(v)

    def rate(self, row: Dict) -> float:
        return float(np.exp(self.log_rate(row)))


def _design(rows: pl.DataFrame, params: MLBParams, index: Optional[Dict] = None):
    """Matriz rala (renglones × parámetros) y el índice de parámetros."""
    build = index is None
    index = {} if build else index
    def col(key):
        if key not in index:
            if not build:
                return None
            index[key] = len(index)
        return index[key]
    for k in ("mu", "home", "night", "platoon", "fatigue"):
        col((k, ""))
    for c in params.covariates:
        col(("cov", c))
    I, J, V = [], [], []
    for i, r in enumerate(rows.iter_rows(named=True)):
        for key, val in ((("mu", ""), 1.0), (("home", ""), r["is_home"]), (("night", ""), r["night"]), (("park", r["park"]), 1.0)):
            c = col(key)
            if c is not None and val:
                I.append(i); J.append(c); V.append(val)
        if params.use_players:
            for h in r["lineup"].split(";"):
                c = col(("hit", h))
                if c is not None:
                    I.append(i); J.append(c); V.append(1.0 / 9.0)
            for key in ((("sp", r["sp"]),) + ((("pen", f"{r['pit']}_{r['season']}"),) if not params.use_team else ())):
                c = col(key)
                if c is not None:
                    I.append(i); J.append(c); V.append(-1.0)
        if params.use_team or not params.use_players:
            for key, val in ((("team", f"off_{r['bat']}_{r['season']}"), 1.0), (("team", f"def_{r['pit']}_{r['season']}"), -1.0)):
                c = col(key)
                if c is not None:
                    I.append(i); J.append(c); V.append(val)
        if params.use_context:
            for key, val in ((("platoon", ""), r["platoon"]), (("fatigue", ""), r["fatigue"])):
                if val:
                    I.append(i); J.append(index[key]); V.append(val)
        for c in params.covariates:
            val = float(r.get(c) or 0.0)
            if val:
                I.append(i); J.append(index[("cov", c)]); V.append(val)
    X = sparse.csr_matrix((V, (I, J)), shape=(rows.height, len(index)))
    return X, index


def fit(rows: pl.DataFrame, ref, params: MLBParams = MLBParams(), init: Optional[MLBFit] = None) -> MLBFit:
    ref = np.datetime64(ref, "D")
    d = rows["date"].to_numpy().astype("datetime64[D]")
    days = (ref - d).astype(int)
    tr = rows.filter(pl.Series((days > 0) & (days <= params.window_days)))
    days = days[(days > 0) & (days <= params.window_days)]
    X, index = _design(tr, params)
    y = tr[params.target].to_numpy().astype(float)
    w = np.exp(-params.xi * days)
    lam = np.zeros(len(index))
    for (g, _), k in index.items():
        lam[k] = params.ridge.get(g, 0.0)
    b0 = np.zeros(len(index))
    b0[index[("mu", "")]] = np.log(max(np.average(y, weights=w), 0.5))
    if init is not None:
        for key, k in index.items():
            if key in init.index:
                b0[k] = init.beta[init.index[key]]

    def f(b):
        eta = X @ b
        mu = np.exp(eta)
        val = float(np.sum(w * (mu - y * eta)) + np.sum(lam * b * b))
        g = X.T @ (w * (mu - y)) + 2 * lam * b
        return val, g

    res = minimize(f, b0, jac=True, method="L-BFGS-B", options={"maxiter": 3000})
    mu = np.exp(X @ res.x)
    # Dispersión de la binomial negativa: Var = μ + μ²/r  →  r por momentos
    excess = np.sum(w * ((y - mu) ** 2 - mu)) / max(np.sum(w * mu ** 2), 1e-9)
    r = 1.0 / max(excess, 1e-3)
    return MLBFit(index=index, beta=res.x, dispersion=float(r), params=params)


def runs_pmf(mean: float, r: float) -> np.ndarray:
    k = np.arange(MAX_RUNS + 1)
    p = nbinom.pmf(k, r, r / (r + mean))
    return p / p.sum()


def game_probs(lh: float, la: float, r: float, run_line: Optional[float] = -1.5, total: Optional[float] = None) -> Dict:
    """Probabilidades del juego: local gana (con entradas extra), local cubre el run line, más del total."""
    ph, pa = runs_pmf(lh, r), runs_pmf(la, r)
    m = np.outer(ph, pa)
    k = np.arange(MAX_RUNS + 1)
    diff = k[:, None] - k[None, :]
    tot = k[:, None] + k[None, :]
    win = float(m[diff > 0].sum() + HOME_TIE_SHARE * m[diff == 0].sum())
    out = {"home_win": win, "exp_home": lh, "exp_away": la}
    if run_line is not None:
        # El empate en 9 entradas se decide en extras por 1 carrera: cuenta como margen ±1
        d1 = m[diff == 0].sum()
        p_cover = float(m[diff + run_line > 0].sum() + (HOME_TIE_SHARE * d1 if run_line > -1 else 0.0))
        out["home_cover"] = p_cover
    if total is not None:
        over, push = float(m[tot > total].sum()), float(m[tot == total].sum())
        out["over"] = over / max(1 - push, 1e-9)
    return out


SKILLS = ("k", "bb", "hr")


def skill_params(base: "MLBParams") -> "MLBParams":
    """Modelos de ponches, bases por bolas y jonrones por juego: abridor + bateadores + equipo, sin contexto."""
    from dataclasses import replace
    return replace(base, use_context=False, covariates=(), ridge={**base.ridge, "sp": 20.0, "hit": 20.0})


def add_skills(rows: pl.DataFrame, fits: Dict[str, "MLBFit"]) -> pl.DataFrame:
    """Columnas sp_k, sp_bb, sp_hr: lo que el abridor rival le cambia (en escala log) a los ponches,
    bases por bolas y jonrones del equipo que batea. Más ponches = mejor abridor."""
    out = {}
    for t, f in fits.items():
        ix, b = f.index, f.beta
        out[f"sp_{t}"] = [float(b[ix[("sp", s)]]) if ("sp", s) in ix else 0.0 for s in rows["sp"].to_list()]
    return rows.with_columns([pl.Series(k, v) for k, v in out.items()])
