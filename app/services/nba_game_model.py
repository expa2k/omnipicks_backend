"""
OmniPicks — NBA v2 (N3 + N4): del rating de los jugadores al marcador del partido.

  · Contexto (N4): descanso, back-to-back, 3 partidos en 4 noches, viaje, cambio de
    zona horaria, altitud y momento de la temporada, calculados del calendario propio.
  · Ritmo: posesiones de cada equipo con descuento y regresión a la media de liga.
  · Estructural: puntos de cada lado = posesiones × (eficiencia de liga + ataque propio
    − defensa rival ± localía). El ataque y la defensa del equipo son Σ minutos × rating.
  · Corrección (N4): una regresión ridge sobre los ERRORES del estructural con las
    variables de contexto (sin nada del mercado). Se entrena en el backtest.
  · Simulación (N3): se sortea quién juega entre los que tienen duda; con pocos casos
    dudosos se enumeran todos los escenarios con su probabilidad exacta, y de esa mezcla
    se sacan 10,000 marcadores.
  · Calibración: regresión isotónica sobre la probabilidad de victoria.

Los parámetros entrenados viven en ml/saved_models/nba_v2.json (los escribe
ml/scripts/nba_backtest.py).
"""
import itertools
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np

MODEL_NAME = "nba_players_v2"
PARAMS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ml", "saved_models", "nba_v2.json"))

# ── Arenas: latitud, longitud, huso horario (UTC, sin horario de verano) y altitud en metros ──
ARENAS = {
    1610612737: (33.757, -84.396, -5, 320),    # ATL
    1610612738: (42.366, -71.062, -5, 5),      # BOS
    1610612751: (40.683, -73.976, -5, 10),     # BKN
    1610612766: (35.225, -80.839, -5, 230),    # CHA
    1610612741: (41.881, -87.674, -6, 180),    # CHI
    1610612739: (41.496, -81.688, -5, 200),    # CLE
    1610612742: (32.790, -96.810, -6, 140),    # DAL
    1610612743: (39.749, -105.008, -7, 1609),  # DEN
    1610612765: (42.341, -83.055, -5, 180),    # DET
    1610612744: (37.768, -122.388, -8, 5),     # GSW
    1610612745: (29.751, -95.362, -6, 15),     # HOU
    1610612754: (39.764, -86.155, -5, 220),    # IND
    1610612746: (33.945, -118.343, -8, 30),    # LAC
    1610612747: (34.043, -118.267, -8, 90),    # LAL
    1610612763: (35.138, -90.051, -6, 80),     # MEM
    1610612748: (25.781, -80.188, -5, 2),      # MIA
    1610612749: (43.045, -87.917, -6, 190),    # MIL
    1610612750: (44.979, -93.276, -6, 260),    # MIN
    1610612740: (29.949, -90.082, -6, 2),      # NOP
    1610612752: (40.751, -73.993, -5, 10),     # NYK
    1610612760: (35.463, -97.515, -6, 370),    # OKC
    1610612753: (28.539, -81.384, -5, 30),     # ORL
    1610612755: (39.901, -75.172, -5, 10),     # PHI
    1610612756: (33.446, -112.071, -7, 330),   # PHX
    1610612757: (45.532, -122.667, -8, 15),    # POR
    1610612758: (38.580, -121.500, -8, 10),    # SAC
    1610612759: (29.427, -98.438, -6, 200),    # SAS
    1610612761: (43.643, -79.379, -5, 80),     # TOR
    1610612762: (40.768, -111.901, -7, 1288),  # UTA
    1610612764: (38.898, -77.021, -5, 10),     # WAS
}
ALTITUDE_M = 1000.0

MARGIN_FEATURES = [
    "b2b_h", "b2b_a", "rest_diff", "three_in_four_h", "three_in_four_a",
    "travel_h", "travel_a", "east_a", "west_a", "altitude_visit", "playoffs", "early_season",
]
TOTAL_FEATURES = ["b2b_sum", "three_in_four_sum", "playoffs", "early_season", "altitude_visit", "late_season"]


def _km(a: int, b: int) -> float:
    if a == b or a not in ARENAS or b not in ARENAS:
        return 0.0
    la1, lo1, _, _ = ARENAS[a]
    la2, lo2, _, _ = ARENAS[b]
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


# ══════════════════════════════════════════════════════════════════════
# Calendario → contexto
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ScheduleState:
    """Partidos jugados por cada equipo (fecha y equipo local, que define la sede)."""
    games: Dict[int, List[Tuple[date, int]]] = field(default_factory=lambda: defaultdict(list))
    season_start: Dict[str, date] = field(default_factory=dict)
    season_end: Dict[str, date] = field(default_factory=dict)

    def add(self, day: date, home: int, away: int):
        self.games[home].append((day, home))
        self.games[away].append((day, home))

    def team(self, team: int, day: date, venue: int) -> Dict[str, float]:
        hist = self.games.get(team, [])
        last = hist[-1] if hist else None
        gap = (day - last[0]).days if last else 30
        recent = [g for g in hist[-4:] if 0 < (day - g[0]).days <= 3]
        prev_venue = last[1] if last and gap <= 4 else team          # tras un parón, sale de casa
        tz_from = ARENAS.get(prev_venue, (0, 0, 0, 0))[2]
        tz_to = ARENAS.get(venue, (0, 0, 0, 0))[2]
        return {
            "rest": float(min(max(gap - 1, 0), 3)),
            "b2b": float(gap == 1),
            "three_in_four": float(len(recent) >= 2),
            "travel": _km(prev_venue, venue) / 1000.0,
            "tz": float(tz_to - tz_from),                            # + = viajó al este
        }

    def game(self, day: date, season: str, season_type: str, home: int, away: int) -> Dict[str, float]:
        h, a = self.team(home, day, home), self.team(away, day, home)
        start = self.season_start.get(season)
        end = self.season_end.get(season)
        alt = float(ARENAS.get(home, (0, 0, 0, 0))[3] >= ALTITUDE_M and ARENAS.get(away, (0, 0, 0, 0))[3] < ALTITUDE_M)
        playoffs = float(season_type == "Playoffs")
        early = float(start is not None and (day - start).days < 21 and not playoffs)
        late = float(end is not None and 0 <= (end - day).days < 10 and not playoffs)
        return {
            "b2b_h": h["b2b"], "b2b_a": a["b2b"], "rest_diff": h["rest"] - a["rest"],
            "three_in_four_h": h["three_in_four"], "three_in_four_a": a["three_in_four"],
            "travel_h": h["travel"], "travel_a": a["travel"],
            "east_a": max(a["tz"], 0.0), "west_a": max(-a["tz"], 0.0),
            "altitude_visit": alt, "playoffs": playoffs, "early_season": early, "late_season": late,
            "b2b_sum": h["b2b"] + a["b2b"], "three_in_four_sum": h["three_in_four"] + a["three_in_four"],
        }


# ══════════════════════════════════════════════════════════════════════
# Ritmo y eficiencia de liga (con descuento, point-in-time)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PaceState:
    decay: float = 0.93                 # por partido del equipo
    prior_games: float = 6.0
    carry: float = 2.0 / 3.0            # cuánto del ritmo se conserva entre temporadas
    league_decay: float = 0.997         # por partido de liga
    team_sum: Dict[int, float] = field(default_factory=dict)
    team_n: Dict[int, float] = field(default_factory=dict)
    lg_poss: float = 0.0
    lg_pts: float = 0.0
    lg_n: float = 0.0
    season: Optional[str] = None

    def league_pace(self) -> float:
        return self.lg_poss / self.lg_n if self.lg_n > 5 else 99.0

    def league_ppp(self) -> float:
        return 100.0 * self.lg_pts / self.lg_poss if self.lg_poss > 0 else 114.0

    def pace(self, team: int) -> float:
        lp = self.league_pace()
        s, n = self.team_sum.get(team, 0.0), self.team_n.get(team, 0.0)
        return (s + self.prior_games * lp) / (n + self.prior_games)

    def new_season(self, season: str):
        if self.season is not None and season != self.season:
            lp = self.league_pace()
            for t in list(self.team_sum):
                mean = self.team_sum[t] / max(self.team_n[t], 1e-9)
                self.team_n[t] = min(self.team_n[t], 10.0)
                self.team_sum[t] = (lp + self.carry * (mean - lp)) * self.team_n[t]
        self.season = season

    def add(self, home: int, away: int, poss: float, pts: float):
        for t in (home, away):
            self.team_sum[t] = self.decay * self.team_sum.get(t, 0.0) + poss
            self.team_n[t] = self.decay * self.team_n.get(t, 0.0) + 1.0
        self.lg_poss = self.league_decay * self.lg_poss + poss
        self.lg_pts = self.league_decay * self.lg_pts + pts / 2.0
        self.lg_n = self.league_decay * self.lg_n + 1.0


def game_possessions(box_rows) -> Optional[float]:
    """Posesiones del partido (promedio de los dos equipos) desde los box scores."""
    if not len(box_rows):
        return None
    t = box_rows.groupby("team_id")[["fga", "fta", "tov", "oreb"]].sum()
    if len(t) != 2:
        return None
    return float((t.fga + 0.44 * t.fta + t.tov - t.oreb).mean())


# ══════════════════════════════════════════════════════════════════════
# Estructural
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TeamStrength:
    off: float          # Σ minutos × O / 48 (pts/100 sobre la media)
    deff: float         # Σ minutos × D / 48 (+ = defiende mejor)


def team_strength(minutes: np.ndarray, o: np.ndarray, d: np.ndarray) -> TeamStrength:
    return TeamStrength(float((minutes * o).sum() / 48.0), float((minutes * d).sum() / 48.0))


def structural(h: TeamStrength, a: TeamStrength, pace_h: float, pace_a: float, league_pace: float,
               league_ppp: float, league_eff: float, home_edge: float) -> Tuple[float, float]:
    """Puntos esperados (local, visita)."""
    poss = pace_h * pace_a / league_pace
    scale = poss * league_ppp / 100.0 / league_eff
    hp = scale * (league_eff + h.off - a.deff + home_edge / 2.0)
    ap = scale * (league_eff + a.off - h.deff - home_edge / 2.0)
    return hp, ap


# ══════════════════════════════════════════════════════════════════════
# Parámetros entrenados, corrección y calibración
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ModelParams:
    margin_coef: Dict[str, float] = field(default_factory=dict)
    total_coef: Dict[str, float] = field(default_factory=dict)
    margin_bias: float = 0.0
    total_bias: float = 0.0
    sigma_margin: float = 13.0
    sigma_total: float = 18.0
    rho: float = 0.0                       # correlación entre los errores de margen y total
    iso_x: List[float] = field(default_factory=list)
    iso_y: List[float] = field(default_factory=list)
    trained_until: Optional[str] = None
    metrics: Dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str = PARAMS_PATH) -> "ModelParams":
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            raw = json.load(f)
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    def save(self, path: str = PARAMS_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2, ensure_ascii=False)

    def correction(self, ctx: Dict[str, float]) -> Tuple[float, float]:
        dm = self.margin_bias + sum(c * ctx.get(k, 0.0) for k, c in self.margin_coef.items())
        dt = self.total_bias + sum(c * ctx.get(k, 0.0) for k, c in self.total_coef.items())
        return dm, dt

    def calibrate(self, p: float) -> float:
        if len(self.iso_x) < 2:
            return p
        return float(np.clip(np.interp(p, self.iso_x, self.iso_y), 0.01, 0.99))


def normal_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, float) / math.sqrt(2.0)))


# ══════════════════════════════════════════════════════════════════════
# Simulación con sorteo de disponibilidad (N3)
# ══════════════════════════════════════════════════════════════════════

MAX_EXACT_UNCERTAIN = 8       # hasta 2^8 escenarios se enumeran; con más se sortea


def availability_scenarios(uncertain: Dict[int, float], rng: np.random.Generator,
                           n_draws: int = 400) -> List[Tuple[Dict[int, float], float]]:
    """Escenarios de quién juega: [(prob_de_jugar por jugador (0/1), peso)]."""
    players = list(uncertain)
    if not players:
        return [({}, 1.0)]
    if len(players) <= MAX_EXACT_UNCERTAIN:
        out = []
        for combo in itertools.product((0, 1), repeat=len(players)):
            w = 1.0
            for p, plays in zip(players, combo):
                w *= uncertain[p] if plays else 1.0 - uncertain[p]
            if w > 1e-6:
                out.append(({p: float(c) for p, c in zip(players, combo)}, w))
        s = sum(w for _, w in out)
        return [(sc, w / s) for sc, w in out]
    draws = rng.random((n_draws, len(players))) < np.array([uncertain[p] for p in players])
    return [({p: float(v) for p, v in zip(players, row)}, 1.0 / n_draws) for row in draws]


def simulate_mixture(means: List[Tuple[float, float, float]], params: ModelParams, n: int = 10000,
                     seed: int = 7) -> Dict:
    """means = [(puntos local, puntos visita, peso)] → resumen de 10,000 marcadores."""
    rng = np.random.default_rng(seed)
    w = np.array([m[2] for m in means])
    idx = rng.choice(len(means), size=n, p=w / w.sum())
    mu_m = np.array([m[0] - m[1] for m in means])[idx]
    mu_t = np.array([m[0] + m[1] for m in means])[idx]
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    rho = params.rho
    margin = mu_m + params.sigma_margin * z1
    total = mu_t + params.sigma_total * (rho * z1 + math.sqrt(max(1.0 - rho ** 2, 0.0)) * z2)
    # Sin empates: un margen a menos de medio punto se va a tiempo extra (moneda con el favorito)
    tie = np.abs(margin) < 0.5
    margin[tie] = np.where(rng.random(tie.sum()) < 0.5 + 0.02 * np.sign(mu_m[tie]), 1.0, -1.0) * rng.integers(1, 8, tie.sum())
    home = (total + margin) / 2.0
    away = (total - margin) / 2.0
    raw_p = float((w * normal_cdf(np.array([m[0] - m[1] for m in means]) / params.sigma_margin)).sum() / w.sum())
    p = params.calibrate(raw_p)
    hs, as_ = np.rint(home).astype(int), np.rint(away).astype(int)
    keys, counts = np.unique(np.stack([hs, as_], 1), axis=0, return_counts=True)
    likely = keys[counts.argmax()]
    return {
        "home_win_prob": p,
        "home_win_prob_raw": raw_p,
        "exp_home": float(home.mean()), "exp_away": float(away.mean()),
        "exp_margin": float(margin.mean()), "exp_total": float(total.mean()),
        "margin_p10": float(np.percentile(margin, 10)), "margin_p90": float(np.percentile(margin, 90)),
        "total_p10": float(np.percentile(total, 10)), "total_p90": float(np.percentile(total, 90)),
        "most_likely_score": [int(likely[0]), int(likely[1])],
        "num_simulations": n, "scenarios": len(means),
    }


# ══════════════════════════════════════════════════════════════════════
# Ritmo según quién juega
# ══════════════════════════════════════════════════════════════════════

PACE_PRIOR_MIN = 400.0      # minutos "virtuales" de ritmo de liga para encoger a cada jugador


def player_pace_devs(engine) -> Dict[str, Dict[int, float]]:
    """{temporada: {jugador: desviación de ritmo}} en posesiones por 48 min por equipo.

    Para cada jugador, las posesiones por minuto mientras estuvo en cancha esa temporada
    contra el promedio de liga, encogidas hacia 0 con PACE_PRIOR_MIN minutos. Para predecir
    una temporada se usan solo las anteriores (ver `pace_devs_before`).
    """
    import pandas as pd
    from sqlalchemy import text
    with engine.connect() as conn:
        st = pd.read_sql(text(
            "SELECT g.season, s.start_t, s.end_t, s.home_players, s.away_players, s.home_poss, s.away_poss "
            "FROM nba_stints s JOIN nba_games g USING (game_id)"), conn)
    st["min"] = (st.end_t - st.start_t) / 600.0
    st = st[st["min"] > 0]
    st["poss"] = (st.home_poss + st.away_poss) / 2.0
    out: Dict[str, Dict[int, float]] = {}
    for season, g in st.groupby("season"):
        league = g.poss.sum() / g["min"].sum() * 48.0
        rows = []
        for col in ("home_players", "away_players"):
            ex = g[[col, "min", "poss"]].assign(pid=g[col].str.split(",")).explode("pid")
            rows.append(ex[["pid", "min", "poss"]])
        a = pd.concat(rows).groupby("pid")[["min", "poss"]].sum()
        dev = (a.poss / a["min"] * 48.0 - league) * a["min"] / (a["min"] + PACE_PRIOR_MIN)
        out[season] = {int(p): float(v) for p, v in dev.items()}
    return out


def pace_devs_before(devs: Dict[str, Dict[int, float]], season: str) -> Dict[int, float]:
    """Desviación de ritmo vigente al empezar `season`: la de la temporada anterior más reciente
    en que jugó cada jugador (las más viejas pesan menos)."""
    out: Dict[int, float] = {}
    prior = sorted(s for s in devs if s < season)
    for i, s in enumerate(prior):
        w = 0.6 ** (len(prior) - 1 - i)
        for p, v in devs[s].items():
            out[p] = v * w if p not in out or w == 1.0 else out[p]
    return out
