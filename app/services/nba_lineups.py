"""
OmniPicks — NBA v2 (N2): quién juega, cuántos minutos y cuánto pesa cada baja.

Piezas (plan, sección 3):
  · 3a  estado de cada jugador: fuera, de regreso, duda (con probabilidad) o normal
  · 3b  reparto de minutos: los minutos de quien falta se reparten entre los disponibles,
        en proporción a lo que ya juegan y con tope por jugador
  ·     fuerza del equipo = Σ minutos × rating (N1) / 48
  ·     impacto de una baja = fuerza con él − fuerza sin él, en puntos de margen

RosterTracker se alimenta en orden cronológico con los box scores; para producción
basta darle los últimos ~60 días.
"""
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np

GAME_MINUTES = 240.0
RECENT_GAMES = 5            # "promedio reciente" (60%)
ROLE_GAMES = 20             # "rol en la rotación" (40%)
TEAM_WINDOW = 10            # partidos del equipo para la tasa de participación
ROSTER_MAX_DAYS = 200       # un jugador sin jugar más de esto deja de contarse en el equipo
RETURN_FACTOR = 0.75        # minutos en el regreso de una ausencia larga
CAP_EXTRA = 6.0             # nadie sube más de esto sobre sus minutos habituales…
CAP_MAX = 40.0              # …ni pasa de aquí en tiempo regular
ROTATION_MIN = 15.0         # jugador "de rotación" (para detectar bajas en el histórico)
LEARN_GAMES = 60            # partidos del equipo para aprender el reparto de minutos
LEARN_DAYS = 150
LEARN_SHRINK = 15.0
REPL_DECAY = 0.985          # por partido del equipo (~45 partidos de vida media)
REPL_PRIOR = 2.0            # peso del parecido por posición y estatura, en sustituciones


@dataclass
class Candidate:
    player: int                 # índice o id del jugador (lo que se le dio al tracker)
    base: float                 # minutos habituales cuando juega (60% reciente + 40% rol)
    rate: float                 # fracción de los últimos partidos del equipo en que jugó
    games: int                  # partidos con el equipo en la ventana


@dataclass
class RosterTracker:
    """Estado de las plantillas a partir de los box scores vistos hasta ahora."""
    player_games: Dict[int, deque] = field(default_factory=dict)       # jugador → (fecha, equipo, minutos)
    team_games: Dict[int, deque] = field(default_factory=dict)         # equipo → (fecha, {jugadores})
    team_minutes: Dict[int, deque] = field(default_factory=dict)       # equipo → (fecha, {jugador: minutos})
    replacements: Dict[int, Dict[int, Dict[int, float]]] = field(default_factory=dict)   # equipo → sale → entra → peso

    def add_substitutions(self, team: int, pairs):
        """Suma las sustituciones de un partido del equipo, descontando las viejas."""
        rep = self.replacements.setdefault(team, {})
        for row in rep.values():
            for k in row:
                row[k] *= REPL_DECAY
        for out_p, in_p, w in pairs:
            row = rep.setdefault(int(out_p), {})
            row[int(in_p)] = row.get(int(in_p), 0.0) + w

    def update(self, day: date, team: int, players, minutes):
        played = set()
        for p, m in zip(players, minutes):
            if m > 0:
                self.player_games.setdefault(int(p), deque(maxlen=ROLE_GAMES)).append((day, team, float(m)))
                played.add(int(p))
        self.team_games.setdefault(team, deque(maxlen=TEAM_WINDOW)).append((day, played))
        self.team_minutes.setdefault(team, deque(maxlen=LEARN_GAMES)).append(
            (day, {int(p): float(m) for p, m in zip(players, minutes) if m > 0}))

    def learned_shift(self, team: int, day: date, absent: List[int], k: float = LEARN_SHRINK) -> Dict[int, float]:
        """3b aprendido: minutos de más (o de menos) de cada compañero cuando faltó cada ausente.

        Compara, en los partidos recientes del equipo, los minutos de cada compañero con y
        sin el ausente; se encoge hacia 0 según cuántos partidos sin él hay (k = peso del
        reparto proporcional).
        """
        hist = [g for g in self.team_minutes.get(team, ()) if (day - g[0]).days <= LEARN_DAYS]
        shift: Dict[int, float] = {}
        for x in absent:
            without = [m for _, m in hist if x not in m]
            with_x = [m for _, m in hist if x in m]
            if not without or len(with_x) < 3:
                continue
            w = len(without) / (len(without) + k)
            mates = {p for _, m in hist for p in m if p != x}
            for p in mates:
                a = sum(m.get(p, 0.0) for m in without) / len(without)
                b = sum(m.get(p, 0.0) for m in with_x) / len(with_x)
                shift[p] = shift.get(p, 0.0) + w * (a - b)
        return shift

    def candidates(self, team: int, day: date) -> List[Candidate]:
        """Jugadores cuyo último partido fue con este equipo, con sus minutos habituales."""
        window = [g for g in self.team_games.get(team, ()) if (day - g[0]).days <= ROSTER_MAX_DAYS]
        out = []
        for p, games in self.player_games.items():
            last_day, last_team, _ = games[-1]
            if last_team != team or (day - last_day).days > ROSTER_MAX_DAYS:
                continue
            mins = [m for (_, t, m) in games]
            base = 0.6 * float(np.mean(mins[-RECENT_GAMES:])) + 0.4 * float(np.mean(mins))
            with_team = sum(1 for _, pl in window if p in pl)
            rate = with_team / len(window) if window else 0.0
            out.append(Candidate(p, base, rate, with_team))
        return out


# ══════════════════════════════════════════════════════════════════════
# 3a + 3b: minutos esperados
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ReplacementContext:
    """A quién van los minutos de cada jugador: lo observado en el equipo y, de respaldo,
    el parecido por posición y estatura."""
    observed: Dict[int, Dict[int, float]]                  # sale → entra → peso (del tracker)
    height: Dict[int, float] = field(default_factory=dict)
    position: Dict[int, str] = field(default_factory=dict)
    prior: float = REPL_PRIOR

    def similarity(self, x: int, y: int) -> float:
        hx, hy = self.height.get(x), self.height.get(y)
        s = float(np.exp(-((hx - hy) / 3.0) ** 2)) if hx and hy else 0.5
        px, py = self.position.get(x) or "", self.position.get(y) or ""
        if px and py:
            s *= 1.0 if set(px.split("-")) & set(py.split("-")) else 0.4
        return s

    def shares(self, x: int, ys: List[int]) -> np.ndarray:
        """Fracción de los minutos de x que va a cada y disponible."""
        row = self.observed.get(x, {})
        obs = np.array([row.get(y, 0.0) for y in ys])
        sim = np.array([self.similarity(x, y) for y in ys])
        sim = sim / sim.sum() if sim.sum() > 0 else np.full(len(ys), 1.0 / max(len(ys), 1))
        w = obs + self.prior * sim
        return w / w.sum() if w.sum() > 0 else sim

def expected_minutes(cands: List[Candidate], play_prob: Optional[Dict[int, float]] = None,
                     returning: Optional[set] = None, shift: Optional[Dict[int, float]] = None,
                     ctx: Optional["ReplacementContext"] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Minutos esperados de cada candidato para un partido. Devuelve (minutos, tope).

    play_prob: probabilidad de jugar de quienes están en el reporte (0 = fuera). Quien no
    aparece ahí juega según su participación reciente. `returning`: jugadores que vuelven
    de una ausencia larga (juegan con minutos restringidos).
    """
    play_prob = play_prob or {}
    returning = returning or set()
    want = np.zeros(len(cands))
    cap = np.zeros(len(cands))
    full = np.zeros(len(cands))
    for i, c in enumerate(cands):
        if c.player in returning:
            want[i] = RETURN_FACTOR * c.base * play_prob.get(c.player, 1.0)
        elif c.player in play_prob:
            # En el reporte: si juega, un jugador de rotación recupera su rol completo (aunque
            # lleve tiempo fuera); un suplente sigue dependiendo de cuánto lo usan
            role = 1.0 if c.base >= ROTATION_MIN else max(c.rate, 0.5)
            want[i] = c.base * role * play_prob[c.player]
        else:
            want[i] = c.base * c.rate
        cap[i] = min(CAP_MAX, c.base + CAP_EXTRA)
        # Minutos que tendría disponible del todo (para saber cuántos deja libres)
        full[i] = c.base * (1.0 if c.base >= ROTATION_MIN else max(c.rate, 0.5)) if (c.player in play_prob or c.player in returning) else c.base * c.rate
        if shift and want[i] > 0 and c.player in shift:
            # Lo aprendido sobre a quién van los minutos; el tope sube con lo aprendido
            want[i] = max(0.0, want[i] + shift[c.player])
            cap[i] = min(CAP_MAX, max(cap[i], want[i]))
    if ctx is not None:
        # 3b por reemplazos: los minutos libres de cada ausente van a quienes suelen entrar por él
        free = np.maximum(full - want, 0.0)
        avail = np.flatnonzero(want > 0)
        if len(avail) and free.sum() > 0:
            ys = [cands[j].player for j in avail]
            add = np.zeros(len(cands))
            for i in np.flatnonzero(free > 0.5):
                add[avail] += free[i] * ctx.shares(cands[i].player, ys)
            want = want + add
            cap = np.minimum(CAP_MAX, np.maximum(cap, want))
    return fill_minutes(want, cap), cap


def fill_minutes(want: np.ndarray, cap: np.ndarray, total: float = GAME_MINUTES) -> np.ndarray:
    """Escala los minutos deseados hasta sumar `total`, sin que nadie pase de su tope."""
    want = np.asarray(want, float)
    cap = np.where(want > 0, np.asarray(cap, float), 0.0)
    if want.sum() <= 0:
        return np.zeros_like(want)
    if cap.sum() <= total:                      # no alcanza ni con todos al tope: se relaja el tope
        return want * total / want.sum() if cap.sum() <= 0 else cap * total / cap.sum()
    lo, hi = 0.0, 1.0
    while np.minimum(cap, hi * want).sum() < total:
        hi *= 2.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if np.minimum(cap, mid * want).sum() < total:
            lo = mid
        else:
            hi = mid
    return np.minimum(cap, hi * want)


def team_net(minutes: np.ndarray, total_rating: np.ndarray) -> float:
    """Rating neto del equipo en pts/100 posesiones: Σ minutos × (O + D) / 48."""
    return float((minutes * total_rating).sum() / 48.0)


# ══════════════════════════════════════════════════════════════════════
# Impacto de una baja
# ══════════════════════════════════════════════════════════════════════

def absence_impact(cands: List[Candidate], total_rating: np.ndarray, player: int, pace: float = 0.97,
                   other_probs: Optional[Dict[int, float]] = None, ctx: Optional[ReplacementContext] = None,
                   returning: Optional[set] = None) -> Optional[Dict]:
    """Cuánto cambia el margen esperado del equipo si `player` no juega.

    total_rating[i] es O + D del candidato i. Devuelve los puntos de margen (negativo =
    el equipo empeora) y a quién se van sus minutos.
    """
    idx = next((i for i, c in enumerate(cands) if c.player == player), None)
    if idx is None:
        return None
    others = dict(other_probs or {})
    others.pop(player, None)
    with_min, _ = expected_minutes(cands, {**others, player: 1.0}, returning, ctx=ctx)
    without_min, _ = expected_minutes(cands, {**others, player: 0.0}, returning, ctx=ctx)
    delta = without_min - with_min
    gains = sorted(((cands[i].player, float(delta[i])) for i in range(len(cands)) if i != idx and delta[i] > 0.5),
                   key=lambda x: -x[1])
    return {
        "player": player,
        "minutes": float(with_min[idx]),
        "margin": pace * (team_net(without_min, total_rating) - team_net(with_min, total_rating)),
        "minutes_to": gains[:3],
    }


# ══════════════════════════════════════════════════════════════════════
# Estado actual desde la base (producción)
# ══════════════════════════════════════════════════════════════════════

_ESPN_TEAM_ALIASES = {"la clippers": "los angeles clippers"}


def team_ids_by_name() -> Dict[str, int]:
    """Nombre de equipo (como lo escribe ESPN, en minúsculas) → id de la NBA."""
    from nba_api.stats.static import teams
    out = {t["full_name"].lower(): int(t["id"]) for t in teams.get_teams()}
    for alias, name in _ESPN_TEAM_ALIASES.items():
        out[alias] = out[name]
    return out


def load_tracker(engine, today: date, days: int = ROSTER_MAX_DAYS) -> Tuple[RosterTracker, Dict[int, str]]:
    """Plantillas al día de hoy a partir de los box scores recientes. Devuelve (tracker, nombres)."""
    import pandas as pd
    from datetime import timedelta
    from sqlalchemy import text

    with engine.connect() as conn:
        box = pd.read_sql(text(
            "SELECT game_id, game_date, team_id, player_id, player_name, minutes FROM nba_player_games "
            "WHERE game_date >= :a AND game_date < :b AND minutes > 0 ORDER BY game_date, game_id"),
            conn, params={"a": today - timedelta(days=days), "b": today})
    from app.services.nba_history import load_substitutions
    subs = load_substitutions(box["game_id"].drop_duplicates().tolist())
    tracker = RosterTracker()
    for (gid, tid), g in box.groupby(["game_id", "team_id"], sort=False):
        d = g["game_date"].iloc[0]
        tracker.update(d if isinstance(d, date) else d.date(), int(tid), g["player_id"].tolist(), g["minutes"].tolist())
        tracker.add_substitutions(int(tid), [(o, n, w) for (t, o, n, w) in subs.get(gid, []) if t == int(tid)])
    names = dict(zip(box["player_id"].astype(int), box["player_name"]))
    return tracker, names


def injury_impacts(engine, today: date) -> List[Dict]:
    """Impacto en puntos de margen de cada jugador del reporte de lesiones vigente.

    Usa los ratings guardados más recientes (nba_player_ratings) y el reporte vigente
    (nba_injury_reports). Para cada jugador compara al equipo con él y sin él, dejando
    al resto del reporte con su probabilidad de jugar.
    """
    import pandas as pd
    from sqlalchemy import text
    from app.services.nba_injuries import PLAY_PROBABILITY, norm_name
    from app.services.nba_ratings import MODEL_VERSION

    tracker, names = load_tracker(engine, today)
    with engine.connect() as conn:
        ratings = pd.read_sql(text(
            "SELECT player_id, o_rating + d_rating AS total FROM nba_player_ratings WHERE model_version = :v "
            "AND as_of_date = (SELECT MAX(as_of_date) FROM nba_player_ratings WHERE model_version = :v AND as_of_date <= :d)"),
            conn, params={"v": MODEL_VERSION, "d": today})
        inj = pd.read_sql(text(
            "SELECT player_name, team_name, status, injury_type FROM nba_injury_reports WHERE is_current = 1"), conn)
    total = dict(zip(ratings["player_id"].astype(int), ratings["total"].astype(float)))
    replacement = float(ratings["total"].quantile(0.15)) if len(ratings) else -3.0
    team_id = team_ids_by_name()

    out = []
    for team_name, grp in inj.groupby("team_name"):
        tid = team_id.get(team_name.lower())
        if tid is None:
            continue
        cands = tracker.candidates(tid, today)
        by_name = {norm_name(names.get(c.player, "")): c.player for c in cands}
        probs, rows = {}, []
        for r in grp.itertuples():
            pid = by_name.get(norm_name(r.player_name))
            if pid is not None:
                probs[pid] = PLAY_PROBABILITY.get(r.status, 0.5)
                rows.append((pid, r))
        rating = np.array([total.get(c.player, replacement) for c in cands])
        for pid, r in rows:
            imp = absence_impact(cands, rating, pid, other_probs=probs)
            if imp is None or imp["minutes"] < 1.0:
                continue                                    # lleva tanto fuera que ya no cuenta en la rotación
            out.append({
                "team": team_name, "player": r.player_name, "status": r.status, "injury": r.injury_type,
                "play_probability": probs[pid], "usual_minutes": round(imp["minutes"], 1),
                "margin_without": round(imp["margin"], 2),
                "minutes_to": [(names.get(p, str(p)), round(m, 1)) for p, m in imp["minutes_to"]],
            })
    return sorted(out, key=lambda x: x["margin_without"])


# ══════════════════════════════════════════════════════════════════════
# Modelo de minutos con LightGBM (lo entrena ml/scripts/nba_minutes_model.py)
# ══════════════════════════════════════════════════════════════════════

import os as _os

MINUTES_MODEL_PATH = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "ml", "saved_models", "nba_minutes.txt"))
MINUTES_FEATURES = [
    "base", "rate", "games", "last_min", "mean5", "mean20", "days_since", "team_rest", "returning",
    "m_prop", "m_rep", "rep_gain", "team_free", "base_rank", "n_cands", "rating", "rating_rank",
]


def minutes_features(tracker: RosterTracker, cands: List[Candidate], day: date, outs: Dict[int, float],
                     returning: set, m_prop: np.ndarray, m_rep: np.ndarray, rating: np.ndarray,
                     team_rest: float) -> np.ndarray:
    """Variables por jugador para el modelo de minutos (mismo orden que MINUTES_FEATURES)."""
    n = len(cands)
    base = np.array([c.base for c in cands])
    base_rank = (-base).argsort().argsort().astype(float)
    rating_rank = (-np.asarray(rating, float)).argsort().argsort().astype(float)
    team_free = float(sum(c.base for c in cands if c.player in outs))
    rows = np.zeros((n, len(MINUTES_FEATURES)))
    for i, c in enumerate(cands):
        games = tracker.player_games.get(c.player, ())
        mins = [m for (_, _, m) in games]
        last_day = games[-1][0] if games else day
        rows[i] = [
            c.base, c.rate, c.games, mins[-1] if mins else 0.0,
            float(np.mean(mins[-5:])) if mins else 0.0, float(np.mean(mins)) if mins else 0.0,
            min((day - last_day).days, 60), min(team_rest, 7), float(c.player in returning),
            m_prop[i], m_rep[i], m_rep[i] - m_prop[i], team_free, base_rank[i], n, rating[i], rating_rank[i],
        ]
    return rows


class MinutesModel:
    """Carga el LightGBM guardado y predice los minutos de un equipo (240 en total)."""

    def __init__(self, path: str = MINUTES_MODEL_PATH):
        import lightgbm as lgb
        self.booster = lgb.Booster(model_file=path)

    @classmethod
    def load(cls) -> Optional["MinutesModel"]:
        if not _os.path.exists(MINUTES_MODEL_PATH):
            return None
        try:
            return cls()
        except Exception:
            return None

    def predict(self, X: np.ndarray, out_mask: np.ndarray, play_prob: Optional[np.ndarray] = None) -> np.ndarray:
        pred = np.clip(self.booster.predict(X), 0.0, 44.0)
        if play_prob is not None:
            pred = pred * play_prob
        pred[out_mask] = 0.0
        return fill_minutes(np.where(pred > 0.5, pred, 0.0), np.full(len(pred), 48.0))
