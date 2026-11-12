"""
OmniPicks — NFL: quién juega de QB hoy, con las lesiones y el depth chart de ESPN.

nflverse publica un titular proyectado por partido, pero puede ir atrasado: si el QB se lesiona
a media semana, sigue apareciendo. Aquí se cruza con ESPN:

  1. candidatos, en orden: el titular de nflverse y luego el depth chart de ESPN;
  2. se descartan los que están fuera (Out, IR, suspendidos…);
  3. el primero que queda es el titular. Si está en duda, se mezcla con el siguiente según la
     probabilidad de que juegue, aprendida de los reportes oficiales 2015–2025 (`learn_play_probs`):
     depende del estado y de cómo entrenó en la semana (nflverse publica la práctica a diario).
     Un QB titular Questionable arrancó 60% de las veces (92% si entrenó completo, 53% limitado,
     45% sin entrenar); Doubtful, 7%.

Los ids de ESPN se pasan a los de nflverse (gsis) con la tabla de jugadores de nflverse.
"""
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import httpx
import polars as pl

from app.services.nfl_history import CACHE_DIR, current_season

logger = logging.getLogger(__name__)

ESPN_SITE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
ESPN_CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"

OUT_STATUSES = {"out", "injured reserve", "ir", "suspension", "suspended", "physically unable to perform", "pup",
                "non-football injury", "nfi", "reserve/covid-19"}
# Valores de respaldo (los aprendidos se guardan en nfl_v2.json al entrenar)
DEFAULT_PLAY_PROBS = {"questionable": {"full": 0.92, "limited": 0.53, "dnp": 0.45, "any": 0.60},
                      "doubtful": {"any": 0.07}}


def practice_key(practice: Optional[str]) -> Optional[str]:
    p = (practice or "").lower()
    if p.startswith("full"):
        return "full"
    if p.startswith("limited"):
        return "limited"
    if p.startswith("did not"):
        return "dnp"
    return None


def play_probability(status: Optional[str], practice: Optional[str] = None, table: Optional[Dict] = None) -> float:
    s = (status or "active").strip().lower()
    if s in OUT_STATUSES or s.startswith("injured reserve"):
        return 0.0
    t = (table or DEFAULT_PLAY_PROBS).get(s)
    if not t:
        return 1.0
    return float(t.get(practice_key(practice) or "any", t.get("any", 1.0)))


def learn_play_probs(first: int = 2015, last: Optional[int] = None, smooth: float = 5.0) -> Dict:
    """Qué tan seguido arrancó el QB titular según su estado en el reporte final y su práctica.

    Solo cuenta al titular habitual (fue titular el partido anterior). Cada casilla se encoge
    hacia la tasa general de su estado con `smooth` casos virtuales."""
    import nflreadpy as nfl
    from app.services.nfl_history import load_schedules, normalize_teams
    last = last or current_season() - 1
    inj = nfl.load_injuries(list(range(first, last + 1))).filter(
        (pl.col("position") == "QB") & pl.col("report_status").is_in(["Questionable", "Doubtful"]))
    inj = inj.with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
                           pl.col("team").replace({"OAK": "LV", "SD": "LAC", "STL": "LA"}))
    sch = normalize_teams(load_schedules(), cols=("home_team", "away_team")).filter(
        pl.col("season").is_between(first, last))
    starts = pl.concat([
        sch.select("season", "week", pl.col("home_team").alias("team"), pl.col("home_qb_id").alias("starter")),
        sch.select("season", "week", pl.col("away_team").alias("team"), pl.col("away_qb_id").alias("starter"))])
    starts = starts.sort(["team", "season", "week"]).with_columns(prev=pl.col("starter").shift(1).over(["team", "season"]))
    j = inj.join(starts, on=["season", "week", "team"], how="inner").filter(pl.col("prev") == pl.col("gsis_id"))
    j = j.with_columns(started=(pl.col("starter") == pl.col("gsis_id")).cast(pl.Float64))
    out, counts = {}, {}
    for status in ("Questionable", "Doubtful"):
        sub = j.filter(pl.col("report_status") == status)
        if sub.height == 0:
            continue
        base = float(sub["started"].mean())
        cell = {"any": round(base, 3)}
        counts[status.lower()] = {"any": sub.height}
        for r in sub.with_columns(k=pl.col("practice_status").map_elements(practice_key, return_dtype=pl.Utf8)) \
                .group_by("k").agg(n=pl.len(), y=pl.col("started").sum()).iter_rows(named=True):
            if r["k"]:
                cell[r["k"]] = round((r["y"] + smooth * base) / (r["n"] + smooth), 3)
                counts[status.lower()][r["k"]] = r["n"]
        out[status.lower()] = cell
    return {"table": out, "counts": counts, "seasons": [first, last]}


def current_practice() -> Dict[str, Dict]:
    """gsis id → último reporte de la semana en curso (estado oficial y práctica), de nflverse."""
    try:
        import nflreadpy as nfl
        inj = nfl.load_injuries([current_season()])
        wk = inj["week"].max()
        cur = inj.filter((pl.col("week") == wk) & (pl.col("position") == "QB"))
        return {r["gsis_id"]: {"report": r["report_status"], "practice": r["practice_status"]}
                for r in cur.iter_rows(named=True) if r["gsis_id"]}
    except Exception as e:
        logger.warning(f"nflverse: sin reporte de práctica de la semana ({type(e).__name__}: {e})")
        return {}


def _get(url: str, tries: int = 3, **params) -> Optional[dict]:
    for attempt in range(tries):
        try:
            r = httpx.get(url, params=params or None, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == tries - 1:
                logger.warning(f"ESPN NFL: {url} no respondió ({type(e).__name__}: {e})")
            else:
                time.sleep(1.5 * (attempt + 1))
    return None


def _athlete_id(entry: dict) -> Optional[str]:
    a = entry.get("athlete") or {}
    if a.get("id"):
        return str(a["id"])
    for src in [a.get("$ref", "")] + [l.get("href", "") for l in a.get("links", [])]:
        m = re.search(r"athletes/(\d+)", src) or re.search(r"/id/(\d+)", src)
        if m:
            return m.group(1)
    return None


def fetch_injuries() -> Dict[str, Dict]:
    """ESPN id del jugador → {estado, nombre, equipo, posición, comentario}."""
    d = _get(f"{ESPN_SITE}/injuries") or {}
    out = {}
    for team in d.get("injuries", []):
        for i in team.get("injuries", []):
            aid = _athlete_id(i)
            if not aid:
                continue
            a = i.get("athlete") or {}
            out[aid] = {"status": i.get("status"), "name": a.get("displayName"), "team": team.get("displayName"),
                        "position": (a.get("position") or {}).get("abbreviation"), "comment": i.get("shortComment")}
    return out


_TEAM_IDS: Dict[str, str] = {}


def espn_team_ids() -> Dict[str, str]:
    """Nombre del equipo (ESPN) → id de ESPN."""
    if not _TEAM_IDS:
        d = _get(f"{ESPN_SITE}/teams") or {}
        for t in ((d.get("sports") or [{}])[0].get("leagues") or [{}])[0].get("teams", []):
            _TEAM_IDS[t["team"]["displayName"]] = str(t["team"]["id"])
    return _TEAM_IDS


def fetch_depth_qbs(team_name: str, season: Optional[int] = None) -> List[str]:
    """ESPN ids de los QB del equipo en el orden del depth chart."""
    tid = espn_team_ids().get(team_name)
    if not tid:
        return []
    d = _get(f"{ESPN_CORE}/seasons/{season or current_season()}/teams/{tid}/depthcharts") or {}
    for item in d.get("items", []):
        qb = (item.get("positions") or {}).get("qb")
        if qb:
            ranked = sorted(qb.get("athletes", []), key=lambda a: a.get("rank", 99))
            return [x for x in (_athlete_id(a) for a in ranked) if x]
    return []


_PLAYERS: Dict[str, object] = {}


def espn_to_gsis() -> Tuple[Dict[str, str], Dict[str, str]]:
    """(ESPN id → gsis id, gsis id → nombre) con la tabla de jugadores de nflverse (caché diaria)."""
    if _PLAYERS.get("at", 0) > time.time() - 86400:
        return _PLAYERS["map"], _PLAYERS["names"]
    path = os.path.join(CACHE_DIR, "players.parquet")
    df = None
    if os.path.exists(path) and os.path.getmtime(path) > time.time() - 86400:
        df = pl.read_parquet(path)
        if "pfr_id" not in df.columns:
            df = None
    if df is None:
        try:
            import nflreadpy as nfl
            df = nfl.load_players().select(["gsis_id", "espn_id", "pfr_id", "display_name", "position"])
            os.makedirs(CACHE_DIR, exist_ok=True)
            df.write_parquet(path)
        except Exception as e:
            logger.warning(f"nflverse: no se pudo bajar la tabla de jugadores ({type(e).__name__}: {e})")
            df = pl.read_parquet(path) if os.path.exists(path) else pl.DataFrame(
                {"gsis_id": [], "espn_id": [], "display_name": [], "position": []})
    m = {str(e): g for e, g in zip(df["espn_id"].to_list(), df["gsis_id"].to_list()) if e and g}
    names = {g: n for g, n in zip(df["gsis_id"].to_list(), df["display_name"].to_list()) if g}
    _PLAYERS.update({"at": time.time(), "map": m, "names": names})
    return m, names


class QBStatus:
    """Estado de los QB de la jornada. Una instancia por corrida de predicciones."""

    def __init__(self, play_table: Optional[Dict] = None):
        self.table = play_table
        self.practice = current_practice()
        self.injuries = fetch_injuries()
        self.e2g, self.names = espn_to_gsis()
        self.g2e = {g: e for e, g in self.e2g.items()}
        self._depth: Dict[str, List[str]] = {}

    def status_of(self, gsis: Optional[str]) -> Optional[str]:
        """Estado: el del reporte oficial de nflverse si ya salió; si no, el de ESPN."""
        official = (self.practice.get(gsis) or {}).get("report")
        if official:
            return official
        inj = self.injuries.get(self.g2e.get(gsis, ""), {})
        return inj.get("status")

    def prob_of(self, q: str) -> float:
        if q.startswith("espn:"):
            return play_probability(self.injuries.get(q[5:], {}).get("status"), None, self.table)
        return play_probability(self.status_of(q), (self.practice.get(q) or {}).get("practice"), self.table)

    def lineup(self, team_name: str, projected: Optional[str], projected_name: Optional[str] = None) -> Dict:
        """Mezcla de QB para el partido: [(gsis, probabilidad)], más lo que se muestra en la app."""
        if team_name not in self._depth:
            self._depth[team_name] = fetch_depth_qbs(team_name)
        depth = [self.e2g.get(e) or f"espn:{e}" for e in self._depth[team_name]]
        cands = []
        for q in [projected] + depth:
            if q and q not in cands:
                cands.append(q)
        mix, rest, skipped = [], 1.0, []
        for q in cands:
            p = self.prob_of(q)
            if p <= 0:
                skipped.append(q)
                continue
            mix.append((q, rest * p))
            rest *= (1 - p)
            if rest < 0.01 or len(mix) == 2:
                break
        if not mix:
            mix = [(projected, 1.0)]
        elif rest >= 0.01 and len(mix) == 1:
            mix.append((None, rest))          # en duda y sin suplente conocido: suplente genérico
        total = sum(p for _, p in mix)
        mix = [(q, p / total) for q, p in mix]
        name = lambda q: "suplente (sin identificar)" if q is None else (projected_name if q == projected and projected_name else
                          self.names.get(q) or (self.injuries.get(q[5:], {}).get("name") if q and q.startswith("espn:") else None) or q)
        starter = mix[0][0]
        return {
            "mix": [(None if q and q.startswith("espn:") else q, p) for q, p in mix],
            "starter": starter, "starter_name": name(starter), "starter_status": self.status_of(starter) or "Active",
            "starter_practice": (self.practice.get(starter) or {}).get("practice"),
            "starter_prob": round(mix[0][1], 2),
            "backup_name": name(mix[1][0]) if len(mix) > 1 else None,
            "projected_name": projected_name or name(projected),
            "changed": starter != projected,
            "skipped": [{"name": name(q), "status": self.status_of(q)} for q in skipped if q],
        }
