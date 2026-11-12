"""
OmniPicks — NFL: quién juega de QB hoy, con las lesiones y el depth chart de ESPN.

nflverse publica un titular proyectado por partido, pero puede ir atrasado: si el QB se lesiona
a media semana, sigue apareciendo. Aquí se cruza con ESPN:

  1. candidatos, en orden: el titular de nflverse y luego el depth chart de ESPN;
  2. se descartan los que están fuera (Out, IR, suspendidos…);
  3. el primero que queda es el titular. Si está en duda, se mezcla con el siguiente según la
     probabilidad de que juegue (Questionable 80%, Doubtful 20%: valores típicos de la NFL, no
     aprendidos todavía).

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
PLAY_PROB = {"questionable": 0.80, "doubtful": 0.20}


def play_probability(status: Optional[str]) -> float:
    s = (status or "active").strip().lower()
    if s in OUT_STATUSES or s.startswith("injured reserve"):
        return 0.0
    return PLAY_PROB.get(s, 1.0)


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
    if df is None:
        try:
            import nflreadpy as nfl
            df = nfl.load_players().select(["gsis_id", "espn_id", "display_name", "position"])
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

    def __init__(self):
        self.injuries = fetch_injuries()
        self.e2g, self.names = espn_to_gsis()
        self.g2e = {g: e for e, g in self.e2g.items()}
        self._depth: Dict[str, List[str]] = {}

    def status_of(self, gsis: Optional[str]) -> Optional[str]:
        inj = self.injuries.get(self.g2e.get(gsis, ""), {})
        return inj.get("status")

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
            p = play_probability(self.status_of(q) if not q.startswith("espn:") else
                                 self.injuries.get(q[5:], {}).get("status"))
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
            "starter_prob": round(mix[0][1], 2),
            "backup_name": name(mix[1][0]) if len(mix) > 1 else None,
            "projected_name": projected_name or name(projected),
            "changed": starter != projected,
            "skipped": [{"name": name(q), "status": self.status_of(q)} for q in skipped if q],
        }
