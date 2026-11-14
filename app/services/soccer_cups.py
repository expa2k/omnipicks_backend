"""
OmniPicks — Fútbol v2: histórico de Champions y Europa League desde ESPN (para los cruces entre ligas).

ESPN responde el scoreboard por fecha (no por rango), así que se recorre día por día. Las copas
europeas se juegan martes, miércoles y jueves (fase previa en julio y agosto, fase de liga y
eliminatorias de septiembre a mayo). Caché por competencia y temporada en ml/datasets/soccer/.
"""
import logging
import os
import time
from datetime import date, timedelta
from typing import Dict, List

import httpx
import polars as pl

from app.services.soccer_history import CACHE_DIR, current_season_start

logger = logging.getLogger(__name__)

ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer"
CUPS = {"uefa.champions": "UEFA Champions League", "uefa.europa": "UEFA Europa League"}


def _day(slug: str, d: date) -> List[Dict]:
    for attempt in range(3):
        try:
            r = httpx.get(f"{ESPN}/{slug}/scoreboard", params={"dates": d.strftime("%Y%m%d")}, timeout=20)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    else:
        return []
    out = []
    for e in r.json().get("events", []):
        c = e["competitions"][0]
        st = c.get("status", {}).get("type", {})
        teams = {x.get("homeAway"): x for x in c.get("competitors", [])}
        h, a = teams.get("home", {}), teams.get("away", {})
        out.append({
            "event_id": e.get("id"), "date": d, "home": h.get("team", {}).get("displayName"),
            "away": a.get("team", {}).get("displayName"), "home_id": h.get("team", {}).get("id"),
            "away_id": a.get("team", {}).get("id"),
            "hg": float(h["score"]) if st.get("completed") and h.get("score") not in (None, "") else None,
            "ag": float(a["score"]) if st.get("completed") and a.get("score") not in (None, "") else None,
            "neutral": bool(c.get("neutralSite")), "completed": bool(st.get("completed")),
        })
    return out


def load_cup_season(slug: str, season: int, pause: float = 0.25) -> pl.DataFrame:
    """Una temporada de una copa (de julio de `season` a junio siguiente). Las cerradas se guardan."""
    path = os.path.join(CACHE_DIR, f"cup_{slug}_{season}.parquet")
    if os.path.exists(path) and (season < current_season_start() or os.path.getmtime(path) > time.time() - 12 * 3600):
        return pl.read_parquet(path)        # la temporada en curso se vuelve a bajar cada 12 h
    rows: List[Dict] = []
    d, end = date(season, 7, 1), min(date(season + 1, 6, 15), date.today())
    while d <= end:
        if d.weekday() in (1, 2, 3):           # martes, miércoles, jueves
            rows.extend(_day(slug, d))
            time.sleep(pause)
        d += timedelta(days=1)
    df = pl.DataFrame(rows, schema={"event_id": pl.Utf8, "date": pl.Date, "home": pl.Utf8, "away": pl.Utf8,
                                    "home_id": pl.Utf8, "away_id": pl.Utf8, "hg": pl.Float64, "ag": pl.Float64,
                                    "neutral": pl.Boolean, "completed": pl.Boolean})
    df = df.unique("event_id").with_columns(competition=pl.lit(CUPS[slug]), season=pl.lit(season, dtype=pl.Int32))
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.write_parquet(path)
    logger.info(f"ESPN {slug} {season}: {df.height} partidos")
    return df


def load_cups(first_season: int = 2016) -> pl.DataFrame:
    frames = [load_cup_season(s, y) for s in CUPS for y in range(first_season, current_season_start() + 1)]
    return pl.concat([f for f in frames if f.height], how="diagonal_relaxed")


# ══════════════════════════════════════════════════════════════════════
# Nombres de ESPN → nombres de football-data
# ══════════════════════════════════════════════════════════════════════

import re
import unicodedata
from difflib import SequenceMatcher

ALIASES = {  # ESPN (normalizado) → football-data
    "internazionale": "Inter", "inter milan": "Inter", "milan": "Milan", "bayern munich": "Bayern Munich",
    "borussia dortmund": "Dortmund", "borussia monchengladbach": "M'gladbach", "bayer leverkusen": "Leverkusen",
    "eintracht frankfurt": "Ein Frankfurt", "paris saint germain": "Paris SG", "sporting cp": "Sp Lisbon",
    "sporting lisbon": "Sp Lisbon", "braga": "Sp Braga", "sporting braga": "Sp Braga", "real betis": "Betis",
    "real sociedad": "Sociedad", "celta vigo": "Celta", "wolverhampton wanderers": "Wolves",
    "tottenham hotspur": "Tottenham", "psv eindhoven": "PSV Eindhoven", "az alkmaar": "AZ Alkmaar",
    "ajax amsterdam": "Ajax", "feyenoord rotterdam": "Feyenoord", "vitoria guimaraes": "Guimaraes",
    "atletico madrid": "Ath Madrid", "athletic club": "Ath Bilbao", "athletic bilbao": "Ath Bilbao",
    "manchester city": "Man City", "manchester united": "Man United", "newcastle united": "Newcastle",
    "west ham united": "West Ham", "leicester city": "Leicester", "brighton and hove albion": "Brighton",
    "nottingham forest": "Nott'm Forest", "rb leipzig": "RB Leipzig", "1899 hoffenheim": "Hoffenheim",
    "tsg hoffenheim": "Hoffenheim", "vfl wolfsburg": "Wolfsburg", "sc freiburg": "Freiburg", "vfb stuttgart": "Stuttgart",
    "union berlin": "Union Berlin", "fc koln": "FC Koln", "olympique lyonnais": "Lyon", "lyon": "Lyon",
    "olympique de marseille": "Marseille", "marseille": "Marseille", "as monaco": "Monaco", "losc lille": "Lille",
    "lille": "Lille", "stade rennais": "Rennes", "rennes": "Rennes", "ogc nice": "Nice", "rc lens": "Lens",
    "stade brestois 29": "Brest", "brest": "Brest", "as roma": "Roma", "ss lazio": "Lazio", "ssc napoli": "Napoli",
    "napoli": "Napoli", "juventus": "Juventus", "atalanta": "Atalanta", "fiorentina": "Fiorentina", "bologna": "Bologna",
    "villarreal": "Villarreal", "sevilla": "Sevilla", "valencia": "Valencia", "getafe": "Getafe", "granada": "Granada",
    "espanyol": "Espanol", "girona": "Girona", "benfica": "Benfica", "porto": "Porto", "fc porto": "Porto",
    "twente": "Twente", "fc twente": "Twente", "utrecht": "Utrecht", "fc utrecht": "Utrecht", "pec zwolle": "Zwolle",
    "vitesse": "Vitesse", "rangers": None, "celtic": None, "real madrid": "Real Madrid", "barcelona": "Barcelona",
    "arsenal": "Arsenal", "chelsea": "Chelsea", "liverpool": "Liverpool", "everton": "Everton",
    "aston villa": "Aston Villa", "burnley": "Burnley", "west bromwich albion": "West Brom",
    "brighton hove albion": "Brighton", "fc cologne": "FC Koln", "hertha berlin": "Hertha",
    "saint etienne": "St Etienne", "vitoria de guimaraes": "Guimaraes", "fc augsburg": "Augsburg",
    "werder bremen": "Werder Bremen", "hamburg sv": "Hamburg", "athletic club bilbao": "Ath Bilbao",
    "coventry city": "Coventry", "hull city": "Hull", "ipswich town": "Ipswich", "leeds united": "Leeds",
    "deportivo": "La Coruna", "deportivo la coruna": "La Coruna", "racing santander": "Santander",
    "rayo vallecano": "Vallecano", "sc paderborn 07": "Paderborn", "aj auxerre": "Auxerre",
    "atletico de san luis": "Atl. San Luis", "guadalajara": "Guadalajara Chivas", "chivas": "Guadalajara Chivas",
    "pumas unam": "UNAM Pumas", "santos": "Santos Laguna", "santos laguna": "Santos Laguna",
}
_DROP = {"fc", "cf", "ac", "sc", "afc", "ssc", "as", "rc", "cd", "ud", "sv", "vfl", "vfb", "tsg", "1", "club", "de", "the"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s.replace("'", "").replace(".", " ").replace("-", " "))
    return re.sub(r"\s+", " ", s).strip()


def _core(s: str) -> str:
    return " ".join(t for t in _norm(s).split() if t not in _DROP)


def match_names(espn_names, fd_names) -> Dict[str, str]:
    """ESPN → football-data. Solo empareja con mucha confianza; lo demás queda fuera (grupo "otras ligas")."""
    fd = list(fd_names)
    fd_core = {f: _core(f) for f in fd}
    out = {}
    for e in espn_names:
        n = _norm(e)
        if n in ALIASES:
            if ALIASES[n] in fd:
                out[e] = ALIASES[n]
            continue
        c = _core(e)
        exact = [f for f, fc in fd_core.items() if fc == c or _norm(f) == n]
        if exact:
            out[e] = exact[0]
            continue
        best, score = None, 0.0
        for f, fc in fd_core.items():
            r = SequenceMatcher(None, c, fc).ratio()
            if r > score:
                best, score = f, r
        if best is not None and score >= 0.88:
            out[e] = best
    return out
