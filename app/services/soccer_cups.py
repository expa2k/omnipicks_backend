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
    if os.path.exists(path) and season < current_season_start():
        return pl.read_parquet(path)
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
