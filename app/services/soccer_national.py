"""
OmniPicks — Fútbol v2: selecciones nacionales (Nations League de UEFA y Concacaf, y amistosos o
eliminatorias si se agregan).

Datos: todos los partidos internacionales desde 1872 (base pública "international_results",
github.com/martj42), con torneo, sede y si fue en cancha neutral.

Modelo: el mismo Dixon-Coles (ataque y defensa por selección, ajustados por rival, con localía solo
cuando no es cancha neutral), con:
  · descuento por antigüedad más lento que en clubes (las selecciones juegan ~10 partidos al año).
"""
import io
import logging
import os
import time
from typing import Dict, Optional

import httpx
import numpy as np
import polars as pl

from app.services.soccer_dc import DCParams, fit_rates
from app.services.soccer_history import CACHE_DIR

logger = logging.getLogger(__name__)

URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
# Elegidos con el backtest 2018–2021 (ml/scripts/soccer_national_eval.py): vida media ≈ 3.8 años,
# ~8 años de historia; los amistosos pesan igual (bajarles el peso empeoraba).
PARAMS = DCParams(xi=0.0005, ridge=1.0, window_days=2900)
FRIENDLY_WEIGHT = 1.0

ALIASES = {  # ESPN → nombre en la base
    "turkiye": "Turkey", "turkey": "Turkey", "czechia": "Czech Republic", "republic of ireland": "Republic of Ireland",
    "bosnia herzegovina": "Bosnia and Herzegovina", "bosnia and herzegovina": "Bosnia and Herzegovina",
    "usa": "United States", "united states": "United States", "korea republic": "South Korea",
    "north macedonia": "North Macedonia", "faroe islands": "Faroe Islands", "st kitts and nevis": "Saint Kitts and Nevis",
    "st lucia": "Saint Lucia", "st vincent and the grenadines": "Saint Vincent and the Grenadines",
    "st martin": "Saint Martin", "trinidad tobago": "Trinidad and Tobago", "curacao": "Curaçao",
    "us virgin islands": "United States Virgin Islands", "british virgin islands": "British Virgin Islands",
    "antigua barbuda": "Antigua and Barbuda", "antigua and barbuda": "Antigua and Barbuda",
    "turks and caicos islands": "Turks and Caicos Islands", "dr congo": "DR Congo", "ivory coast": "Ivory Coast",
    "cote d ivoire": "Ivory Coast", "cabo verde": "Cape Verde", "sint maarten": "Sint Maarten",
}


def load_results(max_age_hours: float = 12) -> pl.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "international_results.parquet")
    if os.path.exists(path) and os.path.getmtime(path) > time.time() - max_age_hours * 3600:
        return pl.read_parquet(path)
    try:
        r = httpx.get(URL, timeout=60, follow_redirects=True)
        r.raise_for_status()
        df = pl.read_csv(io.StringIO(r.text), schema_overrides={"home_score": pl.Float64, "away_score": pl.Float64})
        df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
                             (pl.col("neutral").cast(pl.Utf8).str.to_uppercase() == "TRUE").alias("neutral"))
        df.write_parquet(path)
        return df
    except Exception as e:
        logger.warning(f"Selecciones: no se pudo bajar el histórico ({type(e).__name__}: {e})")
        return pl.read_parquet(path) if os.path.exists(path) else pl.DataFrame()


def _norm(s: str) -> str:
    import re, unicodedata
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.replace("&", " and ").replace(".", ""))).strip()


def team_name(espn: str, known: set) -> Optional[str]:
    if espn in known:
        return espn
    n = _norm(espn)
    if n in ALIASES and ALIASES[n] in known:
        return ALIASES[n]
    by_norm = {_norm(k): k for k in known}
    return by_norm.get(n)


def fit_national(df: pl.DataFrame, ref: np.datetime64):
    d = df.filter(pl.col("home_score").is_not_null() & (pl.col("date") < pl.lit(ref.astype(object))))
    dates = d["date"].to_numpy().astype("datetime64[D]")
    days = (ref - dates).astype(int).astype(float)
    w = np.where(d["tournament"].to_numpy() == "Friendly", FRIENDLY_WEIGHT, 1.0)
    return fit_rates(d["home_team"].to_list(), d["away_team"].to_list(), d["home_score"].to_numpy(),
                     d["away_score"].to_numpy(), days, PARAMS, rho=-0.05, neutral=d["neutral"].to_numpy(), weight=w)
