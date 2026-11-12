"""
OmniPicks — NFL: histórico de jugadas y calendario (nflverse, gratis).

nflverse publica todas las jugadas desde 1999 con su EPA (puntos esperados que suma o resta cada
jugada) y el calendario con resultados, descanso, techo, clima, QB titular y líneas de cierre.
Aquí se descarga y se guarda en caché (ml/datasets/nfl/) solo lo que usa el modelo.
"""
import logging
import os
from datetime import date
from typing import Iterable, List

import polars as pl

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ml", "datasets", "nfl"))
FIRST_SEASON = 2010

PBP_COLUMNS = [
    "game_id", "season", "week", "season_type", "game_date", "posteam", "defteam", "home_team", "away_team",
    "play_type", "pass", "rush", "qb_dropback", "epa", "success", "wp", "down", "half_seconds_remaining",
    "passer_player_id", "passer_player_name", "cpoe", "qb_epa",
    "id", "name", "qb_scramble", "sack",      # id/name: el QB del dropback (también en scrambles y sacks)
    "weather", "roof",                        # clima observado ("Light Snow Temp: 36° F…") y techo
]


def current_season(today: date = None) -> int:
    """La temporada NFL empieza en septiembre; enero y febrero son de la anterior."""
    today = today or date.today()
    return today.year if today.month >= 3 else today.year - 1


def _pbp_path(season: int) -> str:
    return os.path.join(CACHE_DIR, f"pbp_{season}.parquet")


def load_pbp(seasons: Iterable[int], refresh_current: bool = False) -> pl.DataFrame:
    """Jugadas de ataque (pase o carrera) de las temporadas pedidas, con las columnas que usa el modelo.
    Las temporadas cerradas se leen de la caché; la actual se vuelve a bajar si `refresh_current`."""
    import nflreadpy as nfl
    os.makedirs(CACHE_DIR, exist_ok=True)
    cur = current_season()
    frames: List[pl.DataFrame] = []
    for season in seasons:
        path = _pbp_path(season)
        if os.path.exists(path) and not (refresh_current and season == cur):
            cached = pl.read_parquet(path)
            if all(c in cached.columns for c in ("id", "name", "weather")):     # caché con las columnas de hoy
                frames.append(cached)
                continue
        logger.info(f"NFL: descargando jugadas de {season}")
        df = nfl.load_pbp([season])
        df = df.select([c for c in PBP_COLUMNS if c in df.columns])
        df = df.filter(pl.col("play_type").is_in(["pass", "run"]) & pl.col("epa").is_not_null()
                       & pl.col("posteam").is_not_null())
        df.write_parquet(path)
        frames.append(df)
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def load_schedules(refresh: bool = False) -> pl.DataFrame:
    """Calendario completo (todas las temporadas) con resultados, contexto y líneas de cierre."""
    import nflreadpy as nfl
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "schedules.parquet")
    if os.path.exists(path) and not refresh:
        return pl.read_parquet(path)
    df = nfl.load_schedules()
    df.write_parquet(path)
    return df


# Equipos que se mudaron: siempre con la abreviatura actual
RELOCATED = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


def normalize_teams(df: pl.DataFrame, cols=("posteam", "defteam", "home_team", "away_team")) -> pl.DataFrame:
    return df.with_columns([pl.col(c).replace(RELOCATED) for c in cols if c in df.columns])


def team_games(pbp: pl.DataFrame, wp_lo: float = 0.05, wp_hi: float = 0.95) -> pl.DataFrame:
    """Un renglón por equipo y partido: EPA por jugada de su ataque, en pase y en carrera.

    Se quitan las jugadas con el partido ya decidido (probabilidad de ganar fuera de 5–95%) y las
    de los últimos dos minutos de cada mitad, que distorsionan la eficiencia ("garbage time")."""
    pbp = normalize_teams(pbp)
    plays = pbp.filter(pl.col("wp").is_between(wp_lo, wp_hi) & (pl.col("half_seconds_remaining") > 120))
    plays = plays.with_columns(
        kind=pl.when(pl.col("qb_dropback") == 1).then(pl.lit("pass")).otherwise(pl.lit("rush")))
    agg = plays.group_by(["game_id", "season", "week", "game_date", "posteam", "defteam", "home_team", "kind"]).agg(
        epa=pl.col("epa").sum(), n=pl.len(), success=pl.col("success").mean())
    wide = agg.pivot(on="kind", index=["game_id", "season", "week", "game_date", "posteam", "defteam", "home_team"],
                     values=["epa", "n", "success"])
    vals = [c for c in wide.columns if c.split("_")[0] in ("epa", "n", "success")]
    wide = wide.with_columns([pl.col(c).fill_null(0.0) for c in vals]).with_columns(
        is_home=(pl.col("posteam") == pl.col("home_team")).cast(pl.Int8),
        n=pl.col("n_pass") + pl.col("n_rush"),
        epa=pl.col("epa_pass") + pl.col("epa_rush"),
    )
    return wide.sort(["game_date", "game_id", "posteam"])


RAIN_WORDS = ("rain", "shower", "drizzle", "thunder", "storm")
SNOW_WORDS = ("snow", "flurr", "sleet", "ice", "freezing")


def game_weather(pbp: pl.DataFrame) -> pl.DataFrame:
    """Un renglón por partido: si llovió o nevó (al aire libre), según el clima observado de nflverse."""
    g = pbp.group_by("game_id").agg(weather=pl.col("weather").drop_nulls().first(), roof=pl.col("roof").first())
    w = pl.col("weather").fill_null("").str.to_lowercase()
    outdoor = ~pl.col("roof").is_in(["dome", "closed"])
    # "Snow" va primero: "rain and snow" cuenta como nieve
    snow = w.str.contains("|".join(SNOW_WORDS)) & outdoor
    rain = w.str.contains("|".join(RAIN_WORDS)) & outdoor & ~snow
    return g.select("game_id", rain=rain.cast(pl.Float64), snow=snow.cast(pl.Float64))
