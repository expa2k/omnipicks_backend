"""
OmniPicks — Fútbol v2, fase 0: histórico de partidos (football-data.co.uk, gratis).

Cada CSV trae por partido: resultado y medio tiempo, tiros, tiros a puerta, córners, tarjetas y
momios de varias casas (apertura y cierre) para 1X2, más/menos 2.5 y hándicap asiático. Las
temporadas recientes de las ligas grandes traen además xG (HxG, AxG).

Se guarda en caché en ml/datasets/soccer/ (las temporadas cerradas no cambian; la actual se
vuelve a bajar si se pide).
"""
import io
import logging
import os
import time
from datetime import date
from typing import Dict, Iterable, List, Optional

import httpx
import polars as pl

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ml", "datasets", "soccer"))
BASE = "https://football-data.co.uk"
UA = {"User-Agent": "Mozilla/5.0 (OmniPicks; uso personal)"}

# Ligas "principales" (un CSV por temporada) → nombre que usa la app
MAIN_LEAGUES = {
    "E0": "Premier League", "E1": "Championship", "SP1": "La Liga", "I1": "Serie A", "D1": "Bundesliga",
    "F1": "Ligue 1", "N1": "Eredivisie", "P1": "Primeira Liga",
}
# Ligas "nuevas" (un CSV con todas las temporadas)
EXTRA_LEAGUES = {"MEX": "Liga MX", "BRA": "Brasileirão Serie A", "USA": "MLS", "ARG": "Liga Profesional"}

KEEP = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HxG", "AxG",
        "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR",
        "AvgH", "AvgD", "AvgA", "AvgCH", "AvgCD", "AvgCA", "PSCH", "PSCD", "PSCA", "B365CH", "B365CD", "B365CA",
        "Avg>2.5", "Avg<2.5", "AvgC>2.5", "AvgC<2.5", "AHh", "AHCh", "AvgCAHH", "AvgCAHA"]
EXTRA_MAP = {"Country": "Country", "League": "League", "Season": "Season", "Date": "Date", "Time": "Time",
             "Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG", "Res": "FTR",
             "AvgCH": "AvgCH", "AvgCD": "AvgCD", "AvgCA": "AvgCA", "PSCH": "PSCH", "PSCD": "PSCD", "PSCA": "PSCA",
             "B365CH": "B365CH", "B365CD": "B365CD", "B365CA": "B365CA"}


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season_start(today: Optional[date] = None) -> int:
    """Las ligas europeas empiezan en agosto: de julio en adelante cuenta la temporada nueva."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def _get_csv(url: str, tries: int = 3) -> Optional[pl.DataFrame]:
    for attempt in range(tries):
        try:
            r = httpx.get(url, headers=UA, timeout=30, follow_redirects=True)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            txt = r.content.decode("utf-8-sig", errors="replace")
            return pl.read_csv(io.StringIO(txt), infer_schema_length=0, truncate_ragged_lines=True,
                               ignore_errors=True)
        except Exception as e:
            if attempt == tries - 1:
                logger.warning(f"football-data: {url} no respondió ({type(e).__name__}: {e})")
            time.sleep(1.5 * (attempt + 1))
    return None


def _clean(df: pl.DataFrame, league: str, season: int) -> pl.DataFrame:
    cols = [c for c in KEEP if c in df.columns]
    df = df.select(cols).filter(pl.col("HomeTeam").is_not_null() & (pl.col("HomeTeam") != ""))
    num = [c for c in cols if c not in ("Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTR")]
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in num])
    df = df.with_columns(
        date=pl.coalesce(pl.col("Date").str.strptime(pl.Date, "%d/%m/%Y", strict=False),
                         pl.col("Date").str.strptime(pl.Date, "%d/%m/%y", strict=False)),
        league=pl.lit(league), season=pl.lit(season, dtype=pl.Int32))
    return df


def load_main(div: str, seasons: Iterable[int], refresh_current: bool = False) -> pl.DataFrame:
    """Una liga principal (E0, SP1…) por temporada (año de inicio)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cur = current_season_start()
    frames: List[pl.DataFrame] = []
    for s in seasons:
        path = os.path.join(CACHE_DIR, f"{div}_{season_code(s)}.parquet")
        if os.path.exists(path) and not (refresh_current and s == cur):
            frames.append(pl.read_parquet(path))
            continue
        raw = _get_csv(f"{BASE}/mmz4281/{season_code(s)}/{div}.csv")
        if raw is None or raw.height == 0:
            continue
        df = _clean(raw, MAIN_LEAGUES.get(div, div), s)
        df.write_parquet(path)
        frames.append(df)
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def load_extra(code: str, refresh: bool = False) -> pl.DataFrame:
    """Liga 'nueva' (MEX, BRA…): un archivo con todas las temporadas. La temporada se toma del año de
    inicio ("2025/2026" → 2025; "2025" → 2025)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{code}.parquet")
    if os.path.exists(path) and not refresh and os.path.getmtime(path) > time.time() - 12 * 3600:
        return pl.read_parquet(path)
    raw = _get_csv(f"{BASE}/new/{code}.csv")
    if raw is None:
        return pl.read_parquet(path) if os.path.exists(path) else pl.DataFrame()
    raw = raw.rename({k: v for k, v in EXTRA_MAP.items() if k in raw.columns and k != v})
    season = pl.col("Season").cast(pl.Utf8).str.slice(0, 4).cast(pl.Int32, strict=False) if "Season" in raw.columns else None
    df = _clean(raw, EXTRA_LEAGUES.get(code, code), 0)
    if season is not None:
        df = df.with_columns(season=raw.filter(pl.col("HomeTeam").is_not_null() & (pl.col("HomeTeam") != ""))
                             .select(season)["Season"])
    df.write_parquet(path)
    return df


def load_all(first_season: int = 2010, refresh_current: bool = False) -> pl.DataFrame:
    """Todas las ligas, de `first_season` a la actual."""
    cur = current_season_start()
    frames = [load_main(d, range(first_season, cur + 1), refresh_current) for d in MAIN_LEAGUES]
    for code in EXTRA_LEAGUES:
        e = load_extra(code, refresh=refresh_current)
        if e.height:
            frames.append(e.filter(pl.col("season") >= first_season))
    return pl.concat([f for f in frames if f.height], how="diagonal_relaxed")
