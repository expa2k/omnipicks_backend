"""
OmniPicks — MLB v2, fase 0: histórico de partidos (Retrosheet) y momios de cierre (SBR).

Retrosheet "game logs": un renglón por partido con marcador, parque, día/noche, estadísticas de
equipo, los dos **abridores** y las dos **alineaciones** (9 bateadores con su posición).
SBR: momios de apertura y cierre (moneyline, run line y total) de 2010 a 2021.

Caché en ml/datasets/mlb/. Uso de los datos de Retrosheet: "The information used here was obtained
free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at
www.retrosheet.org."
"""
import io
import logging
import os
import time
import zipfile
from typing import Iterable, List

import httpx
import polars as pl

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ml", "datasets", "mlb"))
UA = {"User-Agent": "Mozilla/5.0 (OmniPicks; uso personal)"}

# Campos del game log de Retrosheet (índices desde 0)
GL = {"date": 0, "game_num": 1, "away": 3, "home": 6, "away_score": 9, "home_score": 10, "outs": 11,
      "day_night": 12, "park": 16, "away_sp_id": 101, "away_sp": 102, "home_sp_id": 103, "home_sp": 104,
      "away_h": 22, "away_hr": 25, "away_bb": 30, "away_k": 32, "home_h": 50, "home_hr": 53, "home_bb": 58, "home_k": 60}
AWAY_LINEUP0, HOME_LINEUP0 = 105, 132          # 9 × (id, nombre, posición)


def _get(url: str, tries: int = 3):
    for attempt in range(tries):
        try:
            r = httpx.get(url, headers=UA, timeout=60, follow_redirects=True)
            if r.status_code == 200:
                return r.content
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    logger.warning(f"MLB: no se pudo bajar {url}")
    return None


def load_gamelogs(seasons: Iterable[int]) -> pl.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    frames: List[pl.DataFrame] = []
    for y in seasons:
        path = os.path.join(CACHE_DIR, f"gl{y}.parquet")
        if os.path.exists(path):
            frames.append(pl.read_parquet(path))
            continue
        raw = _get(f"https://www.retrosheet.org/gamelogs/gl{y}.zip")
        if raw is None:
            continue
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            txt = z.read(z.namelist()[0]).decode("latin-1")
        rows = []
        for line in txt.splitlines():
            f = [x.strip('"') for x in line.split(",")]
            if len(f) < 160:
                continue
            r = {k: f[i] for k, i in GL.items()}
            r["away_lineup"] = ";".join(f[AWAY_LINEUP0 + 3 * k] for k in range(9))
            r["away_lineup_pos"] = ";".join(f[AWAY_LINEUP0 + 3 * k + 2] for k in range(9))
            r["home_lineup"] = ";".join(f[HOME_LINEUP0 + 3 * k] for k in range(9))
            r["home_lineup_pos"] = ";".join(f[HOME_LINEUP0 + 3 * k + 2] for k in range(9))
            rows.append(r)
        df = pl.DataFrame(rows).with_columns(
            pl.col("date").str.strptime(pl.Date, "%Y%m%d"), pl.lit(y).alias("season"),
            *[pl.col(c).cast(pl.Float64, strict=False) for c in ("away_score", "home_score", "outs", "away_h", "away_hr",
                                                                   "away_bb", "away_k", "home_h", "home_hr", "home_bb", "home_k")])
        df.write_parquet(path)
        frames.append(df)
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def download_odds(seasons: Iterable[int]) -> List[str]:
    """Baja los Excel de momios de SBR (2010–2021) a la caché y devuelve las rutas."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    out = []
    for y in seasons:
        path = os.path.join(CACHE_DIR, f"odds{y}.xlsx")
        if not os.path.exists(path):
            raw = _get(f"https://www.sportsbookreviewsonline.com/wp-content/uploads/sportsbookreviewsonline_com_737/mlb-odds-{y}.xlsx")
            if raw is None:
                continue
            open(path, "wb").write(raw)
        out.append(path)
    return out
