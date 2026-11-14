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
from typing import Dict, Iterable, List

import httpx
import polars as pl

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ml", "datasets", "mlb"))
UA = {"User-Agent": "Mozilla/5.0 (OmniPicks; uso personal)"}

# Campos del game log de Retrosheet (índices desde 0)
GL = {"date": 0, "game_num": 1, "away": 3, "home": 6, "away_score": 9, "home_score": 10, "outs": 11,
      "day_night": 12, "park": 16, "away_sp_id": 101, "away_sp": 102, "home_sp_id": 103, "home_sp": 104,
      "away_h": 22, "away_hr": 25, "away_bb": 30, "away_k": 32, "home_h": 50, "home_hr": 53, "home_bb": 58, "home_k": 60,
      "away_pitchers": 38, "home_pitchers": 66}
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
            cached = pl.read_parquet(path)
            if "home_pitchers" in cached.columns:
                frames.append(cached)
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
                                                                   "away_bb", "away_k", "home_h", "home_hr", "home_bb", "home_k",
                                                                   "away_pitchers", "home_pitchers")])
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


def load_people() -> pl.DataFrame:
    """Jugadores (Lahman / Chadwick): id de Retrosheet, id de MLBAM, nombre, con qué mano batea y lanza."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "people.parquet")
    if os.path.exists(path) and os.path.getmtime(path) > time.time() - 30 * 86400:
        return pl.read_parquet(path)
    lah = _get("https://raw.githubusercontent.com/chadwickbureau/baseballdatabank/master/core/People.csv")
    frames = []
    if lah:
        p = pl.read_csv(io.BytesIO(lah), infer_schema_length=0).select(
            pl.col("retroID").alias("retro_id"), "nameFirst", "nameLast", "bats", "throws")
        frames.append(p)
    reg = []
    for c in "0123456789abcdef":
        raw = _get(f"https://raw.githubusercontent.com/chadwickbureau/register/master/data/people-{c}.csv")
        if raw:
            reg.append(pl.read_csv(io.BytesIO(raw), infer_schema_length=0).select(
                pl.col("key_retro").alias("retro_id"), pl.col("key_mlbam").alias("mlbam_id"),
                pl.col("name_first").alias("first"), pl.col("name_last").alias("last")))
    r = pl.concat(reg).filter(pl.col("retro_id").is_not_null() & (pl.col("retro_id") != "")) if reg else pl.DataFrame()
    if frames and r.height:
        out = r.join(frames[0].filter(pl.col("retro_id").is_not_null()), on="retro_id", how="left")
    else:
        out = r
    out.write_parquet(path)
    return out


def load_handedness(seasons: Iterable[int]) -> Dict[str, Dict[str, str]]:
    """id de Retrosheet → {"bats": L/R/S, "throws": L/R}, con la lista de jugadores de cada temporada de la
    MLB StatsAPI (una consulta por temporada) y el registro de Chadwick para pasar de MLBAM a Retrosheet."""
    import json
    path = os.path.join(CACHE_DIR, "handedness.json")
    cache = json.load(open(path)) if os.path.exists(path) else {}
    seasons = [s for s in seasons if str(s) not in cache.get("_seasons", [])]
    if seasons:
        for y in seasons:
            raw = _get(f"https://statsapi.mlb.com/api/v1/sports/1/players?season={y}")
            if not raw:
                continue
            for p in json.loads(raw).get("people", []):
                cache.setdefault("mlbam", {})[str(p["id"])] = {"bats": (p.get("batSide") or {}).get("code"),
                                                              "throws": (p.get("pitchHand") or {}).get("code")}
            cache.setdefault("_seasons", []).append(str(y))
        json.dump(cache, open(path, "w"))
    reg = load_people()
    m2r = dict(zip(reg["mlbam_id"].to_list(), reg["retro_id"].to_list()))
    return {m2r[k]: v for k, v in cache.get("mlbam", {}).items() if k in m2r}


SBR_TO_RETRO = {"CUB": "CHN", "CWS": "CHA", "CHW": "CHA", "KAN": "KCA", "KC": "KCA", "LAA": "ANA", "LAD": "LAN",
                "NYM": "NYN", "NYY": "NYA", "SDG": "SDN", "SD": "SDN", "SFO": "SFN", "SF": "SFN", "STL": "SLN",
                "TAM": "TBA", "TB": "TBA", "WAS": "WAS", "WSH": "WAS", "FLA": "MIA", "MIA": "MIA", "ARI": "ARI",
                "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET",
                "HOU": "HOU", "MIL": "MIL", "MIN": "MIN", "OAK": "OAK", "PHI": "PHI", "PIT": "PIT", "SEA": "SEA",
                "TEX": "TEX", "TOR": "TOR", "LOS": "LAN", "ANA": "ANA", "BRS": "BOS", "CHC": "CHN", "SFG": "SFN"}


def load_odds(seasons: Iterable[int]) -> pl.DataFrame:
    """Momios de cierre de SBR por juego (renglones visitante/local), con los códigos de Retrosheet."""
    import pandas as pd
    out = []
    for y, path in zip(seasons, download_odds(seasons)):
        try:
            df = pd.read_excel(path)
        except Exception as e:
            logger.warning(f"MLB: momios {y} ilegibles ({type(e).__name__})")
            continue
        df.columns = [str(c).replace(" ", "") for c in df.columns]      # "Run Line" / "RunLine" según el año
        df = df.rename(columns={"Unnamed:18": "RunLineOdds", "Unnamed:20": "OpenOUOdds", "Unnamed:22": "CloseOUOdds"})
        rows = df.to_dict("records")
        for v, h in zip(rows[0::2], rows[1::2]):
            if not ((v.get("VH") == "V" and h.get("VH") == "H") or (v.get("VH") == "N" and h.get("VH") == "N")):
                continue                     # "N" = cancha neutral: el segundo renglón juega de local
            try:
                md = int(v["Date"])
                d = pd.Timestamp(year=y, month=md // 100, day=md % 100).date()
                num = lambda x: float(str(x).replace("½", ".5")) if str(x).strip() not in ("", "nan", "NL", "x", "pk") else None
                out.append({"date": d, "away": SBR_TO_RETRO.get(str(v["Team"]).strip(), str(v["Team"]).strip()),
                            "home": SBR_TO_RETRO.get(str(h["Team"]).strip(), str(h["Team"]).strip()),
                            "ml_away": num(v["Close"]), "ml_home": num(h["Close"]),
                            "ml_away_open": num(v["Open"]), "ml_home_open": num(h["Open"]),
                            "rl_home": num(h["RunLine"]), "rl_home_odds": num(h["RunLineOdds"]),
                            "rl_away_odds": num(v["RunLineOdds"]), "total": num(v["CloseOU"]),
                            "over_odds": num(v["CloseOUOdds"]), "under_odds": num(h["CloseOUOdds"]),
                            "final_away": num(v["Final"]), "final_home": num(h["Final"])})
            except (ValueError, KeyError, TypeError):
                continue
    return pl.DataFrame(out)
