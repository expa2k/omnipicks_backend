"""
OmniPicks — NFL: contexto del partido (descanso, viaje, husos horarios, división, localía).

Todo sale del calendario de nflverse (en producción, del calendario de ESPN), así que se sabe
antes del partido. Cada variable va del lado del local: positiva = favorece al local.
"""
import math
from typing import Dict

# Estadio de cada equipo: latitud, longitud y huso horario estándar (UTC)
TEAM_SITE = {
    "ARI": (33.53, -112.26, -7), "ATL": (33.76, -84.40, -5), "BAL": (39.28, -76.62, -5),
    "BUF": (42.77, -78.79, -5), "CAR": (35.23, -80.85, -5), "CHI": (41.86, -87.62, -6),
    "CIN": (39.10, -84.52, -5), "CLE": (41.51, -81.70, -5), "DAL": (32.75, -97.09, -6),
    "DEN": (39.74, -105.02, -7), "DET": (42.34, -83.05, -5), "GB": (44.50, -88.06, -6),
    "HOU": (29.68, -95.41, -6), "IND": (39.76, -86.16, -5), "JAX": (30.32, -81.64, -5),
    "KC": (39.05, -94.48, -6), "LA": (33.95, -118.34, -8), "LAC": (33.95, -118.34, -8),
    "LV": (36.09, -115.18, -8), "MIA": (25.96, -80.24, -5), "MIN": (44.97, -93.26, -6),
    "NE": (42.09, -71.26, -5), "NO": (29.95, -90.08, -6), "NYG": (40.81, -74.07, -5),
    "NYJ": (40.81, -74.07, -5), "PHI": (39.90, -75.17, -5), "PIT": (40.45, -80.02, -5),
    "SEA": (47.60, -122.33, -8), "SF": (37.40, -121.97, -8), "TB": (27.98, -82.50, -5),
    "TEN": (36.17, -86.77, -6), "WAS": (38.91, -76.86, -5),
}
# Antes de mudarse (las abreviaturas ya vienen normalizadas a las actuales)
FORMER_SITE = {
    "LV": (2020, (37.75, -122.20, -8)),    # Oakland hasta 2019
    "LAC": (2017, (32.78, -117.12, -8)),   # San Diego hasta 2016
    "LA": (2016, (38.63, -90.19, -6)),     # St. Louis hasta 2015
}

CONTEXT_FEATURES = ["home_field", "rest_diff", "home_bye", "away_bye", "home_short", "away_short",
                    "travel_k", "tz_diff", "away_early", "div_home", "no_fans", "playoff_home"]


def site(team: str, season: int):
    if team in FORMER_SITE and season < FORMER_SITE[team][0]:
        return FORMER_SITE[team][1]
    return TEAM_SITE.get(team, (39.0, -95.0, -6))


def km(a, b) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def context(home: str, away: str, season: int, *, neutral: bool, home_rest, away_rest, gametime_et: str,
            div_game: bool, playoff: bool) -> Dict[str, float]:
    """Variables de contexto de un partido. gametime_et: hora de inicio en hora del Este ("13:00")."""
    hs, as_ = site(home, season), site(away, season)
    hr = float(home_rest if home_rest is not None else 7)
    ar = float(away_rest if away_rest is not None else 7)
    try:
        hh, mm = (int(x) for x in (gametime_et or "13:00").split(":"))
        kick_et = hh + mm / 60.0
    except ValueError:
        kick_et = 13.0
    away_local = kick_et + (as_[2] + 5)            # hora del reloj del visitante
    home_field = 0.0 if neutral else 1.0
    return {
        "home_field": home_field,
        "rest_diff": max(-7.0, min(7.0, hr - ar)),
        "home_bye": 1.0 if hr >= 13 else 0.0,
        "away_bye": 1.0 if ar >= 13 else 0.0,
        "home_short": 1.0 if hr <= 5 else 0.0,
        "away_short": 1.0 if ar <= 5 else 0.0,
        "travel_k": 0.0 if neutral else km(hs, as_) / 1000.0,
        "tz_diff": 0.0 if neutral else float(hs[2] - as_[2]),    # >0: el visitante viajó hacia el este
        "away_early": 1.0 if (not neutral and away_local < 11.0) else 0.0,
        "div_home": home_field if div_game else 0.0,
        "no_fans": home_field if season == 2020 else 0.0,
        "playoff_home": home_field if playoff else 0.0,
    }
