"""
OmniPicks — Utilities: Constants
"""
from app.db_models import SportType

# Map SportType to API-Sports endpoints
API_SPORTS_ENDPOINTS = {
    SportType.NBA: "https://v2.nba.api-sports.io",
    SportType.NFL: "https://v1.american-football.api-sports.io",
    SportType.MLB: "https://v1.baseball.api-sports.io",
    SportType.SOCCER: "https://v3.football.api-sports.io",
}

# The Odds API sport keys
ODDS_API_KEYS = {
    SportType.NBA: ["basketball_nba"],
    SportType.NFL: ["americanfootball_nfl"],
    SportType.MLB: ["baseball_mlb"],
    SportType.SOCCER: [
        "soccer_mexico_ligamx",
        "soccer_epl",
        "soccer_spain_la_liga",
        "soccer_italy_serie_a",
        "soccer_germany_bundesliga",
        "soccer_france_ligue_one",
        "soccer_uefa_champs_league",
        "soccer_uefa_europa_league",
        "soccer_fifa_world_cup"
    ]
}

# Bzzoiro supported leagues
BZZOIRO_LEAGUES = [
    "Liga MX", "Premier League", "La Liga", "Serie A", 
    "Bundesliga", "Ligue 1", "UEFA Champions League", "UEFA Europa League", "World Cup"
]

# Market types mapping from Odds API to our internal ENUM
ODDS_MARKETS_MAP = {
    "h2h": "moneyline",
    "spreads": "spread",
    "totals": "over_under",
    "btts": "btts"
}
