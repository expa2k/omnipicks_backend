"""
OmniPicks — Odds API Routes
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from datetime import date, datetime
from typing import Optional

from app.database import get_db
from app.db_models import OddsSnapshot, Game, SportType

router = APIRouter()


@router.get("/today")
async def get_today_odds(
    sport: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Get latest odds for today's games."""
    today = date.today()

    # Get today's games
    games_query = select(Game).where(Game.game_date == today)
    if sport:
        games_query = games_query.where(Game.sport == sport)
    games_result = await db.execute(games_query)
    games = games_result.scalars().all()
    game_ids = [g.id for g in games]
    games_map = {g.id: g for g in games}

    if not game_ids:
        return {"date": today.isoformat(), "count": 0, "games": []}

    # Get latest odds for each game
    odds_result = await db.execute(
        select(OddsSnapshot).where(
            OddsSnapshot.game_id.in_(game_ids)
        ).order_by(OddsSnapshot.snapshot_time.desc())
    )
    odds_list = odds_result.scalars().all()

    # Group odds by game
    odds_by_game = {}
    for o in odds_list:
        if o.game_id not in odds_by_game:
            odds_by_game[o.game_id] = []
        odds_by_game[o.game_id].append({
            "sportsbook": o.sportsbook,
            "market": o.market.value if o.market else None,
            "home_odds": o.home_odds,
            "away_odds": o.away_odds,
            "draw_odds": o.draw_odds,
            "spread_home": o.spread_home,
            "spread_home_odds": o.spread_home_odds,
            "spread_away_odds": o.spread_away_odds,
            "total_line": o.total_line,
            "over_odds": o.over_odds,
            "under_odds": o.under_odds,
            "implied_home_prob": round(o.implied_home_prob, 4) if o.implied_home_prob else None,
            "implied_away_prob": round(o.implied_away_prob, 4) if o.implied_away_prob else None,
            "snapshot_time": o.snapshot_time.isoformat() if o.snapshot_time else None,
        })

    return {
        "date": today.isoformat(),
        "count": len(games),
        "games": [
            {
                "game_id": g.id,
                "sport": g.sport.value if g.sport else None,
                "home_team": g.home_team_name,
                "away_team": g.away_team_name,
                "game_time": g.game_time.isoformat() if g.game_time else None,
                "odds": odds_by_game.get(g.id, []),
            }
            for g in games
        ],
    }


@router.get("/compare/{game_id}")
async def compare_odds(
    game_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Compare odds across all sportsbooks for a specific game."""
    # Get game info
    game = await db.execute(select(Game).where(Game.id == game_id))
    game = game.scalar_one_or_none()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    # Get all odds snapshots for this game
    odds_result = await db.execute(
        select(OddsSnapshot).where(
            OddsSnapshot.game_id == game_id
        ).order_by(OddsSnapshot.snapshot_time.desc())
    )
    odds_list = odds_result.scalars().all()

    # Group by sportsbook, keep only latest
    latest_by_book = {}
    for o in odds_list:
        key = f"{o.sportsbook}_{o.market.value if o.market else ''}"
        if key not in latest_by_book:
            latest_by_book[key] = o

    # Find best odds
    best_home_ml = max(
        (o for o in latest_by_book.values() if o.home_odds and o.market and o.market.value == "moneyline"),
        key=lambda x: x.home_odds,
        default=None,
    )
    best_away_ml = max(
        (o for o in latest_by_book.values() if o.away_odds and o.market and o.market.value == "moneyline"),
        key=lambda x: x.away_odds,
        default=None,
    )

    return {
        "game": {
            "id": game.id,
            "sport": game.sport.value if game.sport else None,
            "home_team": game.home_team_name,
            "away_team": game.away_team_name,
            "game_time": game.game_time.isoformat() if game.game_time else None,
        },
        "best_odds": {
            "home_moneyline": {
                "sportsbook": best_home_ml.sportsbook if best_home_ml else None,
                "odds": best_home_ml.home_odds if best_home_ml else None,
            },
            "away_moneyline": {
                "sportsbook": best_away_ml.sportsbook if best_away_ml else None,
                "odds": best_away_ml.away_odds if best_away_ml else None,
            },
        },
        "all_odds": [
            {
                "sportsbook": o.sportsbook,
                "market": o.market.value if o.market else None,
                "home_odds": o.home_odds,
                "away_odds": o.away_odds,
                "draw_odds": o.draw_odds,
                "spread_home": o.spread_home,
                "total_line": o.total_line,
                "over_odds": o.over_odds,
                "under_odds": o.under_odds,
                "snapshot_time": o.snapshot_time.isoformat() if o.snapshot_time else None,
            }
            for o in latest_by_book.values()
        ],
    }


@router.get("/sports")
async def get_available_sports():
    """Get list of supported sports and their odds API keys."""
    return {
        "sports": [
            {
                "key": "nba",
                "name": "NBA",
                "icon": "🏀",
                "odds_key": "basketball_nba",
                "active": True,
            },
            {
                "key": "nfl",
                "name": "NFL",
                "icon": "🏈",
                "odds_key": "americanfootball_nfl",
                "active": True,
            },
            {
                "key": "mlb",
                "name": "MLB",
                "icon": "⚾",
                "odds_key": "baseball_mlb",
                "active": True,
            },
            {
                "key": "soccer",
                "name": "Soccer",
                "icon": "⚽",
                "odds_key": "soccer_epl",
                "active": True,
                "leagues": [
                    {"key": "soccer_mexico_ligamx", "name": "Liga MX 🇲🇽"},
                    {"key": "soccer_epl", "name": "Premier League 🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
                    {"key": "soccer_spain_la_liga", "name": "La Liga 🇪🇸"},
                    {"key": "soccer_italy_serie_a", "name": "Serie A 🇮🇹"},
                    {"key": "soccer_germany_bundesliga", "name": "Bundesliga 🇩🇪"},
                    {"key": "soccer_france_ligue_one", "name": "Ligue 1 🇫🇷"},
                    {"key": "soccer_uefa_champs_league", "name": "Champions League 🏆"},
                    {"key": "soccer_uefa_europa_league", "name": "Europa League 🏆"},
                ],
            },
        ]
    }
