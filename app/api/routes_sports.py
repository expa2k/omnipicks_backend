"""
OmniPicks — Sports Data API Routes
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from datetime import date, datetime, timedelta
from typing import Optional

from app.database import get_db
from app.db_models import Game, Team, SportType, GameStatus

router = APIRouter()


@router.get("/games")
async def get_games(
    sport: Optional[str] = Query(None),
    game_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    status: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Get games with optional filters."""
    target_date = date.fromisoformat(game_date) if game_date else date.today()

    query = select(Game).where(Game.game_date == target_date)
    if sport:
        query = query.where(Game.sport == sport)
    if status:
        query = query.where(Game.status == status)

    query = query.order_by(Game.game_time)
    result = await db.execute(query)
    games = result.scalars().all()

    return {
        "date": target_date.isoformat(),
        "count": len(games),
        "games": [
            {
                "id": g.id,
                "external_id": g.external_id,
                "sport": g.sport.value if g.sport else None,
                "league": g.league,
                "home_team": g.home_team_name,
                "away_team": g.away_team_name,
                "home_team_id": g.home_team_id,
                "away_team_id": g.away_team_id,
                "game_time": g.game_time.isoformat() if g.game_time else None,
                "status": g.status.value if g.status else None,
                "venue": g.venue,
                "home_score": g.home_score,
                "away_score": g.away_score,
                "is_playoff": g.is_playoff,
            }
            for g in games
        ],
    }


@router.get("/games/{game_id}")
async def get_game_detail(
    game_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get detailed info for a specific game."""
    game = await db.execute(select(Game).where(Game.id == game_id))
    game = game.scalar_one_or_none()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    # Get teams
    home_team_result = await db.execute(select(Team).where(Team.id == game.home_team_id))
    home_team = home_team_result.scalar_one_or_none()

    away_team_result = await db.execute(select(Team).where(Team.id == game.away_team_id))
    away_team = away_team_result.scalar_one_or_none()

    return {
        "game": {
            "id": game.id,
            "sport": game.sport.value if game.sport else None,
            "league": game.league,
            "game_date": game.game_date.isoformat(),
            "game_time": game.game_time.isoformat() if game.game_time else None,
            "status": game.status.value if game.status else None,
            "venue": game.venue,
            "home_score": game.home_score,
            "away_score": game.away_score,
            "is_playoff": game.is_playoff,
        },
        "home_team": {
            "id": home_team.id if home_team else game.home_team_id,
            "name": home_team.name if home_team else game.home_team_name,
            "short_name": home_team.short_name if home_team else None,
            "logo_url": home_team.logo_url if home_team else None,
            "elo_rating": home_team.elo_rating if home_team else 1500,
            "record": f"{home_team.wins}-{home_team.losses}" if home_team else "0-0",
            "offensive_rating": home_team.offensive_rating if home_team else None,
            "defensive_rating": home_team.defensive_rating if home_team else None,
        },
        "away_team": {
            "id": away_team.id if away_team else game.away_team_id,
            "name": away_team.name if away_team else game.away_team_name,
            "short_name": away_team.short_name if away_team else None,
            "logo_url": away_team.logo_url if away_team else None,
            "elo_rating": away_team.elo_rating if away_team else 1500,
            "record": f"{away_team.wins}-{away_team.losses}" if away_team else "0-0",
            "offensive_rating": away_team.offensive_rating if away_team else None,
            "defensive_rating": away_team.defensive_rating if away_team else None,
        },
    }


@router.get("/teams")
async def get_teams(
    sport: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Get teams with optional sport filter and search."""
    query = select(Team)
    if sport:
        query = query.where(Team.sport == sport)
    if search:
        query = query.where(Team.name.ilike(f"%{search}%"))

    query = query.order_by(Team.elo_rating.desc())
    result = await db.execute(query)
    teams = result.scalars().all()

    return {
        "count": len(teams),
        "teams": [
            {
                "id": t.id,
                "external_id": t.external_id,
                "sport": t.sport.value if t.sport else None,
                "name": t.name,
                "short_name": t.short_name,
                "logo_url": t.logo_url,
                "league": t.league,
                "elo_rating": round(t.elo_rating, 1),
                "record": f"{t.wins}-{t.losses}" + (f"-{t.draws}" if t.draws else ""),
                "offensive_rating": t.offensive_rating,
                "defensive_rating": t.defensive_rating,
            }
            for t in teams
        ],
    }


@router.get("/schedule")
async def get_schedule(
    sport: Optional[str] = Query(None),
    days: int = Query(default=7, ge=1, le=30),
    db: AsyncSession = Depends(get_db),
):
    """Get upcoming schedule for the next N days."""
    today = date.today()
    end_date = today + timedelta(days=days)

    query = select(Game).where(
        and_(
            Game.game_date >= today,
            Game.game_date <= end_date,
        )
    )
    if sport:
        query = query.where(Game.sport == sport)

    query = query.order_by(Game.game_date, Game.game_time)
    result = await db.execute(query)
    games = result.scalars().all()

    # Group by date
    by_date = {}
    for g in games:
        d = g.game_date.isoformat()
        if d not in by_date:
            by_date[d] = []
        by_date[d].append({
            "id": g.id,
            "sport": g.sport.value if g.sport else None,
            "league": g.league,
            "home_team": g.home_team_name,
            "away_team": g.away_team_name,
            "game_time": g.game_time.isoformat() if g.game_time else None,
            "status": g.status.value if g.status else None,
        })

    return {
        "from": today.isoformat(),
        "to": end_date.isoformat(),
        "total_games": len(games),
        "schedule": by_date,
    }
