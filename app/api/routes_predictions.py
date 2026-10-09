"""
OmniPicks — Predictions API Routes
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from datetime import date, datetime, timedelta
from typing import Optional, List

from app.database import get_db
from app.db_models import (
    Prediction, Pick, Game, OddsSnapshot, SportType, PickResult, GameStatus, MarketType
)

router = APIRouter()


@router.get("/today")
async def get_today_predictions(
    sport: Optional[str] = Query(None, description="Filter by sport: nba, nfl, mlb, soccer"),
    db: AsyncSession = Depends(get_db),
):
    """Get all predictions for today's games."""
    today = date.today()
    query = select(Prediction).where(
        func.date(Prediction.created_at) == today
    )
    if sport:
        query = query.where(Prediction.sport == sport)
    query = query.order_by(Prediction.confidence.desc())

    result = await db.execute(query)
    predictions = result.scalars().all()

    return {
        "date": today.isoformat(),
        "count": len(predictions),
        "predictions": [
            {
                "id": p.id,
                "game_id": p.game_id,
                "sport": p.sport.value if p.sport else None,
                "model_name": p.model_name,
                "home_win_prob": round(p.home_win_prob, 4),
                "away_win_prob": round(p.away_win_prob, 4),
                "draw_prob": round(p.draw_prob, 4) if p.draw_prob else None,
                "predicted_home_score": p.predicted_home_score,
                "predicted_away_score": p.predicted_away_score,
                "predicted_total": p.predicted_total,
                "predicted_spread": p.predicted_spread,
                "confidence": round(p.confidence, 4) if p.confidence else None,
                "feature_importances": p.feature_importances,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in predictions
        ],
    }


@router.get("/games")
async def get_game_predictions(
    sport: Optional[str] = Query(None, description="Filter by sport: nba, nfl, mlb, soccer"),
    game_date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
    db: AsyncSession = Depends(get_db),
):
    """
    CARTELERA COMPLETA: Returns ALL games for the day with predictions, odds, and value picks.
    The mathematical model is the protagonist. Casino odds are secondary.
    """
    if game_date:
        target_date = date.fromisoformat(game_date)
    else:
        # Default to today in US Eastern Time (GMT-4)
        et_now = datetime.utcnow() - timedelta(hours=4)
        target_date = et_now.date()

    # 1. Get ALL games for the date using exact ET day boundaries in UTC
    start_utc = datetime(target_date.year, target_date.month, target_date.day) + timedelta(hours=4)
    end_utc = start_utc + timedelta(days=1)
    
    games_query = select(Game).where(
        and_(
            Game.game_time >= start_utc,
            Game.game_time <= end_utc,
            Game.status == GameStatus.SCHEDULED,
            Game.game_time > datetime.utcnow(),
        )
    )
    if sport:
        games_query = games_query.where(Game.sport == sport)
    # For MLB, only show actual MLB league games (not minor leagues like PCL, IL, LMB)
    if sport and sport.upper() == "MLB":
        games_query = games_query.where(Game.league == "MLB")
    games_query = games_query.order_by(Game.game_time)

    games_result = await db.execute(games_query)
    games = games_result.scalars().all()

    if not games:
        return {
            "date": target_date.isoformat(),
            "sport": sport.upper() if sport else "ALL",
            "count": 0,
            "games": [],
        }

    game_ids = [g.id for g in games]

    # 2. Get ALL predictions for these games (batch query)
    preds_result = await db.execute(
        select(Prediction).where(Prediction.game_id.in_(game_ids))
        .order_by(Prediction.created_at.desc())
    )
    all_preds = preds_result.scalars().all()
    # Keep latest prediction per game
    preds_map = {}
    for p in all_preds:
        if p.game_id not in preds_map:
            preds_map[p.game_id] = p

    # 3. Get ALL odds for these games (batch query, latest per sportsbook)
    odds_result = await db.execute(
        select(OddsSnapshot).where(OddsSnapshot.game_id.in_(game_ids))
        .order_by(OddsSnapshot.snapshot_time.desc())
    )
    all_odds = odds_result.scalars().all()
    odds_map = {}
    for o in all_odds:
        if o.game_id not in odds_map:
            odds_map[o.game_id] = []
        # Only keep latest per sportsbook per market
        existing = [(x.sportsbook, x.market) for x in odds_map[o.game_id]]
        if (o.sportsbook, o.market) not in existing:
            odds_map[o.game_id].append(o)

    # 4. Get ALL value picks for these games (batch query)
    picks_result = await db.execute(
        select(Pick).where(Pick.game_id.in_(game_ids))
        .order_by(Pick.expected_value.desc())
    )
    all_picks = picks_result.scalars().all()
    picks_map = {}
    for pk in all_picks:
        if pk.game_id not in picks_map:
            picks_map[pk.game_id] = []
        picks_map[pk.game_id].append(pk)

    # 5. Build response
    response_games = []
    for g in games:
        pred = preds_map.get(g.id)
        odds_list = odds_map.get(g.id, [])
        game_picks = picks_map.get(g.id, [])

        # Build odds summary (best moneyline odds across sportsbooks)
        odds_summary = None
        if odds_list:
            ml_odds = [o for o in odds_list if o.market == MarketType.MONEYLINE]
            if ml_odds:
                best_home = min(ml_odds, key=lambda o: abs(o.home_odds or 9999))
                best_away = min(ml_odds, key=lambda o: abs(o.away_odds or 9999))
                odds_summary = {
                    "best_home_ml": {
                        "odds": best_home.home_odds,
                        "sportsbook": best_home.sportsbook,
                    } if best_home.home_odds else None,
                    "best_away_ml": {
                        "odds": best_away.away_odds,
                        "sportsbook": best_away.sportsbook,
                    } if best_away.away_odds else None,
                    "best_draw": {
                        "odds": ml_odds[0].draw_odds,
                        "sportsbook": ml_odds[0].sportsbook,
                    } if ml_odds[0].draw_odds else None,
                }

        game_obj = {
            "id": g.id,
            "sport": g.sport.value if g.sport else None,
            "league": g.league,
            "home_team": g.home_team_name,
            "away_team": g.away_team_name,
            "game_time": g.game_time.isoformat() + "Z" if g.game_time else None,
            "venue": g.venue,
            "status": g.status.value if g.status else None,
            "home_score": g.home_score,
            "away_score": g.away_score,
            "lineups": g.raw_data.get("lineups") if g.raw_data and isinstance(g.raw_data, dict) else None,
            # Prediction (protagonist)
            "prediction": {
                "home_win_prob": round(pred.home_win_prob, 4),
                "away_win_prob": round(pred.away_win_prob, 4),
                "draw_prob": round(pred.draw_prob, 4) if pred.draw_prob else None,
                "predicted_home_score": pred.predicted_home_score,
                "predicted_away_score": pred.predicted_away_score,
                "predicted_total": pred.predicted_total,
                "predicted_spread": pred.predicted_spread,
                "confidence": round(pred.confidence, 4) if pred.confidence else None,
                "model_name": pred.model_name,
                "matchup_factors": pred.matchup_factors,
                "simulation_details": pred.simulation_details,
            } if pred else None,
            # Odds (secondary)
            "odds": odds_summary,
            # Value picks (optional list)
            "value_picks": [{
                "id": pk.id,
                "pick_description": pk.pick_description,
                "model_edge": round(pk.model_edge, 4),
                "expected_value": round(pk.expected_value, 4),
                "value_rating": pk.value_rating,
                "best_odds": pk.best_odds,
                "best_sportsbook": pk.best_sportsbook,
            } for pk in game_picks] if game_picks else [],
        }
        response_games.append(game_obj)

    return {
        "date": target_date.isoformat(),
        "sport": sport.upper() if sport else "ALL",
        "count": len(response_games),
        "games": response_games,
    }



@router.get("/picks")
async def get_value_picks(
    sport: Optional[str] = Query(None),
    min_value: Optional[int] = Query(None, description="Minimum value rating (1-5)"),
    min_edge: Optional[float] = Query(None, description="Minimum edge percentage"),
    game_date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
    db: AsyncSession = Depends(get_db),
):
    query = select(Pick, Prediction).join(Game, Pick.game_id == Game.id).join(Prediction, Pick.prediction_id == Prediction.id)
    
    if game_date:
        target_date = date.fromisoformat(game_date)
        query = query.where(Pick.game_date == target_date)
    else:
        # Default to today in US Eastern Time (GMT-4) for betting slates
        # This perfectly separates late US games from early morning international games
        et_now = datetime.utcnow() - timedelta(hours=4)
        local_today = et_now.date()
        target_date = local_today 
        
        # Calculate the start and end of this ET day in UTC
        # ET 00:00:00 -> UTC 04:00:00
        start_utc = datetime(local_today.year, local_today.month, local_today.day) + timedelta(hours=4)
        end_utc = start_utc + timedelta(days=1)
        
        query = query.where(and_(Game.game_time >= start_utc, Game.game_time < end_utc))

    if sport:
        query = query.where(Pick.sport == sport)
    if min_value:
        query = query.where(Pick.value_rating >= min_value)
    if min_edge:
        query = query.where(Pick.model_edge >= min_edge / 100)

    # Only show picks for games that haven't started yet
    now_utc = datetime.utcnow()
    query = query.where(Game.game_time > now_utc)

    query = query.order_by(Pick.value_rating.desc(), Pick.expected_value.desc())
    result = await db.execute(query)
    rows = result.all()
    picks = [r.Pick for r in rows]
    predictions_map = {r.Pick.id: r.Prediction for r in rows}

    # Also fetch associated games
    game_ids = list(set(p.game_id for p in picks))
    games_map = {}
    if game_ids:
        games_result = await db.execute(
            select(Game).where(Game.id.in_(game_ids))
        )
        for g in games_result.scalars().all():
            games_map[g.id] = g

    return {
        "date": target_date.isoformat() if target_date else "ALL",
        "count": len(picks),
        "picks": [
            {
                "id": p.id,
                "game_id": p.game_id,
                "sport": p.sport.value if p.sport else None,
                "market": p.market.value if p.market else None,
                "pick_type": p.pick_description, # Force UI to show description instead of 'home_ml'
                "pick_description": p.pick_description,
                "model_probability": round(p.model_probability, 4),
                "model_edge": round(p.model_edge, 4),
                "expected_value": round(p.expected_value, 4),
                "best_sportsbook": p.best_sportsbook,
                "best_odds": p.best_odds,
                "best_odds_decimal": round(p.best_odds_decimal, 3),
                "implied_probability": round(p.implied_probability, 4),
                "kelly_fraction": round(p.kelly_fraction, 4) if p.kelly_fraction else None,
                "recommended_stake_pct": round(p.recommended_stake_pct, 2) if p.recommended_stake_pct else None,
                "value_rating": p.value_rating,
                "result": p.result.value if p.result else None,
                "profit_loss": p.profit_loss,
                "matchup": f"{games_map[p.game_id].away_team_name} @ {games_map[p.game_id].home_team_name}" if p.game_id in games_map else "Desconocido",
                "game": {
                    "home_team": games_map[p.game_id].home_team_name,
                    "away_team": games_map[p.game_id].away_team_name,
                    "game_time": games_map[p.game_id].game_time.isoformat() + "Z" if games_map[p.game_id].game_time else None,
                    "venue": games_map[p.game_id].venue,
                    "status": games_map[p.game_id].status.value if games_map[p.game_id].status else None,
                } if p.game_id in games_map else None,
                "insights": {
                    "matchup_factors": predictions_map[p.id].matchup_factors if p.id in predictions_map else None,
                    "simulation_details": predictions_map[p.id].simulation_details if p.id in predictions_map else None,
                }
            }
            for p in picks
        ],
    }


@router.get("/dashboard")
async def get_dashboard(
    db: AsyncSession = Depends(get_db),
):
    """Dashboard summary: today's stats across all sports."""
    today = date.today()

    # Count picks by sport
    picks_query = select(
        Pick.sport,
        func.count(Pick.id).label("total"),
        func.sum(func.IF(Pick.value_rating >= 3, 1, 0)).label("high_value"),
    ).where(Pick.game_date == today).group_by(Pick.sport)

    picks_result = await db.execute(picks_query)
    picks_by_sport = {
        row.sport.value: {"total": row.total, "high_value": int(row.high_value or 0)}
        for row in picks_result
    }

    # Count games by sport
    games_query = select(
        Game.sport,
        func.count(Game.id).label("total"),
    ).where(Game.game_date == today).group_by(Game.sport)

    games_result = await db.execute(games_query)
    games_by_sport = {
        row.sport.value: row.total
        for row in games_result
    }

    # Recent performance (last 7 days)
    week_ago = today - timedelta(days=7)
    perf_query = select(
        Pick.sport,
        func.count(Pick.id).label("total"),
        func.sum(func.IF(Pick.result == PickResult.WIN, 1, 0)).label("wins"),
        func.sum(func.IF(Pick.result == PickResult.LOSS, 1, 0)).label("losses"),
        func.sum(Pick.profit_loss).label("total_pl"),
    ).where(
        and_(
            Pick.game_date >= week_ago,
            Pick.result != PickResult.PENDING,
        )
    ).group_by(Pick.sport)

    perf_result = await db.execute(perf_query)
    performance = {}
    for row in perf_result:
        wins = int(row.wins or 0)
        total = row.total
        performance[row.sport.value] = {
            "total_bets": total,
            "wins": wins,
            "losses": int(row.losses or 0),
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            "profit_loss": round(float(row.total_pl or 0), 2),
        }
    
    top_pick = None
    top_pick_game = None
    top_pick_result = await db.execute(
        select(Pick).where(and_(Pick.game_date == today, Pick.expected_value > 0)).order_by(Pick.expected_value.desc()).limit(1)
    )
    tp = top_pick_result.scalar_one_or_none()
    if tp:
        top_pick = tp
        game_result = await db.execute(select(Game).where(Game.id == tp.game_id))
        top_pick_game = game_result.scalar_one_or_none()

    return {
        "date": today.isoformat(),
        "games_today": games_by_sport,
        "picks_today": picks_by_sport,
        "weekly_performance": performance,
        "top_pick": {
            "id": top_pick.id,
            "sport": top_pick.sport.value,
            "pick_description": top_pick.pick_description,
            "value_rating": top_pick.value_rating,
            "expected_value": round(top_pick.expected_value, 4),
            "best_odds": top_pick.best_odds,
            "best_sportsbook": top_pick.best_sportsbook,
            "matchup": f"{top_pick_game.away_team_name} @ {top_pick_game.home_team_name}" if top_pick_game else "Desconocido"
        } if top_pick else None,
    }


@router.get("/history")
async def get_prediction_history(
    sport: Optional[str] = Query(None),
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    """Get historical prediction results for performance tracking."""
    start_date = date.today() - timedelta(days=days)

    query = select(Pick).where(
        and_(
            Pick.game_date >= start_date,
            Pick.result != PickResult.PENDING,
        )
    )
    if sport:
        query = query.where(Pick.sport == sport)

    query = query.order_by(Pick.game_date.desc())
    result = await db.execute(query)
    picks = result.scalars().all()

    # Calculate stats
    total = len(picks)
    wins = sum(1 for p in picks if p.result == PickResult.WIN)
    losses = sum(1 for p in picks if p.result == PickResult.LOSS)
    total_pl = sum(p.profit_loss or 0 for p in picks)

    return {
        "period_start": start_date.isoformat(),
        "period_end": date.today().isoformat(),
        "summary": {
            "total_picks": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            "total_profit_loss": round(total_pl, 2),
            "roi": round(total_pl / total * 100, 2) if total > 0 else 0,
        },
        "picks": [
            {
                "id": p.id,
                "sport": p.sport.value if p.sport else None,
                "pick_description": p.pick_description,
                "value_rating": p.value_rating,
                "model_edge": round(p.model_edge, 4),
                "best_odds": p.best_odds,
                "result": p.result.value if p.result else None,
                "profit_loss": p.profit_loss,
                "game_date": p.game_date.isoformat(),
            }
            for p in picks
        ],
    }
