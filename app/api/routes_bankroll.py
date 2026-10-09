"""
OmniPicks — Bankroll Management API Routes
"""

from fastapi import APIRouter, Depends, Query, HTTPException, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from datetime import date, datetime, timedelta
from typing import Optional
from pydantic import BaseModel

from app.database import get_db
from app.db_models import BankrollHistory, Pick, PickResult

router = APIRouter()


class BankrollUpdate(BaseModel):
    starting_balance: float


@router.get("/summary")
async def get_bankroll_summary(
    db: AsyncSession = Depends(get_db),
):
    """Get current bankroll status and overall performance."""
    # Get latest bankroll entry
    latest = await db.execute(
        select(BankrollHistory).order_by(BankrollHistory.date.desc()).limit(1)
    )
    latest = latest.scalar_one_or_none()

    # Get all-time stats from picks
    all_picks = await db.execute(
        select(
            func.count(Pick.id).label("total"),
            func.sum(func.IF(Pick.result == PickResult.WIN, 1, 0)).label("wins"),
            func.sum(func.IF(Pick.result == PickResult.LOSS, 1, 0)).label("losses"),
            func.sum(func.IF(Pick.result == PickResult.PUSH, 1, 0)).label("pushes"),
            func.sum(Pick.profit_loss).label("total_pl"),
        ).where(Pick.result != PickResult.PENDING)
    )
    stats = all_picks.one()

    total = stats.total or 0
    wins = int(stats.wins or 0)
    losses = int(stats.losses or 0)
    pushes = int(stats.pushes or 0)
    total_pl = float(stats.total_pl or 0)

    return {
        "current_balance": latest.ending_balance if latest else 0,
        "starting_balance": latest.starting_balance if latest else 0,
        "all_time": {
            "total_bets": total,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            "total_profit_loss": round(total_pl, 2),
            "roi": round(total_pl / total * 100, 2) if total > 0 else 0,
        },
        "last_updated": latest.date.isoformat() if latest else None,
    }


@router.get("/history")
async def get_bankroll_history(
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    """Get bankroll history for charting."""
    start_date = date.today() - timedelta(days=days)

    result = await db.execute(
        select(BankrollHistory).where(
            BankrollHistory.date >= start_date
        ).order_by(BankrollHistory.date)
    )
    history = result.scalars().all()

    return {
        "period": f"Last {days} days",
        "data": [
            {
                "date": h.date.isoformat(),
                "balance": h.ending_balance,
                "daily_pl": round(h.ending_balance - h.starting_balance, 2),
                "num_bets": h.num_bets,
                "wins": h.num_wins,
                "losses": h.num_losses,
                "daily_roi": round(h.daily_roi, 2) if h.daily_roi else 0,
                "cumulative_roi": round(h.cumulative_roi, 2) if h.cumulative_roi else 0,
            }
            for h in history
        ],
    }


@router.post("/set-balance")
async def set_starting_balance(
    data: BankrollUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Set or update starting bankroll balance."""
    today = date.today()

    # Check if entry exists for today
    existing = await db.execute(
        select(BankrollHistory).where(BankrollHistory.date == today)
    )
    existing = existing.scalar_one_or_none()

    if existing:
        existing.starting_balance = data.starting_balance
        existing.ending_balance = data.starting_balance
    else:
        new_entry = BankrollHistory(
            date=today,
            starting_balance=data.starting_balance,
            ending_balance=data.starting_balance,
        )
        db.add(new_entry)

    await db.flush()

    return {
        "message": "Balance updated",
        "date": today.isoformat(),
        "balance": data.starting_balance,
    }


@router.get("/kelly-calculator")
async def kelly_calculator(
    probability: float = Query(..., ge=0, le=1, description="Your estimated win probability"),
    odds: float = Query(..., description="American odds (e.g., +150 or -110)"),
    bankroll: float = Query(default=1000, description="Current bankroll"),
    fraction: float = Query(default=0.25, ge=0.1, le=1.0, description="Kelly fraction (0.25 = quarter)"),
):
    """Interactive Kelly Criterion calculator."""
    # Convert American to decimal
    if odds > 0:
        decimal_odds = (odds / 100) + 1
    else:
        decimal_odds = (100 / abs(odds)) + 1

    # Implied probability
    if odds > 0:
        implied_prob = 100 / (odds + 100)
    else:
        implied_prob = abs(odds) / (abs(odds) + 100)

    # Full Kelly
    b = decimal_odds - 1
    q = 1 - probability
    full_kelly = (b * probability - q) / b if b > 0 else 0
    full_kelly = max(0, full_kelly)  # No negative bets

    # Fractional Kelly
    frac_kelly = full_kelly * fraction

    # Expected Value
    ev = (probability * decimal_odds) - 1

    # Edge
    edge = probability - implied_prob

    return {
        "inputs": {
            "probability": probability,
            "american_odds": odds,
            "decimal_odds": round(decimal_odds, 3),
            "bankroll": bankroll,
            "kelly_fraction": fraction,
        },
        "results": {
            "implied_probability": round(implied_prob, 4),
            "edge": round(edge, 4),
            "edge_pct": round(edge * 100, 2),
            "expected_value": round(ev, 4),
            "is_value_bet": ev > 0,
            "full_kelly_pct": round(full_kelly * 100, 2),
            "fractional_kelly_pct": round(frac_kelly * 100, 2),
            "recommended_stake": round(bankroll * frac_kelly, 2),
            "potential_profit": round(bankroll * frac_kelly * (decimal_odds - 1), 2),
        },
    }
