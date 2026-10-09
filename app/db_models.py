"""
OmniPicks — SQLAlchemy ORM Models
All database table definitions for MySQL.
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, Boolean,
    Enum as SAEnum, Index, JSON, Date,
)
from sqlalchemy.sql import func
from app.database import Base
import enum


# ── Enums ─────────────────────────────────────────────────────────────

class SportType(str, enum.Enum):
    NBA = "NBA"
    NFL = "NFL"
    MLB = "MLB"
    SOCCER = "SOCCER"


class GameStatus(str, enum.Enum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class PickResult(str, enum.Enum):
    PENDING = "pending"
    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    VOID = "void"


class MarketType(str, enum.Enum):
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    OVER_UNDER = "over_under"
    BTTS = "btts"


# ── Teams ─────────────────────────────────────────────────────────────

class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(String(50), nullable=False)
    sport = Column(SAEnum(SportType), nullable=False)
    name = Column(String(200), nullable=False)
    short_name = Column(String(20), nullable=True)
    logo_url = Column(String(500), nullable=True)
    league = Column(String(200), nullable=True)
    country = Column(String(100), nullable=True)

    # Current ratings (updated after each game)
    elo_rating = Column(Float, default=1500.0)
    offensive_rating = Column(Float, nullable=True)
    defensive_rating = Column(Float, nullable=True)

    # Record
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    draws = Column(Integer, default=0)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_team_sport_ext", "sport", "external_id", unique=True),
        Index("idx_team_sport", "sport"),
    )


class TeamHistoricalProfile(Base):
    """Stores the Similar-Opponent Benchmarking data (Dimension A & B)"""
    __tablename__ = "team_historical_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    team_name = Column(String(200), nullable=False)
    opponent_tier = Column(Integer, nullable=False)
    timeframe = Column(String(50), nullable=False) # e.g., "last_year", "recent_5_games"
    avg_goals_for = Column(Float, nullable=False)
    avg_goals_against = Column(Float, nullable=False)
    win_rate = Column(Float, nullable=False)
    matches_played = Column(Integer, nullable=False)
    unavailable_players_count = Column(Integer, default=0, nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_historical_profile", "team_name", "opponent_tier", "timeframe", unique=True),
    )


class TournamentSquad(Base):
    """Caches the squad rating for a specific tournament (Dimension C)"""
    __tablename__ = "tournament_squads"

    id = Column(Integer, primary_key=True, autoincrement=True)
    team_name = Column(String(200), nullable=False)
    league_id = Column(String(50), nullable=False)
    season = Column(String(20), nullable=False)
    squad_rating = Column(Float, nullable=False)
    players_json = Column(JSON, nullable=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_tournament_squad", "team_name", "league_id", "season", unique=True),
    )


# ── Games ─────────────────────────────────────────────────────────────

class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(String(100), nullable=False)
    sport = Column(SAEnum(SportType), nullable=False)
    league = Column(String(200), nullable=True)
    season = Column(String(20), nullable=True)

    home_team_id = Column(Integer, nullable=False)
    away_team_id = Column(Integer, nullable=False)
    home_team_name = Column(String(200), nullable=False)
    away_team_name = Column(String(200), nullable=False)

    game_date = Column(Date, nullable=False)
    game_time = Column(DateTime, nullable=True)
    status = Column(SAEnum(GameStatus), default=GameStatus.SCHEDULED)
    venue = Column(String(300), nullable=True)

    # Scores
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)

    # Additional context
    home_rest_days = Column(Integer, nullable=True)
    away_rest_days = Column(Integer, nullable=True)
    is_playoff = Column(Boolean, default=False)
    weather_temp = Column(Float, nullable=True)
    weather_wind = Column(Float, nullable=True)

    # Raw API response (for debugging)
    raw_data = Column(JSON, nullable=True)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_game_sport_date", "sport", "game_date"),
        Index("idx_game_ext", "sport", "external_id", unique=True),
        Index("idx_game_status", "status"),
    )


# ── Predictions ───────────────────────────────────────────────────────

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, nullable=False)
    sport = Column(SAEnum(SportType), nullable=False)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=True)

    # Probabilities
    home_win_prob = Column(Float, nullable=False)
    away_win_prob = Column(Float, nullable=False)
    draw_prob = Column(Float, nullable=True)  # Soccer only

    # Score predictions
    predicted_home_score = Column(Float, nullable=True)
    predicted_away_score = Column(Float, nullable=True)
    predicted_total = Column(Float, nullable=True)
    predicted_spread = Column(Float, nullable=True)

    # Confidence
    confidence = Column(Float, nullable=True)  # 0-1

    # Feature importances (top 5)
    feature_importances = Column(JSON, nullable=True)
    
    # Feature Store and Simulation Engine results
    matchup_factors = Column(JSON, nullable=True)
    simulation_details = Column(JSON, nullable=True)

    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_pred_game", "game_id"),
        Index("idx_pred_sport_date", "sport", "created_at"),
    )


# ── Odds Snapshots ────────────────────────────────────────────────────

class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, nullable=False)
    sport = Column(SAEnum(SportType), nullable=False)
    sportsbook = Column(String(100), nullable=False)
    market = Column(SAEnum(MarketType), nullable=False)

    # Moneyline odds (American format)
    home_odds = Column(Float, nullable=True)
    away_odds = Column(Float, nullable=True)
    draw_odds = Column(Float, nullable=True)

    # Spread
    spread_home = Column(Float, nullable=True)
    spread_home_odds = Column(Float, nullable=True)
    spread_away_odds = Column(Float, nullable=True)

    # Over/Under
    total_line = Column(Float, nullable=True)
    over_odds = Column(Float, nullable=True)
    under_odds = Column(Float, nullable=True)

    # Implied probabilities (calculated)
    implied_home_prob = Column(Float, nullable=True)
    implied_away_prob = Column(Float, nullable=True)
    implied_draw_prob = Column(Float, nullable=True)

    snapshot_time = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_odds_game", "game_id"),
        Index("idx_odds_sport_time", "sport", "snapshot_time"),
    )


# ── Value Picks ───────────────────────────────────────────────────────

class Pick(Base):
    __tablename__ = "picks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, nullable=False)
    sport = Column(SAEnum(SportType), nullable=False)
    prediction_id = Column(Integer, nullable=False)

    # What we're betting on
    market = Column(SAEnum(MarketType), nullable=False)
    pick_type = Column(String(50), nullable=False)  # "home_ml", "away_ml", "over", "under", "home_spread", etc.
    pick_description = Column(String(500), nullable=False)

    # Our model's numbers
    model_probability = Column(Float, nullable=False)
    model_edge = Column(Float, nullable=False)  # model_prob - implied_prob
    expected_value = Column(Float, nullable=False)

    # Best sportsbook for this pick
    best_sportsbook = Column(String(100), nullable=False)
    best_odds = Column(Float, nullable=False)  # American odds
    best_odds_decimal = Column(Float, nullable=False)
    implied_probability = Column(Float, nullable=False)

    # Kelly sizing
    kelly_fraction = Column(Float, nullable=True)
    recommended_stake_pct = Column(Float, nullable=True)  # % of bankroll

    # Value rating (1-5 stars)
    value_rating = Column(Integer, nullable=False)  # 1-5

    # Result tracking
    result = Column(SAEnum(PickResult), default=PickResult.PENDING)
    profit_loss = Column(Float, nullable=True)

    game_date = Column(Date, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    settled_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_pick_game", "game_id"),
        Index("idx_pick_sport_date", "sport", "game_date"),
        Index("idx_pick_result", "result"),
        Index("idx_pick_value", "value_rating"),
    )


# ── Bankroll History ──────────────────────────────────────────────────

class BankrollHistory(Base):
    __tablename__ = "bankroll_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    starting_balance = Column(Float, nullable=False)
    ending_balance = Column(Float, nullable=False)
    total_wagered = Column(Float, default=0.0)
    total_won = Column(Float, default=0.0)
    total_lost = Column(Float, default=0.0)
    num_bets = Column(Integer, default=0)
    num_wins = Column(Integer, default=0)
    num_losses = Column(Integer, default=0)
    daily_roi = Column(Float, nullable=True)
    cumulative_roi = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_bankroll_date", "date", unique=True),
    )


# ── Model Performance Tracking ────────────────────────────────────────

class ModelPerformance(Base):
    __tablename__ = "model_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    sport = Column(SAEnum(SportType), nullable=False)
    date = Column(Date, nullable=False)

    total_predictions = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    accuracy = Column(Float, nullable=True)
    brier_score = Column(Float, nullable=True)
    log_loss = Column(Float, nullable=True)
    roi = Column(Float, nullable=True)

    # Value picks performance
    value_picks_total = Column(Integer, default=0)
    value_picks_won = Column(Integer, default=0)
    value_picks_roi = Column(Float, nullable=True)

    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_perf_model_sport", "model_name", "sport", "date"),
    )
