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


# ══════════════════════════════════════════════════════════════════════
# NBA v2 — histórico propio (fuente: nba_api / ESPN)
# ══════════════════════════════════════════════════════════════════════

class NBAGame(Base):
    """Un partido de NBA (id oficial de la NBA)."""
    __tablename__ = "nba_games"

    game_id = Column(String(12), primary_key=True)
    season = Column(String(7), nullable=False)            # '2026-27'
    season_type = Column(String(20), nullable=False)      # 'Regular Season' | 'Playoffs'
    game_date = Column(Date, nullable=False)
    home_team_id = Column(Integer, nullable=False)
    away_team_id = Column(Integer, nullable=False)
    home_pts = Column(Integer, nullable=True)
    away_pts = Column(Integer, nullable=True)
    stints_done = Column(Boolean, default=False, nullable=False)

    __table_args__ = (
        Index("ix_nba_games_date", "game_date"),
        Index("ix_nba_games_season", "season", "season_type"),
    )


class NBAPlayerGame(Base):
    """Box score de un jugador en un partido."""
    __tablename__ = "nba_player_games"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String(12), nullable=False)
    season = Column(String(7), nullable=False)
    game_date = Column(Date, nullable=False)
    team_id = Column(Integer, nullable=False)
    opp_team_id = Column(Integer, nullable=False)
    is_home = Column(Boolean, nullable=False)
    player_id = Column(Integer, nullable=False)
    player_name = Column(String(100), nullable=False)
    minutes = Column(Float, nullable=False, default=0.0)
    pts = Column(Integer, default=0)
    fgm = Column(Integer, default=0)
    fga = Column(Integer, default=0)
    fg3m = Column(Integer, default=0)
    fg3a = Column(Integer, default=0)
    ftm = Column(Integer, default=0)
    fta = Column(Integer, default=0)
    oreb = Column(Integer, default=0)
    dreb = Column(Integer, default=0)
    ast = Column(Integer, default=0)
    stl = Column(Integer, default=0)
    blk = Column(Integer, default=0)
    tov = Column(Integer, default=0)
    pf = Column(Integer, default=0)
    plus_minus = Column(Float, default=0.0)

    __table_args__ = (
        Index("ux_nba_pg_game_player", "game_id", "player_id", unique=True),
        Index("ix_nba_pg_player_date", "player_id", "game_date"),
        Index("ix_nba_pg_team_date", "team_id", "game_date"),
    )


class NBAStint(Base):
    """Tramo de partido con los mismos 10 jugadores en cancha."""
    __tablename__ = "nba_stints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String(12), nullable=False)
    start_t = Column(Integer, nullable=False)             # décimas de segundo desde el inicio
    end_t = Column(Integer, nullable=False)
    home_players = Column(String(80), nullable=False)     # 5 ids ordenados, separados por coma
    away_players = Column(String(80), nullable=False)
    home_pts = Column(Integer, nullable=False, default=0)
    away_pts = Column(Integer, nullable=False, default=0)
    home_poss = Column(Integer, nullable=False, default=0)
    away_poss = Column(Integer, nullable=False, default=0)

    __table_args__ = (Index("ix_nba_stints_game", "game_id"),)


class NBAInjuryReport(Base):
    """Cada cambio de estado de lesión visto en ESPN (historial point-in-time)."""
    __tablename__ = "nba_injury_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    captured_at = Column(DateTime, server_default=func.now(), nullable=False)
    espn_athlete_id = Column(String(20), nullable=False)
    player_name = Column(String(100), nullable=False)
    team_name = Column(String(100), nullable=False)
    status = Column(String(30), nullable=False)           # Out | Doubtful | Questionable | Probable | Day-To-Day
    injury_type = Column(String(100), nullable=True)
    comment = Column(Text, nullable=True)
    report_date = Column(DateTime, nullable=True)         # fecha que ESPN da al reporte
    is_current = Column(Boolean, default=True, nullable=False)
    ended_at = Column(DateTime, nullable=True)            # cuándo dejó de valer (cambió de estado o salió del reporte)

    __table_args__ = (
        Index("ix_nba_inj_athlete", "espn_athlete_id", "captured_at"),
        Index("ix_nba_inj_current", "is_current"),
    )


class NBAPlayerSeason(Base):
    """Ficha de un jugador en una temporada (edad y medidas)."""
    __tablename__ = "nba_player_seasons"

    season = Column(String(7), primary_key=True)
    player_id = Column(Integer, primary_key=True)
    player_name = Column(String(100), nullable=True)
    age = Column(Float, nullable=True)
    height_in = Column(Float, nullable=True)
    weight_lb = Column(Float, nullable=True)


class NBAPlayer(Base):
    """Ficha fija de un jugador (posición oficial y medidas)."""
    __tablename__ = "nba_players"

    player_id = Column(Integer, primary_key=True)
    player_name = Column(String(100), nullable=True)
    position = Column(String(5), nullable=True)            # G, F, C, G-F, F-C, ...
    height_in = Column(Float, nullable=True)
    weight_lb = Column(Float, nullable=True)


class NBAPlayerRating(Base):
    """Rating de un jugador en una fecha (se usa solo info anterior a esa fecha)."""
    __tablename__ = "nba_player_ratings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    as_of_date = Column(Date, nullable=False)
    player_id = Column(Integer, nullable=False)
    player_name = Column(String(100), nullable=True)
    o_rating = Column(Float, nullable=False)              # pts/100 pos sobre el promedio (ataque)
    d_rating = Column(Float, nullable=False)              # pts/100 pos sobre el promedio (defensa; + = bueno)
    uncertainty = Column(Float, nullable=True)
    minutes_sample = Column(Float, nullable=True)
    usage_rate = Column(Float, nullable=True)             # (FGA + 0.44·FTA + TOV) por 100 posesiones en cancha
    model_version = Column(String(30), nullable=False, default="v1")

    __table_args__ = (
        Index("ux_nba_rating", "as_of_date", "player_id", "model_version", unique=True),
        Index("ix_nba_rating_player", "player_id", "as_of_date"),
    )


class NBAPredictionLog(Base):
    """Última predicción ANTES del partido (la que se mide en vivo), con la línea del mercado."""
    __tablename__ = "nba_prediction_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, nullable=False)                  # games.id
    model_name = Column(String(50), nullable=False)
    game_date = Column(Date, nullable=False)
    game_time = Column(DateTime, nullable=True)                # UTC
    home_team = Column(String(100), nullable=False)
    away_team = Column(String(100), nullable=False)
    predicted_at = Column(DateTime, nullable=False)            # UTC
    minutes_before_tip = Column(Float, nullable=True)
    home_win_prob = Column(Float, nullable=False)
    pred_margin = Column(Float, nullable=False)                # local − visita
    pred_total = Column(Float, nullable=True)
    market_margin = Column(Float, nullable=True)               # −spread del local (ESPN/DraftKings)
    market_total = Column(Float, nullable=True)
    market_provider = Column(String(50), nullable=True)
    ml_home = Column(Float, nullable=True)                     # momios americanos al momento de predecir
    ml_away = Column(Float, nullable=True)
    spread_home_odds = Column(Float, nullable=True)
    spread_away_odds = Column(Float, nullable=True)
    over_odds = Column(Float, nullable=True)
    under_odds = Column(Float, nullable=True)
    home_pts = Column(Integer, nullable=True)
    away_pts = Column(Integer, nullable=True)
    settled_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ux_nba_log_game_model", "game_id", "model_name", unique=True),
        Index("ix_nba_log_date", "game_date"),
    )


class NBAOfficialInjury(Base):
    """Reporte oficial de lesiones de la NBA (PDF de la liga), una fila por jugador y reporte."""
    __tablename__ = "nba_official_injuries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_time = Column(DateTime, nullable=False)             # hora del reporte (ET)
    game_date = Column(Date, nullable=False)
    matchup = Column(String(20), nullable=True)                # BOS@TOR
    team_name = Column(String(100), nullable=False)
    player_name = Column(String(100), nullable=False)          # "Nombre Apellido"
    status = Column(String(30), nullable=False)                # Out | Doubtful | Questionable | Probable | Available
    reason = Column(String(200), nullable=True)
    is_current = Column(Boolean, default=True, nullable=False)

    __table_args__ = (
        Index("ix_nba_off_current", "is_current", "game_date"),
        Index("ix_nba_off_report", "report_time"),
    )
