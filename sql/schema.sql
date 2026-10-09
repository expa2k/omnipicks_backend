-- ============================================================
-- OmniPicks — MySQL Database Schema
-- Run: mysql -u root -p < schema.sql
-- ============================================================

CREATE DATABASE IF NOT EXISTS omnipicks
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE omnipicks;

-- ── Teams ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS teams (
    id INT AUTO_INCREMENT PRIMARY KEY,
    external_id VARCHAR(50) NOT NULL,
    sport ENUM('nba','nfl','mlb','soccer') NOT NULL,
    name VARCHAR(200) NOT NULL,
    short_name VARCHAR(20),
    logo_url VARCHAR(500),
    league VARCHAR(200),
    country VARCHAR(100),
    elo_rating FLOAT DEFAULT 1500.0,
    offensive_rating FLOAT,
    defensive_rating FLOAT,
    wins INT DEFAULT 0,
    losses INT DEFAULT 0,
    draws INT DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE INDEX idx_team_sport_ext (sport, external_id),
    INDEX idx_team_sport (sport)
) ENGINE=InnoDB;

-- ── Games ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS games (
    id INT AUTO_INCREMENT PRIMARY KEY,
    external_id VARCHAR(100) NOT NULL,
    sport ENUM('nba','nfl','mlb','soccer') NOT NULL,
    league VARCHAR(200),
    season VARCHAR(20),
    home_team_id INT NOT NULL,
    away_team_id INT NOT NULL,
    home_team_name VARCHAR(200) NOT NULL,
    away_team_name VARCHAR(200) NOT NULL,
    game_date DATE NOT NULL,
    game_time DATETIME,
    status ENUM('scheduled','live','finished','postponed','cancelled') DEFAULT 'scheduled',
    venue VARCHAR(300),
    home_score INT,
    away_score INT,
    home_rest_days INT,
    away_rest_days INT,
    is_playoff TINYINT(1) DEFAULT 0,
    weather_temp FLOAT,
    weather_wind FLOAT,
    raw_data JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE INDEX idx_game_ext (sport, external_id),
    INDEX idx_game_sport_date (sport, game_date),
    INDEX idx_game_status (status)
) ENGINE=InnoDB;

-- ── Predictions ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    game_id INT NOT NULL,
    sport ENUM('nba','nfl','mlb','soccer') NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    home_win_prob FLOAT NOT NULL,
    away_win_prob FLOAT NOT NULL,
    draw_prob FLOAT,
    predicted_home_score FLOAT,
    predicted_away_score FLOAT,
    predicted_total FLOAT,
    predicted_spread FLOAT,
    confidence FLOAT,
    feature_importances JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_pred_game (game_id),
    INDEX idx_pred_sport_date (sport, created_at)
) ENGINE=InnoDB;

-- ── Odds Snapshots ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS odds_snapshots (
    id INT AUTO_INCREMENT PRIMARY KEY,
    game_id INT NOT NULL,
    sport ENUM('nba','nfl','mlb','soccer') NOT NULL,
    sportsbook VARCHAR(100) NOT NULL,
    market ENUM('moneyline','spread','over_under') NOT NULL,
    home_odds FLOAT,
    away_odds FLOAT,
    draw_odds FLOAT,
    spread_home FLOAT,
    spread_home_odds FLOAT,
    spread_away_odds FLOAT,
    total_line FLOAT,
    over_odds FLOAT,
    under_odds FLOAT,
    implied_home_prob FLOAT,
    implied_away_prob FLOAT,
    implied_draw_prob FLOAT,
    snapshot_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_odds_game (game_id),
    INDEX idx_odds_sport_time (sport, snapshot_time)
) ENGINE=InnoDB;

-- ── Value Picks ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS picks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    game_id INT NOT NULL,
    sport ENUM('nba','nfl','mlb','soccer') NOT NULL,
    prediction_id INT NOT NULL,
    market ENUM('moneyline','spread','over_under') NOT NULL,
    pick_type VARCHAR(50) NOT NULL,
    pick_description VARCHAR(500) NOT NULL,
    model_probability FLOAT NOT NULL,
    model_edge FLOAT NOT NULL,
    expected_value FLOAT NOT NULL,
    best_sportsbook VARCHAR(100) NOT NULL,
    best_odds FLOAT NOT NULL,
    best_odds_decimal FLOAT NOT NULL,
    implied_probability FLOAT NOT NULL,
    kelly_fraction FLOAT,
    recommended_stake_pct FLOAT,
    value_rating INT NOT NULL,
    result ENUM('pending','win','loss','push','void') DEFAULT 'pending',
    profit_loss FLOAT,
    game_date DATE NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    settled_at DATETIME,
    INDEX idx_pick_game (game_id),
    INDEX idx_pick_sport_date (sport, game_date),
    INDEX idx_pick_result (result),
    INDEX idx_pick_value (value_rating)
) ENGINE=InnoDB;

-- ── Bankroll History ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS bankroll_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date DATE NOT NULL,
    starting_balance FLOAT NOT NULL,
    ending_balance FLOAT NOT NULL,
    total_wagered FLOAT DEFAULT 0,
    total_won FLOAT DEFAULT 0,
    total_lost FLOAT DEFAULT 0,
    num_bets INT DEFAULT 0,
    num_wins INT DEFAULT 0,
    num_losses INT DEFAULT 0,
    daily_roi FLOAT,
    cumulative_roi FLOAT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE INDEX idx_bankroll_date (date)
) ENGINE=InnoDB;

-- ── Model Performance ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_performance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    sport ENUM('nba','nfl','mlb','soccer') NOT NULL,
    date DATE NOT NULL,
    total_predictions INT DEFAULT 0,
    correct_predictions INT DEFAULT 0,
    accuracy FLOAT,
    brier_score FLOAT,
    log_loss_val FLOAT,
    roi FLOAT,
    value_picks_total INT DEFAULT 0,
    value_picks_won INT DEFAULT 0,
    value_picks_roi FLOAT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_perf_model_sport (model_name, sport, date)
) ENGINE=InnoDB;
