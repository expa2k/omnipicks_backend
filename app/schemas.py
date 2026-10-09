"""
OmniPicks — Schemas
Pydantic schemas for API responses
"""
from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime

class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

class TeamBase(BaseSchema):
    id: int
    name: str
    short_name: Optional[str] = None
    logo_url: Optional[str] = None
    elo_rating: float
    record: str

class GameResponse(BaseSchema):
    id: int
    sport: str
    league: Optional[str]
    home_team: str
    away_team: str
    game_time: Optional[str]
    status: Optional[str]
    venue: Optional[str]
    home_score: Optional[int]
    away_score: Optional[int]
    is_playoff: bool

class PredictionResponse(BaseSchema):
    id: int
    game_id: int
    sport: str
    home_win_prob: float
    away_win_prob: float
    draw_prob: Optional[float]
    predicted_home_score: Optional[float]
    predicted_away_score: Optional[float]
    predicted_total: Optional[float]
    predicted_spread: Optional[float]
    confidence: Optional[float]

class PickResponse(BaseSchema):
    id: int
    sport: str
    pick_description: str
    model_edge: float
    expected_value: float
    best_odds: float
    best_sportsbook: str
    value_rating: int
    game_date: str
    result: Optional[str]
