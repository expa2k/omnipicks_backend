"""
OmniPicks — Prediction Engine Orchestrator
Coordinates feature engineering, models, and value calculation.
"""
import logging
from datetime import date
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db_models import Game, Prediction, OddsSnapshot, Pick, PickResult, SportType, MarketType
from app.services.feature_engineering import FeatureEngineer
from app.models.xgboost_model import XGBoostPredictor
from app.models.poisson_model import DixonColesModel
from app.services.simulation_engine import SimulationEngine
from app.services.value_calculator import ValueCalculator
from app.services.bzzoiro_client import BzzoiroClient
from app.services.meta_model import SoccerMetaModel

logger = logging.getLogger(__name__)

class PredictionEngine:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.feature_eng = FeatureEngineer()
        self.simulation_engine = SimulationEngine(num_simulations=5000)
        self.value_calc = ValueCalculator()
        self.bzzoiro = BzzoiroClient()
        self.meta_model = SoccerMetaModel(self.db)

    async def run_predictions_for_date(self, target_date: date):
        """Run ML predictions for all games on a specific date."""
        logger.info(f"Running predictions for {target_date}")
        
        games_result = await self.db.execute(select(Game).where(Game.game_date == target_date))
        games = games_result.scalars().all()
        
        # Clear existing picks for today to avoid duplicates
        from sqlalchemy import delete
        await self.db.execute(delete(Pick).where(Pick.game_date == target_date))
        await self.db.flush()
        
        for game in games:
            game_date_str = game.game_date.isoformat() if game.game_date else None
            
            # 1. Compute real advanced features for ALL sports (including Soccer via ESPN/API-Sports)
            features = await self.feature_eng.compute_features(game.raw_data or {}, game.sport.value, game_date_str)
            
            if game.sport == SportType.SOCCER:
                # 2a. Organic Simulation based purely on real data fetched
                sim_result = self.simulation_engine.simulate(game.sport, features)
                
                # 2b. Fetch ML Prediction from Bzzoiro for hybrid blending
                bz_pred = await self.bzzoiro.get_event_prediction(game.external_id)
                
                if bz_pred and "markets" in bz_pred:
                    bz_home_prob = bz_pred.get("markets", {}).get("match_result", {}).get("prob_home", 33.3) / 100.0
                    bz_away_prob = bz_pred.get("markets", {}).get("match_result", {}).get("prob_away", 33.3) / 100.0
                    bz_home_xg = bz_pred.get("markets", {}).get("expected_goals", {}).get("home", 1.0)
                    bz_away_xg = bz_pred.get("markets", {}).get("expected_goals", {}).get("away", 1.0)
                    
                    # 70% Real Data (ESPN/API-Sports) + 30% Bzzoiro ML
                    w_org, w_bz = 0.70, 0.30
                    
                    final_home_prob = (sim_result["home_win_prob"] * w_org) + (bz_home_prob * w_bz)
                    final_away_prob = (sim_result["away_win_prob"] * w_org) + (bz_away_prob * w_bz)
                    final_draw = max(0.01, 1.0 - (final_home_prob + final_away_prob))
                    
                    final_home_xg = (sim_result["predicted_home_score"] * w_org) + (bz_home_xg * w_bz)
                    final_away_xg = (sim_result["predicted_away_score"] * w_org) + (bz_away_xg * w_bz)
                    
                    pred_result = {
                        "home_win_prob": final_home_prob,
                        "away_win_prob": final_away_prob,
                        "draw_prob": final_draw,
                        "predicted_home_score": round(final_home_xg, 2),
                        "predicted_away_score": round(final_away_xg, 2),
                        "predicted_total": round(final_home_xg + final_away_xg, 2),
                        "predicted_spread": round(final_away_xg - final_home_xg, 2),
                        "confidence": sim_result["confidence"],
                        "simulation_details": {
                            "source": "Hybrid: MonteCarlo RealData (70%) + Bzzoiro ML (30%)",
                            "organic_xg": f"{sim_result['predicted_home_score']} - {sim_result['predicted_away_score']}",
                            "bzzoiro_xg": f"{bz_home_xg} - {bz_away_xg}"
                        }
                    }
                    model_name = "omnipicks_hybrid_v4"
                else:
                    # Fallback to pure organic simulation if Bzzoiro fails
                    pred_result = sim_result
                    model_name = "monte_carlo_soccer_v2"
            else:
                # 1. Run Models & Simulations for US Sports
                pred_result = self.simulation_engine.simulate(game.sport, features)
                model_name = "monte_carlo_v1"
            
            # 2. Save Prediction to DB
            prediction = Prediction(
                game_id=game.id,
                sport=game.sport,
                model_name=model_name,
                home_win_prob=pred_result.get("home_win_prob", 0.5),
                away_win_prob=pred_result.get("away_win_prob", 0.5),
                draw_prob=pred_result.get("draw_prob"),
                predicted_home_score=pred_result.get("predicted_home_score"),
                predicted_away_score=pred_result.get("predicted_away_score"),
                predicted_total=pred_result.get("predicted_total"),
                predicted_spread=pred_result.get("predicted_spread"),
                confidence=pred_result.get("confidence"),
                matchup_factors=features,
                simulation_details=pred_result.get("simulation_details")
            )
            self.db.add(prediction)
            await self.db.flush() # get ID
            
            # 3. Get Odds for this game
            odds_result = await self.db.execute(
                select(OddsSnapshot).where(OddsSnapshot.game_id == game.id).order_by(OddsSnapshot.snapshot_time.desc())
            )
            all_odds = odds_result.scalars().all()
            
            # Deduplicate odds: only keep the latest per sportsbook and market
            seen = set()
            odds_list = []
            for o in all_odds:
                key = (o.sportsbook, o.market)
                if key not in seen:
                    seen.add(key)
                    odds_list.append(o)
            
            # 4. Calculate Value Picks
            picks = self.value_calc.find_value_picks(prediction, odds_list, game)
            for p in picks:
                p_db = Pick(
                    game_id=game.id,
                    sport=game.sport,
                    prediction_id=prediction.id,
                    market=p.get("market", MarketType.MONEYLINE),
                    pick_type=p.get("pick_type"),
                    pick_description=p.get("pick_description"),
                    model_probability=p.get("model_probability"),
                    model_edge=p.get("model_edge"),
                    expected_value=p.get("expected_value"),
                    best_sportsbook=p.get("best_sportsbook"),
                    best_odds=p.get("best_odds"),
                    best_odds_decimal=p.get("best_odds_decimal"),
                    implied_probability=p.get("implied_probability"),
                    value_rating=p.get("value_rating"),
                    game_date=game.game_date
                )
                self.db.add(p_db)

        await self.db.commit()
        logger.info(f"Generated predictions for {len(games)} games")
