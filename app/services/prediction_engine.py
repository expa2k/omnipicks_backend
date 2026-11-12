"""
OmniPicks — Prediction Engine Orchestrator
Coordinates feature engineering, models, and value calculation.
"""
import asyncio

from sqlalchemy.exc import DBAPIError
import logging
import os
from datetime import date
from typing import Any, Dict, List, Optional
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

    async def run_predictions_for_date(self, target_date: date, sports: Optional[List[SportType]] = None):
        """Run ML predictions for the games on a specific date.
        `sports` limita la corrida a ciertos deportes (None = todos)."""
        logger.info(f"Running predictions for {target_date}" + (f" ({[s.value for s in sports]})" if sports else ""))

        games_query = select(Game).where(Game.game_date == target_date)
        if sports:
            games_query = games_query.where(Game.sport.in_(sports))
        games_result = await self.db.execute(games_query)
        games = games_result.scalars().all()

        # Un partido que ya empezó (o terminó) no se vuelve a predecir: su predicción queda
        # congelada tal como estaba antes del inicio, que es la que se mide.
        from datetime import datetime
        from app.db_models import GameStatus
        now = datetime.utcnow()
        started = [g for g in games if g.status not in (None, GameStatus.SCHEDULED)
                   or (g.game_time is not None and g.game_time <= now)]
        if started:
            logger.info(f"{len(started)} partidos ya empezaron: se conserva su predicción previa")
        games = [g for g in games if g not in started]
        if not games:
            return

        # Clear existing picks of the games being re-predicted (avoid duplicates)
        from sqlalchemy import delete
        await self.db.execute(delete(Pick).where(Pick.game_id.in_([g.id for g in games])))
        await self.db.flush()

        # NBA v2 (modelo de jugadores): se calcula todo el día de una vez; si falla, queda v1
        nba_v2 = {}
        nba_games = [g for g in games if g.sport == SportType.NBA]
        if nba_games and os.getenv("NBA_MODEL", "v2").lower() != "v1":
            from app.services.nba_v2 import predict_games_sync
            try:
                nba_v2 = await asyncio.to_thread(
                    predict_games_sync, target_date, [(g.id, g.home_team_name, g.away_team_name) for g in nba_games])
                logger.info(f"NBA v2: {len(nba_v2)}/{len(nba_games)} partidos")
            except Exception as e:
                logger.warning(f"NBA v2 no disponible ({type(e).__name__}: {e}); se usa nba_possessions_v1")
            # Una predicción vigente por partido: se reemplazan las anteriores de NBA
            await self.db.execute(delete(Prediction).where(Prediction.game_id.in_([g.id for g in nba_games])))
            await self.db.flush()
        
        # NFL v2 (EPA + marcador + QB titular, calibrado): si falla, queda el simulador viejo
        nfl_v2 = {}
        nfl_games = [g for g in games if g.sport == SportType.NFL]
        if nfl_games and os.getenv("NFL_MODEL", "v2").lower() != "v1":
            from app.services.nfl_v2 import predict_games_sync as nfl_predict
            try:
                nfl_v2 = await asyncio.to_thread(
                    nfl_predict, [(g.id, g.home_team_name, g.away_team_name, g.game_date, g.external_id) for g in nfl_games])
                logger.info(f"NFL v2: {len(nfl_v2)}/{len(nfl_games)} partidos")
            except Exception as e:
                logger.warning(f"NFL v2 no disponible ({type(e).__name__}: {e}); se usa monte_carlo_v1")
            await self.db.execute(delete(Prediction).where(Prediction.game_id.in_([g.id for g in nfl_games])))
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
            elif game.sport == SportType.NBA and game.id in nba_v2:
                pred_result = nba_v2[game.id]
                model_name = pred_result["simulation_details"]["model"]
                # Medición en vivo: se guarda la predicción previa de v2 y, para comparar, la de v1
                from app.services.nba_tracking import log_prediction
                try:
                    await log_prediction(self.db, game, model_name, pred_result)
                    v1 = self.simulation_engine.simulate(game.sport, features)
                    if v1.get("predicted_spread") is not None and not (v1.get("simulation_details") or {}).get("error"):
                        await log_prediction(self.db, game, "nba_possessions_v1", v1)
                except DBAPIError:
                    raise                       # la transacción ya se deshizo: no seguir a medias
                except Exception as e:
                    logger.warning(f"NBA: no se pudo guardar la predicción en el registro ({type(e).__name__}: {e})")
            elif game.sport == SportType.NFL and game.id in nfl_v2:
                pred_result = nfl_v2[game.id]
                model_name = "nfl_v2"
                from app.services.nba_tracking import log_prediction
                try:
                    await log_prediction(self.db, game, model_name, pred_result)
                except DBAPIError:
                    raise                       # la transacción ya se deshizo: no seguir a medias
                except Exception as e:
                    logger.warning(f"NFL: no se pudo guardar la predicción en el registro ({type(e).__name__}: {e})")
            else:
                # 1. Run Models & Simulations for US Sports
                pred_result = self.simulation_engine.simulate(game.sport, features)
                model_name = "nba_possessions_v1" if game.sport == SportType.NBA else "monte_carlo_v1"
            
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
            if model_name == "nfl_v2":
                picks = self._nfl_v2_picks(prediction, game)
            else:
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

    @staticmethod
    def _nfl_v2_picks(prediction, game) -> List[Dict[str, Any]]:
        """Picks de la NFL: solo spread o total con nivel "recomendado" según las probabilidades
        calibradas (el backtest no encontró ventaja, así que casi nunca habrá). Nunca moneyline."""
        from app.services.nba_tracking import _implied, _market
        from app.services.nfl_v2 import nfl_picks
        mk = _market(game)
        picks = nfl_picks(prediction.home_win_prob, -prediction.predicted_spread, prediction.predicted_total,
                          mk["market_margin"], mk["market_total"], mk["ml_home"], mk["ml_away"],
                          mk["spread_home_odds"], mk["spread_away_odds"], mk["over_odds"], mk["under_odds"],
                          home_name=game.home_team_name, away_name=game.away_team_name)
        out = []
        for key, market in (("spread", MarketType.SPREAD), ("total", MarketType.OVER_UNDER)):
            pk = picks.get(key)
            if not pk or pk.get("tier") != "recomendado":
                continue
            odds = pk["odds"]
            dec = 1 + (odds / 100 if odds > 0 else 100 / -odds)
            out.append({
                "market": market, "pick_type": f"{key}_{pk['side']}", "pick_description": pk["label"],
                "model_probability": pk["prob"], "model_edge": pk["edge"],
                "expected_value": pk["prob"] * dec - 1, "best_sportsbook": "DraftKings", "best_odds": odds,
                "best_odds_decimal": dec, "implied_probability": _implied(odds),
                "value_rating": min(5, max(1, int(pk["edge"] * 100))),
            })
        return out
