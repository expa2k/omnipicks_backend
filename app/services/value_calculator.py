"""
OmniPicks — ML: Value Calculator Stub
"""
from typing import List, Dict, Any
from app.db_models import Prediction, OddsSnapshot, MarketType
import math

class ValueCalculator:
    def _poisson_prob(self, lam: float, k: int) -> float:
        """Probability of exactly k events given expected lam."""
        return (math.exp(-lam) * (lam ** k)) / math.factorial(k)
        
    def _calc_under_prob(self, expected_total: float, line: float) -> float:
        """Calculate probability of total goals/runs being strictly UNDER the line."""
        prob = 0.0
        for k in range(int(math.ceil(line))):
            prob += self._poisson_prob(expected_total, k)
        return prob
        
    def _calc_btts_prob(self, expected_home: float, expected_away: float) -> float:
        """BTTS Yes means both teams score at least 1 goal."""
        prob_home_zero = self._poisson_prob(expected_home, 0)
        prob_away_zero = self._poisson_prob(expected_away, 0)
        return (1.0 - prob_home_zero) * (1.0 - prob_away_zero)

    def _american_to_decimal(self, american_odds: float) -> float:
        if american_odds > 0:
            return (american_odds / 100) + 1
        elif american_odds < 0:
            return (100 / abs(american_odds)) + 1
        return 1.0

    def _american_to_implied_prob(self, american_odds: float) -> float:
        if american_odds > 0:
            return 100 / (american_odds + 100)
        elif american_odds < 0:
            return abs(american_odds) / (abs(american_odds) + 100)
        return 0.0

    def calculate_ev(self, model_prob: float, decimal_odds: float) -> float:
        return (model_prob * decimal_odds) - 1
        
    def _calculate_kelly(self, prob: float, decimal_odds: float) -> float:
        b = decimal_odds - 1
        q = 1 - prob
        kelly = (b * prob - q) / b
        return max(0.0, kelly)
    
    def find_value_picks(self, prediction: Prediction, odds_list: List[OddsSnapshot], game) -> List[Dict[str, Any]]:
        picks = []
        
        best_picks = {}
        
        for odds in odds_list:
            if odds.market == MarketType.MONEYLINE:
                ev_home = 0
                ev_away = 0
                home_pick = None
                away_pick = None
                
                # Calculate Home EV
                if odds.home_odds:
                    dec_odds = self._american_to_decimal(odds.home_odds)
                    imp_prob = self._american_to_implied_prob(odds.home_odds)
                    ev_home = self.calculate_ev(prediction.home_win_prob, dec_odds)
                    
                    if ev_home > 0.02:
                        edge = prediction.home_win_prob - imp_prob
                        kelly = self._calculate_kelly(prediction.home_win_prob, dec_odds)
                        home_pick = {
                            "market": MarketType.MONEYLINE,
                            "pick_type": "home_ml",
                            "pick_description": f"{game.home_team_name} ML",
                            "model_probability": prediction.home_win_prob,
                            "model_edge": edge,
                            "expected_value": ev_home,
                            "best_sportsbook": odds.sportsbook,
                            "best_odds": odds.home_odds,
                            "best_odds_decimal": dec_odds,
                            "implied_probability": imp_prob,
                            "kelly_fraction": kelly,
                            "recommended_stake_pct": min(kelly * 0.5 * 100, 5.0),
                            "value_rating": min(5, max(1, int(ev_home * 100)))
                        }
                        
                # Calculate Away EV
                if odds.away_odds:
                    dec_odds = self._american_to_decimal(odds.away_odds)
                    imp_prob = self._american_to_implied_prob(odds.away_odds)
                    ev_away = self.calculate_ev(prediction.away_win_prob, dec_odds)
                    
                    if ev_away > 0.02:
                        edge = prediction.away_win_prob - imp_prob
                        kelly = self._calculate_kelly(prediction.away_win_prob, dec_odds)
                        away_pick = {
                            "market": MarketType.MONEYLINE,
                            "pick_type": "away_ml",
                            "pick_description": f"{game.away_team_name} ML",
                            "model_probability": prediction.away_win_prob,
                            "model_edge": edge,
                            "expected_value": ev_away,
                            "best_sportsbook": odds.sportsbook,
                            "best_odds": odds.away_odds,
                            "best_odds_decimal": dec_odds,
                            "implied_probability": imp_prob,
                            "kelly_fraction": kelly,
                            "recommended_stake_pct": min(kelly * 0.5 * 100, 5.0),
                            "value_rating": min(5, max(1, int(ev_away * 100)))
                        }
                        
                # Determine the best pick for this specific sportsbook
                best_sp_pick = None
                if home_pick and away_pick:
                    best_sp_pick = home_pick if ev_home >= ev_away else away_pick
                elif home_pick:
                    best_sp_pick = home_pick
                elif away_pick:
                    best_sp_pick = away_pick
                    
                if best_sp_pick:
                    key = best_sp_pick["market"]
                    if key not in best_picks or best_sp_pick["expected_value"] > best_picks[key]["expected_value"]:
                        best_picks[key] = best_sp_pick
                        
            elif odds.market == MarketType.OVER_UNDER and odds.total_line and prediction.predicted_total:
                under_prob = self._calc_under_prob(prediction.predicted_total, odds.total_line)
                over_prob = 1.0 - under_prob
                
                ev_over = 0
                ev_under = 0
                over_pick = None
                under_pick = None
                
                if odds.over_odds:
                    dec_odds = self._american_to_decimal(odds.over_odds)
                    imp_prob = self._american_to_implied_prob(odds.over_odds)
                    ev_over = self.calculate_ev(over_prob, dec_odds)
                    
                    if ev_over > 0.02:
                        edge = over_prob - imp_prob
                        kelly = self._calculate_kelly(over_prob, dec_odds)
                        over_pick = {
                            "market": MarketType.OVER_UNDER,
                            "pick_type": "over",
                            "pick_description": f"Over {odds.total_line}",
                            "model_probability": over_prob,
                            "model_edge": edge,
                            "expected_value": ev_over,
                            "best_sportsbook": odds.sportsbook,
                            "best_odds": odds.over_odds,
                            "best_odds_decimal": dec_odds,
                            "implied_probability": imp_prob,
                            "kelly_fraction": kelly,
                            "recommended_stake_pct": min(kelly * 0.5 * 100, 5.0),
                            "value_rating": min(5, max(1, int(ev_over * 100)))
                        }
                        
                if odds.under_odds:
                    dec_odds = self._american_to_decimal(odds.under_odds)
                    imp_prob = self._american_to_implied_prob(odds.under_odds)
                    ev_under = self.calculate_ev(under_prob, dec_odds)
                    
                    if ev_under > 0.02:
                        edge = under_prob - imp_prob
                        kelly = self._calculate_kelly(under_prob, dec_odds)
                        under_pick = {
                            "market": MarketType.OVER_UNDER,
                            "pick_type": "under",
                            "pick_description": f"Under {odds.total_line}",
                            "model_probability": under_prob,
                            "model_edge": edge,
                            "expected_value": ev_under,
                            "best_sportsbook": odds.sportsbook,
                            "best_odds": odds.under_odds,
                            "best_odds_decimal": dec_odds,
                            "implied_probability": imp_prob,
                            "kelly_fraction": kelly,
                            "recommended_stake_pct": min(kelly * 0.5 * 100, 5.0),
                            "value_rating": min(5, max(1, int(ev_under * 100)))
                        }
                        
                best_ou_pick = over_pick if (over_pick and (not under_pick or ev_over >= ev_under)) else under_pick
                if best_ou_pick:
                    key = best_ou_pick["market"]
                    if key not in best_picks or best_ou_pick["expected_value"] > best_picks[key]["expected_value"]:
                        best_picks[key] = best_ou_pick

            elif odds.market == MarketType.BTTS and prediction.predicted_home_score is not None and prediction.predicted_away_score is not None:
                btts_yes_prob = self._calc_btts_prob(prediction.predicted_home_score, prediction.predicted_away_score)
                btts_no_prob = 1.0 - btts_yes_prob
                
                ev_yes = 0
                ev_no = 0
                yes_pick = None
                no_pick = None
                
                if odds.over_odds:  # Mapped YES to over_odds in data_ingestion
                    dec_odds = self._american_to_decimal(odds.over_odds)
                    imp_prob = self._american_to_implied_prob(odds.over_odds)
                    ev_yes = self.calculate_ev(btts_yes_prob, dec_odds)
                    
                    if ev_yes > 0.02:
                        edge = btts_yes_prob - imp_prob
                        kelly = self._calculate_kelly(btts_yes_prob, dec_odds)
                        yes_pick = {
                            "market": MarketType.BTTS,
                            "pick_type": "btts_yes",
                            "pick_description": "BTTS - Yes",
                            "model_probability": btts_yes_prob,
                            "model_edge": edge,
                            "expected_value": ev_yes,
                            "best_sportsbook": odds.sportsbook,
                            "best_odds": odds.over_odds,
                            "best_odds_decimal": dec_odds,
                            "implied_probability": imp_prob,
                            "kelly_fraction": kelly,
                            "recommended_stake_pct": min(kelly * 0.5 * 100, 5.0),
                            "value_rating": min(5, max(1, int(ev_yes * 100)))
                        }
                        
                if odds.under_odds: # Mapped NO to under_odds in data_ingestion
                    dec_odds = self._american_to_decimal(odds.under_odds)
                    imp_prob = self._american_to_implied_prob(odds.under_odds)
                    ev_no = self.calculate_ev(btts_no_prob, dec_odds)
                    
                    if ev_no > 0.02:
                        edge = btts_no_prob - imp_prob
                        kelly = self._calculate_kelly(btts_no_prob, dec_odds)
                        no_pick = {
                            "market": MarketType.BTTS,
                            "pick_type": "btts_no",
                            "pick_description": "BTTS - No",
                            "model_probability": btts_no_prob,
                            "model_edge": edge,
                            "expected_value": ev_no,
                            "best_sportsbook": odds.sportsbook,
                            "best_odds": odds.under_odds,
                            "best_odds_decimal": dec_odds,
                            "implied_probability": imp_prob,
                            "kelly_fraction": kelly,
                            "recommended_stake_pct": min(kelly * 0.5 * 100, 5.0),
                            "value_rating": min(5, max(1, int(ev_no * 100)))
                        }
                        
                best_btts_pick = yes_pick if (yes_pick and (not no_pick or ev_yes >= ev_no)) else no_pick
                if best_btts_pick:
                    key = best_btts_pick["market"]
                    if key not in best_picks or best_btts_pick["expected_value"] > best_picks[key]["expected_value"]:
                        best_picks[key] = best_btts_pick
                        
        picks.extend(best_picks.values())
                        
        return picks
