import random
import math
from typing import Dict, Any, Tuple
from app.db_models import SportType

class SimulationEngine:
    """
    Motor de Simulaciones Monte Carlo.
    Toma los features extraídos por el FeatureStore y corre miles de simulaciones
    para predecir el score más probable, y las probabilidades de Over/Under.
    """

    def __init__(self, num_simulations: int = 5000):
        self.num_simulations = num_simulations

    def simulate(self, sport: SportType, features: Dict[str, float]) -> Dict[str, Any]:
        """Corre la simulación adecuada según el deporte."""
        if sport == SportType.MLB:
            return self._simulate_mlb(features)
        elif sport == SportType.NBA:
            return self._simulate_nba(features)
        elif sport == SportType.NFL:
            return self._simulate_nfl(features)
        elif sport == SportType.SOCCER:
            return self._simulate_soccer(features)
        else:
            return self._fallback_simulation()

    def _simulate_mlb(self, features: Dict[str, float]) -> Dict[str, Any]:
        # Usar la ERA literal como pidió el usuario (sin regresiones forzadas)
        home_sp_era = features.get("home_pitcher_era", 4.20)
        away_sp_era = features.get("away_pitcher_era", 4.20)
        
        home_bullpen_era = features.get("home_bullpen_era", 4.10)
        away_bullpen_era = features.get("away_bullpen_era", 4.10)
        
        home_runs_scored = features.get("home_runs_scored_per_game", 4.5)
        away_runs_scored = features.get("away_runs_scored_per_game", 4.5)
        home_runs_allowed = features.get("home_runs_allowed_per_game", 4.5)
        away_runs_allowed = features.get("away_runs_allowed_per_game", 4.5)
        
        home_h2h = features.get("home_h2h_runs", 4.5)
        away_h2h = features.get("away_h2h_runs", 4.5)
        
        park_factor = features.get("park_factor", 1.0)
        wind_info = features.get("wind_info", "")
        
        # Modificador de Clima/Viento
        weather_mod = 1.0
        if "Out To" in wind_info and "In From" not in wind_info:
            try:
                mph = int(''.join(filter(str.isdigit, wind_info.split('mph')[0])))
                if mph >= 8: weather_mod += 0.05
            except: pass
        if "In From" in wind_info:
            try:
                mph = int(''.join(filter(str.isdigit, wind_info.split('mph')[0])))
                if mph >= 8: weather_mod -= 0.05
            except: pass
            
        # 55% Starter, 45% Bullpen
        home_pitching_expected = (home_sp_era * 0.55) + (home_bullpen_era * 0.45)
        away_pitching_expected = (away_sp_era * 0.55) + (away_bullpen_era * 0.45)
        
        # Mezcla Ofensiva vs Defensiva Global
        home_offensive_power = (home_runs_scored + away_runs_allowed) / 2.0
        away_offensive_power = (away_runs_scored + home_runs_allowed) / 2.0
        
        # Promedio final con Pitching del partido y H2H
        # Peso: 40% Pitching del día, 40% Ofensiva/Defensiva Global, 20% Historial H2H
        exp_home_runs = (away_pitching_expected * 0.40) + (home_offensive_power * 0.40) + (home_h2h * 0.20)
        exp_away_runs = (home_pitching_expected * 0.40) + (away_offensive_power * 0.40) + (away_h2h * 0.20)
        
        # Multiplicadores Finales
        exp_home_runs = exp_home_runs * park_factor * weather_mod * 1.04 # Home advantage
        exp_away_runs = exp_away_runs * park_factor * weather_mod * 0.96

        home_wins = 0
        away_wins = 0
        total_runs_sim = []
        
        for _ in range(self.num_simulations):
            # Usar Poisson para simular carreras discretas
            home_sim = self._poisson_random(exp_home_runs)
            away_sim = self._poisson_random(exp_away_runs)
            
            # En béisbol no hay empates, si empatan, se va a extra innings (moneda al aire con leve ventaja local)
            while home_sim == away_sim:
                home_sim += self._poisson_random(0.5)
                away_sim += self._poisson_random(0.4)
                
            if home_sim > away_sim:
                home_wins += 1
            else:
                away_wins += 1
            
            total_runs_sim.append(home_sim + away_sim)

        avg_total = sum(total_runs_sim) / self.num_simulations
        
        return {
            "home_win_prob": home_wins / self.num_simulations,
            "away_win_prob": away_wins / self.num_simulations,
            "draw_prob": 0.0,
            "predicted_home_score": exp_home_runs,
            "predicted_away_score": exp_away_runs,
            "predicted_total": avg_total,
            "predicted_spread": exp_away_runs - exp_home_runs, # Negativo = home favorito
            "confidence": 0.75,
            "simulation_details": {
                "num_simulations": self.num_simulations,
                "avg_simulated_total": round(avg_total, 2),
                "home_implied_runs": round(exp_home_runs, 2),
                "away_implied_runs": round(exp_away_runs, 2)
            }
        }

    def _simulate_nba(self, features: Dict[str, float]) -> Dict[str, Any]:
        pace = features.get("expected_pace", 100.0)
        home_elo = features.get("adjusted_home_elo", 1500.0)
        away_elo = features.get("adjusted_away_elo", 1500.0)
        
        elo_diff = home_elo - away_elo + 50.0 # +50 localía
        
        # Convertir ELO diff a Spread (regla general NBA: 25 puntos ELO = 1 punto de spread)
        spread = - (elo_diff / 25.0)
        
        # Puntos base por el ritmo
        base_points = (pace * 1.1)
        exp_home_pts = base_points - (spread / 2)
        exp_away_pts = base_points + (spread / 2)

        home_wins = 0
        away_wins = 0
        total_pts_sim = []

        for _ in range(self.num_simulations):
            # Usar distribución normal para basket
            home_sim = random.gauss(exp_home_pts, 10.0)
            away_sim = random.gauss(exp_away_pts, 10.0)
            
            if home_sim > away_sim:
                home_wins += 1
            elif away_sim > home_sim:
                away_wins += 1
            else:
                home_wins += 1 # OT coinflip
                
            total_pts_sim.append(home_sim + away_sim)

        avg_total = sum(total_pts_sim) / self.num_simulations

        return {
            "home_win_prob": home_wins / self.num_simulations,
            "away_win_prob": away_wins / self.num_simulations,
            "draw_prob": 0.0,
            "predicted_home_score": exp_home_pts,
            "predicted_away_score": exp_away_pts,
            "predicted_total": avg_total,
            "predicted_spread": spread,
            "confidence": 0.80,
            "simulation_details": {
                "num_simulations": self.num_simulations,
                "pace_factor": round(pace, 1),
                "elo_diff": round(elo_diff, 1)
            }
        }

    def _simulate_nfl(self, features: Dict[str, float]) -> Dict[str, Any]:
        wind = features.get("wind_speed", 5.0)
        qb_diff = features.get("qb_diff", 0.0)
        
        base_total = 45.0
        if wind > 15.0:
            base_total -= 4.5 # Mucho viento baja puntos
            
        spread = - (qb_diff / 10.0) - 2.5 # Localía
        
        exp_home_pts = (base_total / 2) - (spread / 2)
        exp_away_pts = (base_total / 2) + (spread / 2)

        home_wins = 0
        away_wins = 0
        
        for _ in range(self.num_simulations):
            home_sim = random.gauss(exp_home_pts, 7.0)
            away_sim = random.gauss(exp_away_pts, 7.0)
            if home_sim > away_sim:
                home_wins += 1
            else:
                away_wins += 1

        return {
            "home_win_prob": home_wins / self.num_simulations,
            "away_win_prob": away_wins / self.num_simulations,
            "draw_prob": 0.0,
            "predicted_home_score": exp_home_pts,
            "predicted_away_score": exp_away_pts,
            "predicted_total": base_total,
            "predicted_spread": spread,
            "confidence": 0.75,
            "simulation_details": {
                "num_simulations": self.num_simulations,
                "wind_penalty_applied": wind > 15.0
            }
        }

    def _simulate_soccer(self, features: Dict[str, float]) -> Dict[str, Any]:
        home_xg = features.get("home_xg_for", 1.3)
        away_xg = features.get("away_xg_for", 1.1)
        home_xga = features.get("home_xga_against", 1.2)
        away_xga = features.get("away_xga_against", 1.2)
        home_adv = features.get("home_advantage_weight", 0.35)
        home_form = features.get("home_form_points", 1.0)
        away_form = features.get("away_form_points", 1.0)
        
        # Adjusted expected goals:
        # Home attack strength * Away defensive weakness + home advantage
        # Normalize by league average (~1.3 goals per game per team)
        league_avg = 1.3
        
        home_attack = home_xg / league_avg  # >1 means better than average
        away_defense = away_xga / league_avg  # >1 means concedes more than average
        away_attack = away_xg / league_avg
        home_defense = home_xga / league_avg
        
        # Form multiplier: scale between 0.85 and 1.15
        home_form_mult = 0.85 + (home_form / 3.0) * 0.30  # form_points max ~3
        away_form_mult = 0.85 + (away_form / 3.0) * 0.30
        
        exp_home_goals = max(0.3, league_avg * home_attack * away_defense * home_form_mult + home_adv)
        exp_away_goals = max(0.3, league_avg * away_attack * home_defense * away_form_mult)
        
        home_wins = 0
        away_wins = 0
        draws = 0
        total_goals_sim = []
        
        for _ in range(self.num_simulations):
            home_sim = self._poisson_random(exp_home_goals)
            away_sim = self._poisson_random(exp_away_goals)
            
            if home_sim > away_sim:
                home_wins += 1
            elif away_sim > home_sim:
                away_wins += 1
            else:
                draws += 1
                
            total_goals_sim.append(home_sim + away_sim)

        avg_total = sum(total_goals_sim) / self.num_simulations

        return {
            "home_win_prob": home_wins / self.num_simulations,
            "away_win_prob": away_wins / self.num_simulations,
            "draw_prob": draws / self.num_simulations,
            "predicted_home_score": round(exp_home_goals, 2),
            "predicted_away_score": round(exp_away_goals, 2),
            "predicted_total": round(avg_total, 2),
            "predicted_spread": round(exp_away_goals - exp_home_goals, 2),
            "confidence": 0.85,
            "simulation_details": {
                "num_simulations": self.num_simulations,
                "most_frequent_score": f"{round(exp_home_goals)} - {round(exp_away_goals)}",
                "home_xg_simulated": round(exp_home_goals, 2),
                "away_xg_simulated": round(exp_away_goals, 2),
                "avg_simulated_total": round(avg_total, 2),
            }
        }

    def _fallback_simulation(self):
        return {
            "home_win_prob": 0.5,
            "away_win_prob": 0.5,
            "draw_prob": 0.0,
            "predicted_home_score": 100,
            "predicted_away_score": 100,
            "predicted_total": 200,
            "predicted_spread": 0,
            "confidence": 0.5,
            "simulation_details": {"error": "unknown sport"}
        }

    def _poisson_random(self, lam: float) -> int:
        """Genera un número aleatorio de una distribución Poisson."""
        L = math.exp(-lam)
        p = 1.0
        k = 0
        while p > L:
            k += 1
            p *= random.random()
        return k - 1
