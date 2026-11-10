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

    # Parámetros del modelo NBA
    NBA_HOME_COURT = 2.5     # puntos de ventaja de local (margen)
    NBA_SHRINK_GAMES = 10    # "partidos virtuales" de promedio de liga para suavizar inicio de temporada
    NBA_B2B_PENALTY = 1.0    # puntos que pierde un equipo en back-to-back (y +0.5 al rival)
    NBA_MARGIN_SD = 12.5     # desviación típica histórica del margen final
    NBA_TOTAL_SD = 17.0      # desviación típica histórica del total de puntos

    def _simulate_nba(self, features: Dict[str, float]) -> Dict[str, Any]:
        if features.get("api_error") or "home_off_rating" not in features:
            result = self._fallback_simulation()
            result["simulation_details"] = {"error": features.get("api_error", "Sin estadísticas de NBA")}
            return result

        league_off = features.get("league_off_rating", 114.0)
        league_pace = features.get("league_pace", 99.0)
        k = self.NBA_SHRINK_GAMES

        def prior(prefix: str, key: str, mean: float) -> float:
            # Prior = temporada pasada regresada 1/3 hacia la media (cambios de plantilla);
            # si no hay temporada pasada, el promedio de liga.
            prev = features.get(f"{prefix}_prev_{key}")
            return mean if prev is None else (2 * prev + mean) / 3

        def shrink(prefix: str, key: str, gp: int, mean: float) -> float:
            # Con pocos partidos pesa más el prior; con muchos, la temporada actual
            value = features[f"{prefix}_{key}"]
            return (value * gp + prior(prefix, key, mean) * k) / (gp + k)

        h_gp, a_gp = int(features.get("home_gp", 0)), int(features.get("away_gp", 0))
        h_off = shrink("home", "off_rating", h_gp, league_off)
        h_def = shrink("home", "def_rating", h_gp, league_off)
        a_off = shrink("away", "off_rating", a_gp, league_off)
        a_def = shrink("away", "def_rating", a_gp, league_off)
        h_pace = shrink("home", "pace", h_gp, league_pace)
        a_pace = shrink("away", "pace", a_gp, league_pace)

        # Posesiones esperadas y eficiencia cruzada (ataque propio vs defensa rival)
        possessions = h_pace * a_pace / league_pace
        exp_home_pts = possessions * (h_off * a_def / league_off) / 100.0
        exp_away_pts = possessions * (a_off * h_def / league_off) / 100.0

        exp_home_pts += self.NBA_HOME_COURT / 2
        exp_away_pts -= self.NBA_HOME_COURT / 2

        if features.get("home_b2b"):
            exp_home_pts -= self.NBA_B2B_PENALTY
            exp_away_pts += self.NBA_B2B_PENALTY / 2
        if features.get("away_b2b"):
            exp_away_pts -= self.NBA_B2B_PENALTY
            exp_home_pts += self.NBA_B2B_PENALTY / 2

        exp_margin = exp_home_pts - exp_away_pts
        exp_total = exp_home_pts + exp_away_pts

        home_wins = 0
        totals = []
        for _ in range(self.num_simulations):
            margin = random.gauss(exp_margin, self.NBA_MARGIN_SD)
            while abs(margin) < 0.5:          # no hay empates: el tiempo extra decide
                margin = random.gauss(exp_margin, self.NBA_MARGIN_SD)
            if margin > 0:
                home_wins += 1
            totals.append(random.gauss(exp_total, self.NBA_TOTAL_SD))

        n = self.num_simulations
        min_gp = min(h_gp, a_gp)
        confidence = round(0.55 + 0.30 * min(1.0, min_gp / 30.0), 2)

        return {
            "home_win_prob": home_wins / n,
            "away_win_prob": 1 - home_wins / n,
            "draw_prob": 0.0,
            "predicted_home_score": round(exp_home_pts, 1),
            "predicted_away_score": round(exp_away_pts, 1),
            "predicted_total": round(sum(totals) / n, 1),
            "predicted_spread": round(-exp_margin, 1),   # negativo = local favorito
            "confidence": confidence,
            "simulation_details": {
                "source": "Monte Carlo NBA: posesiones × eficiencia (nba_api)",
                "num_simulations": n,
                "expected_possessions": round(possessions, 1),
                "home_off_adj": round(h_off, 1), "home_def_adj": round(h_def, 1),
                "away_off_adj": round(a_off, 1), "away_def_adj": round(a_def, 1),
                "home_court_pts": self.NBA_HOME_COURT,
                "home_b2b": bool(features.get("home_b2b")),
                "away_b2b": bool(features.get("away_b2b")),
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
