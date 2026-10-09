"""
OmniPicks — ML: Advanced Feature Engineering
Calculates sport-specific features (injuries, weather, pitchers, pace, xG, etc.) 
to feed into the XGBoost and Poisson models.
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from app.services.apisports_client import APISportsClient

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Transforms raw API data into structured numerical vectors for Machine Learning models.
    """

    def __init__(self):
        # En una implementación final, aquí se inicializarían clientes de cache (Redis)
        # para buscar historiales rápidos de xG, ERA de pitchers, etc.
        self.apisports = APISportsClient()

    async def compute_features(self, game_data: Dict[str, Any], sport: str, game_date_str: str = None) -> Dict[str, float]:
        """
        Router principal. Toma el JSON en crudo del juego y redirige a la lógica por deporte.
        """
        sport_lower = sport.lower()
        if sport_lower == "mlb":
            return self._compute_mlb_features(game_data, game_date_str)
        elif sport_lower == "nba":
            return self._compute_nba_features(game_data)
        elif sport_lower == "nfl":
            return await self._compute_nfl_features(game_data)
        elif sport_lower in ["soccer", "fútbol", "futbol"]:
            return await self._compute_soccer_features(game_data)
        else:
            return self._compute_generic_features(game_data)

    def _compute_mlb_features(self, game: Dict[str, Any], game_date_str: str = None) -> Dict[str, Any]:
        """
        Béisbol: Factores Avanzados de 5 Niveles (Generales, H2H, Racha, Umpire, FIP).
        """
        from app.services.mlb_advanced_metrics import extract_advanced_mlb_features
        features = extract_advanced_mlb_features(game, game_date_str)
        
        # Mantenemos las variables base (Elo y Bullpen Fatigue simularán métricas base mientras conectamos las de Fangraphs)
        import random
        random.seed(game.get("id", 1))
        features["home_bullpen_fatigue"] = round(random.uniform(80.0, 130.0), 1)
        features["away_bullpen_fatigue"] = round(random.uniform(80.0, 130.0), 1)
        features["home_elo"] = round(random.uniform(1450, 1600), 1)
        features["away_elo"] = round(random.uniform(1450, 1600), 1)
        random.seed()
        
        park_name = game.get("venue", {}).get("name", "Estadio Local")
        features["park_factor"] = self._get_park_factor(park_name)
        features["estadio_confirmado"] = park_name if park_name else "Por Confirmar"
        
        return features

    def _compute_nba_features(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """
        Básquetbol: Ritmo, Ratings (Off/Def) y 4 Factores.
        """
        from app.services.nba_advanced_metrics import extract_advanced_nba_features
        # Handle date dictionary format from NBA API
        date_data = game.get("date")
        if isinstance(date_data, dict):
            game_date_str = date_data.get("start", "").split("T")[0] if date_data.get("start") else ""
        else:
            game_date_str = str(date_data).split("T")[0] if date_data else ""
        
        features = extract_advanced_nba_features(game, game_date_str)
        
        # Mantenemos las variables base requeridas
        import random
        random.seed(game.get("id", 1))
        features["home_b2b"] = random.choice([0.0, 1.0])
        features["away_b2b"] = random.choice([0.0, 1.0])
        features["lesiones_reportadas"] = "Reporte pendiente"
        features["home_injury_penalty"] = 0.0
        features["away_injury_penalty"] = 0.0
        features["adjusted_home_elo"] = round(random.uniform(1450.0, 1600.0), 1)
        features["adjusted_away_elo"] = round(random.uniform(1450.0, 1600.0), 1)
        random.seed()
        
        return features

    async def _compute_nfl_features(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fútbol Americano: EPA/Play, Success Rate, Puntos, Yardas y Turnovers.
        Datos reales de la API-Sports (American Football API).
        """
        from app.services.nfl_advanced_metrics import extract_advanced_nfl_features
        game_date_str = game.get("date", "").split("T")[0]
        
        # Pass the shared APISports client so the function can fetch team stats
        features = await extract_advanced_nfl_features(game, game_date_str, apisports_client=self.apisports)
        
        # Mantenemos las variables base del simulador
        import random
        random.seed(game.get("id", 1))
        features["home_rest_advantage"] = 0
        features["home_qb_injury"] = 0.0
        features["away_qb_injury"] = 0.0
        # Elo derivado del EPA (si disponible), si no, valor aleatorio estable
        home_off_epa = features.get("home_off_epa", 0.0)
        away_off_epa = features.get("away_off_epa", 0.0)
        features["home_elo"] = round(1500 + home_off_epa * 500, 1)
        features["away_elo"] = round(1500 + away_off_epa * 500, 1)
        random.seed()
        
        return features

    async def _compute_soccer_features(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fútbol: xG, PPDA, Forma, H2H y Ventaja de Localía — datos REALES de API-Football.
        """
        from app.services.soccer_advanced_metrics import extract_advanced_soccer_features
        game_date_str = game.get("date", "").split("T")[0]
        
        # Now async — fetches real data from API-Football
        features = await extract_advanced_soccer_features(game, game_date_str)
        
        # Derive Elo from win percentage instead of random
        home_win_pct = features.get("home_win_pct", 33.3)
        away_win_pct = features.get("away_win_pct", 33.3)
        # Win% to Elo: 50% = 1500, each 10% = ~100 Elo points
        features["home_elo"] = round(1500 + (home_win_pct - 50) * 10, 1)
        features["away_elo"] = round(1500 + (away_win_pct - 50) * 10, 1)
        
        league = features.get("competicion", "Liga")
        base_hfa = 0.35 
        if "Liga MX" in league or "Premier League" in league:
            base_hfa = 0.45
            
        features["home_advantage_weight"] = base_hfa
        
        return features

    def _compute_generic_features(self, game: Dict[str, Any]) -> Dict[str, float]:
        """Fallback genérico si no reconocemos el deporte."""
        return {
            "home_elo": float(game.get("home_elo", 1500.0)),
            "away_elo": float(game.get("away_elo", 1500.0)),
            "is_neutral_site": 1.0 if game.get("is_neutral") else 0.0
        }

    # --- Funciones Auxiliares Privadas ---
    
    def _get_park_factor(self, stadium_name: str) -> float:
        """Retorna el multiplicador ofensivo del estadio de MLB."""
        stadiums = {
            "Coors Field": 1.350,       # Paraíso de bateadores
            "Great American Ball Park": 1.120,
            "Petco Park": 0.850,        # Paraíso de pitchers
            "T-Mobile Park": 0.880
        }
        return stadiums.get(stadium_name, 1.000)
        
    def _get_player_value(self, player_data: Dict[str, Any]) -> float:
        """Estima cuántos puntos de ELO vale un jugador (NBA)."""
        # En producción esto usaría el WAR o el VORP del jugador.
        # Por ejemplo, Jokic vale ~120 puntos de ELO.
        is_star = player_data.get("is_star", False)
        is_starter = player_data.get("is_starter", False)
        
        if is_star: return 85.0
        if is_starter: return 35.0
        return 10.0
