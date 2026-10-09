import datetime
import logging
from nba_api.stats.endpoints import teamestimatedmetrics
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams

logger = logging.getLogger(__name__)

def get_nba_team_id(team_name: str):
    """Obtiene el ID de nba_api dado el nombre del equipo."""
    nba_teams = teams.get_teams()
    for t in nba_teams:
        if t['full_name'] == team_name or t['nickname'] == team_name or t['city'] == team_name:
            return t['id']
    return None

def extract_advanced_nba_features(game_data: dict, game_date_str: str) -> dict:
    """Extrae factores avanzados para la NBA usando Nivel 1 a 5."""
    features = {}
    
    home_name = game_data.get("teams", {}).get("home", {}).get("name", "Local")
    away_name = game_data.get("teams", {}).get("away", {}).get("name", "Visitante")
    
    home_id = get_nba_team_id(home_name)
    away_id = get_nba_team_id(away_name)
    
    # ---------------------------------------------------------
    # LEVEL 1 & 5: RATINGS Y 4 FACTORS
    # ---------------------------------------------------------
    # Default values
    for prefix in ["home", "away"]:
        features[f"{prefix}_off_rating"] = 110.0
        features[f"{prefix}_def_rating"] = 110.0
        features[f"{prefix}_net_rating"] = 0.0
        features[f"{prefix}_pace"] = 100.0
        features[f"{prefix}_efg"] = 0.500
        features[f"{prefix}_tov_pct"] = 0.150
        features[f"{prefix}_orb_pct"] = 0.250
        features[f"{prefix}_ft_rate"] = 0.200
        features[f"{prefix}_form_net_rating"] = 0.0
    
    try:
        # Extraer Rating Ofensivo, Defensivo y Pace
        metrics = teamestimatedmetrics.TeamEstimatedMetrics(season='2025-26')
        df_metrics = metrics.get_data_frames()[0]
        
        # Extraer los 4 factores
        factors = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Four Factors', season='2025-26')
        df_factors = factors.get_data_frames()[0]
        
        for t_id, prefix in [(home_id, "home"), (away_id, "away")]:
            if not t_id: continue
            
            # Ratings
            t_metrics = df_metrics[df_metrics['TEAM_ID'] == t_id]
            if not t_metrics.empty:
                features[f"{prefix}_off_rating"] = float(t_metrics['E_OFF_RATING'].values[0])
                features[f"{prefix}_def_rating"] = float(t_metrics['E_DEF_RATING'].values[0])
                features[f"{prefix}_net_rating"] = float(t_metrics['E_NET_RATING'].values[0])
                features[f"{prefix}_pace"] = float(t_metrics['E_PACE'].values[0])
                
            # 4 Factors
            t_factors = df_factors[df_factors['TEAM_ID'] == t_id]
            if not t_factors.empty:
                features[f"{prefix}_efg"] = float(t_factors['EFG_PCT'].values[0])
                features[f"{prefix}_tov_pct"] = float(t_factors['TM_TOV_PCT'].values[0])
                features[f"{prefix}_orb_pct"] = float(t_factors['OREB_PCT'].values[0])
                features[f"{prefix}_ft_rate"] = float(t_factors['FTA_RATE'].values[0])
                
            # Form (Last 5 Games)
            gamelog = teamgamelog.TeamGameLog(team_id=t_id, season='2025-26')
            df_log = gamelog.get_data_frames()[0]
            if not df_log.empty:
                # Top 5 rows are most recent
                last_5 = df_log.head(5)
                # Calculate simple form diff (PTS - PTS_OPP is not directly in standard GameLog without more joining, so we use W/L% for now)
                wins = sum([1 for r in last_5['WL'] if r == 'W'])
                features[f"{prefix}_form_net_rating"] = (wins / len(last_5)) * 10.0  # Proxy for now
                
    except Exception as e:
        logger.error(f"Error extrayendo stats de NBA API: {e}")
        
    return features
