"""
OmniPicks — NFL Advanced Metrics (ESPN-powered)
Métricas avanzadas de NFL usando ESPN Core API.
- Temporada actual (2026) si ya hay partidos jugados
- Fallback automático a temporada anterior (2025) si la actual está vacía
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def _default_features() -> Dict[str, Any]:
    """Safe defaults cuando no hay datos disponibles."""
    return {
        "home_off_epa":                0.0,
        "away_off_epa":                0.0,
        "home_def_epa":                0.0,
        "away_def_epa":                0.0,
        "home_success_rate":           0.40,
        "away_success_rate":           0.40,
        "home_form_epa":               0.0,
        "away_form_epa":               0.0,
        "home_points_per_game":        22.0,
        "away_points_per_game":        22.0,
        "home_points_allowed_per_game":22.0,
        "away_points_allowed_per_game":22.0,
        "home_yards_per_game":         340.0,
        "away_yards_per_game":         340.0,
        "home_turnovers_per_game":     1.5,
        "away_turnovers_per_game":     1.5,
        "home_completion_pct":         64.0,
        "away_completion_pct":         64.0,
        "home_sacks_per_game":         2.5,
        "away_sacks_per_game":         2.5,
        "lesiones_reportadas":         "Reporte Oficial pendiente (Viernes)",
    }


def _espn_stats_to_features(stats: Dict[str, float], prefix: str) -> Dict[str, Any]:
    """
    Convierte el dict plano de ESPN stats (categoria_nombre: valor)
    a las features que usa el algoritmo de pronósticos.
    """
    features = {}
    games = max(stats.get("general_gamesPlayed", 1), 1)

    # ── Puntos ────────────────────────────────────────────────────────────────
    pts_for    = stats.get("scoring_totalPoints",    stats.get("scoring_points", 0))
    pts_against = stats.get("scoring_pointsAgainst", 0)
    ppg = pts_for / games
    papg = pts_against / games
    features[f"{prefix}_points_per_game"]        = round(ppg, 2)
    features[f"{prefix}_points_allowed_per_game"] = round(papg, 2)

    # ── Yardas ────────────────────────────────────────────────────────────────
    pass_yards   = stats.get("passing_netPassingYards",  stats.get("passing_passingYards", 0))
    rush_yards   = stats.get("rushing_rushingYards",     stats.get("rushing_netTotalYards", 0))
    total_yards  = pass_yards + rush_yards
    features[f"{prefix}_yards_per_game"]    = round(total_yards / games, 1)
    features[f"{prefix}_pass_yards_per_game"] = round(pass_yards / games, 1)
    features[f"{prefix}_rush_yards_per_game"] = round(rush_yards / games, 1)

    # ── Turnover ─────────────────────────────────────────────────────────────
    fumbles_lost = stats.get("general_fumblesLost", 0)
    # interceptions están en defensiveInterceptions cuando es el equipo que interceptó
    # Para ofensiva: passingInterceptions
    int_thrown   = stats.get("passing_interceptions", 0)
    turnovers    = fumbles_lost + int_thrown
    features[f"{prefix}_turnovers_per_game"] = round(turnovers / games, 2)

    # ── Passing efficiency ────────────────────────────────────────────────────
    comp_pct = stats.get("passing_completionPct", 64.0)
    features[f"{prefix}_completion_pct"] = round(comp_pct, 1)

    # ── Sacks (defensa) ───────────────────────────────────────────────────────
    sacks = stats.get("defensive_sacks", stats.get("defensive_totalSacks", 0))
    features[f"{prefix}_sacks_per_game"] = round(sacks / games, 2)

    # ── EPA proxy — derivado del diferencial de puntos normalizado ─────────────
    # NFL average ~22 ppg. Cada punto de diferencia ≈ 0.007 EPA/play (basado en nflverse)
    pt_diff = ppg - papg
    features[f"{prefix}_off_epa"]  = round(pt_diff / 7.0, 4)
    features[f"{prefix}_def_epa"]  = round(-pt_diff / 7.0, 4)
    features[f"{prefix}_form_epa"] = features[f"{prefix}_off_epa"]

    # ── Success rate proxy — yards per game normalizado ────────────────────────
    # 340 yds/game ≈ 0.40 success rate (NFL baseline)
    ypg = features[f"{prefix}_yards_per_game"]
    features[f"{prefix}_success_rate"] = round(min(max(ypg / 850.0, 0.25), 0.65), 4)

    return features


async def extract_advanced_nfl_features(
    game_data: dict,
    game_date_str: str,
    apisports_client=None,   # Kept for compatibility, not used
) -> Dict[str, Any]:
    """
    Calcula métricas avanzadas de NFL usando ESPN Core API.

    Flujo:
    1. Obtener ESPN team IDs desde los nombres de los equipos
    2. Llamar ESPN stats — temporada actual primero, fallback a anterior
    3. Derivar EPA / Success Rate / Puntos / Yardas / Turnovers
    4. Si ESPN falla → defaults seguros (no rompe el pipeline)
    """
    from app.services.espn_nfl_client import get_nfl_team_stats_espn, get_espn_nfl_id

    features = _default_features()

    home_data = game_data.get("teams", {}).get("home", {})
    away_data = game_data.get("teams", {}).get("away", {})
    home_name = home_data.get("name", "")
    away_name = away_data.get("name", "")

    # ESPN usa su propio sistema de IDs — mapear por nombre
    home_espn_id = int(home_data.get("id", 0)) or get_espn_nfl_id(home_name)
    away_espn_id = int(away_data.get("id", 0)) or get_espn_nfl_id(away_name)

    if not home_espn_id or not away_espn_id:
        logger.warning(f"NFL metrics: no ESPN IDs found for {home_name} vs {away_name} — using defaults")
        return features

    try:
        home_stats = await get_nfl_team_stats_espn(home_espn_id)
        away_stats = await get_nfl_team_stats_espn(away_espn_id)

        if home_stats:
            features.update(_espn_stats_to_features(home_stats, "home"))
            logger.info(
                f"NFL ESPN: {home_name} off_epa={features['home_off_epa']:.3f} "
                f"ppg={features['home_points_per_game']:.1f} "
                f"ypg={features['home_yards_per_game']:.0f}"
            )
        else:
            logger.warning(f"NFL ESPN: no stats for {home_name}")

        if away_stats:
            features.update(_espn_stats_to_features(away_stats, "away"))
            logger.info(
                f"NFL ESPN: {away_name} off_epa={features['away_off_epa']:.3f} "
                f"ppg={features['away_points_per_game']:.1f} "
                f"ypg={features['away_yards_per_game']:.0f}"
            )
        else:
            logger.warning(f"NFL ESPN: no stats for {away_name}")

    except Exception as e:
        logger.error(f"NFL ESPN stats error for {home_name} vs {away_name}: {e} — using defaults")

    return features
