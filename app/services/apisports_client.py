"""
OmniPicks — API Clients: API-Sports (NBA, NFL, MLB)
"""
import httpx
import logging
from typing import Dict, Any, List, Optional
from app.config import get_settings
from app.utils.constants import API_SPORTS_ENDPOINTS
from app.db_models import SportType

logger = logging.getLogger(__name__)

class APISportsClient:
    """Client for API-Sports (Basketball, American Football, Baseball)."""
    
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.api_sports_key
        self.headers = {
            "x-apisports-key": self.api_key,
            "Accept": "application/json"
        }
    
    async def _request(self, sport: SportType, endpoint: str, params: Optional[Dict] = None) -> List[Dict]:
        """Generic request handler mapping to the correct sport subdomain."""
        if not self.api_key:
            logger.warning("API-Sports key not configured")
            return []
            
        base_url = API_SPORTS_ENDPOINTS.get(sport)
        if not base_url:
            logger.error(f"No API-Sports endpoint configured for {sport}")
            return []

        url = f"{base_url}{endpoint}"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.request("GET", url, headers=self.headers, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                
                # Log API limit usage from headers
                remaining = response.headers.get("x-ratelimit-requests-remaining")
                if remaining:
                    logger.debug(f"API-Sports {sport} daily requests remaining: {remaining}")
                    
                return data.get("response", [])
            except httpx.HTTPError as e:
                logger.error(f"API-Sports error: {e}")
                return []

    # === NBA ===
    async def get_nba_games(self, date_str: str) -> List[Dict]:
        """Get NBA games for a specific date (YYYY-MM-DD)."""
        return await self._request(SportType.NBA, "/games", params={"date": date_str})

    async def get_nba_team_stats(self, team_id: int, season: str) -> List[Dict]:
        """Get NBA team stats for a season."""
        return await self._request(SportType.NBA, "/statistics", params={"team": team_id, "season": season})

    # === NFL ===
    async def get_nfl_games(self, date_str: str) -> List[Dict]:
        """Get NFL games for a specific date."""
        return await self._request(SportType.NFL, "/games", params={"date": date_str})

    async def get_nfl_team_stats(self, team_id: int, season: str) -> List[Dict]:
        """Get NFL team stats."""
        return await self._request(SportType.NFL, "/teams/statistics", params={"team": team_id, "season": season})

    # === MLB ===
    async def get_mlb_games(self, date_str: str) -> List[Dict]:
        """Get MLB games for a specific date."""
        return await self._request(SportType.MLB, "/games", params={"date": date_str})

    async def get_mlb_team_stats(self, team_id: int, season: str) -> List[Dict]:
        """Get MLB team stats."""
        return await self._request(SportType.MLB, "/teams/statistics", params={"team": team_id, "season": season})

    # === Soccer ===
    async def get_soccer_games(self, date_str: str) -> List[Dict]:
        """Get soccer games (fixtures) for a specific date."""
        return await self._request(SportType.SOCCER, "/fixtures", params={"date": date_str})

    async def get_soccer_team_stats(self, team_id: int, season: int, league_id: int) -> Dict:
        """Get soccer team statistics for a specific season and league."""
        results = await self._request(
            SportType.SOCCER, "/teams/statistics",
            params={"team": team_id, "season": season, "league": league_id}
        )
        # This endpoint returns a single object, not a list
        return results if isinstance(results, dict) else (results[0] if results else {})

    async def get_soccer_h2h(self, team_a_id: int, team_b_id: int, last: int = 10) -> List[Dict]:
        """Get head-to-head history between two soccer teams."""
        return await self._request(
            SportType.SOCCER, "/fixtures/headtohead",
            params={"h2h": f"{team_a_id}-{team_b_id}", "last": last}
        )

    async def get_soccer_standings(self, league_id: int, season: int) -> List[Dict]:
        """Get standings for a league to derive form and position."""
        return await self._request(
            SportType.SOCCER, "/standings",
            params={"league": league_id, "season": season}
        )

    async def get_fifa_rankings(self) -> List[Dict]:
        """Get official FIFA rankings to determine Team Tiers."""
        return await self._request(
            SportType.SOCCER, "/rankings/fifa"
        )

    async def get_team_form(self, team_id: int, last: int = 5) -> List[Dict]:
        """Get the recent matches for a team to calculate Momentum (xG diff)."""
        return await self._request(
            SportType.SOCCER, "/fixtures",
            params={"team": team_id, "last": last}
        )

    async def get_fixture_lineups(self, fixture_id: int) -> List[Dict]:
        """Get the confirmed or predicted lineups for a fixture."""
        return await self._request(
            SportType.SOCCER, "/fixtures/lineups",
            params={"fixture": fixture_id}
        )
