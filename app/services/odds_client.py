"""
OmniPicks — API Clients: The Odds API
"""
import httpx
import logging
from typing import Dict, Any, List, Optional
from app.config import get_settings

logger = logging.getLogger(__name__)

class OddsClient:
    """Client for The Odds API."""
    
    def __init__(self):
        settings = get_settings()
        self.base_url = settings.odds_api_url
        self.api_key = settings.odds_api_key
        self.credits_used = 0
        self.credits_remaining = 0
    
    async def _request(self, endpoint: str, params: Optional[Dict] = None) -> List[Dict]:
        """Generic request handler mapping to Odds API."""
        if not self.api_key:
            logger.warning("Odds API key not configured")
            return []
            
        if params is None:
            params = {}
        params["apiKey"] = self.api_key

        url = f"{self.base_url}{endpoint}"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.request("GET", url, params=params, timeout=15.0)
                response.raise_for_status()
                
                # Track usage
                self.credits_used = int(response.headers.get("x-requests-used", 0))
                self.credits_remaining = int(response.headers.get("x-requests-remaining", 0))
                logger.debug(f"Odds API Credits: {self.credits_used} used, {self.credits_remaining} remaining")
                
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"Odds API error: {e}")
                if hasattr(e, "response") and e.response is not None:
                    raise e
                return []

    async def get_sports(self) -> List[Dict]:
        """Get list of active sports from The Odds API."""
        return await self._request("/sports")

    async def get_odds(self, sport_key: str, regions: str = "us", markets: str = "h2h,spreads,totals") -> List[Dict]:
        """Get odds for a specific sport."""
        if "soccer" in sport_key:
            markets += ",btts"
            
        params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",
            "dateFormat": "iso"
        }
        
        try:
            return await self._request(f"/sports/{sport_key}/odds", params=params)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 422 and "btts" in markets:
                logger.warning(f"Market 'btts' not supported for {sport_key}, retrying without it.")
                params["markets"] = params["markets"].replace(",btts", "").replace("btts", "")
                try:
                    return await self._request(f"/sports/{sport_key}/odds", params=params)
                except Exception:
                    return []
            return []
        except Exception:
            return []

    async def get_event_odds(self, sport_key: str, event_id: str, regions: str = "us", markets: str = "h2h,spreads,totals,btts") -> List[Dict]:
        """Get odds for a specific event."""
        params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",
            "dateFormat": "iso"
        }
        # The odds API returns a single object for an event, we wrap it in a list to match get_odds
        data = await self._request(f"/sports/{sport_key}/events/{event_id}/odds", params=params)
        return [data] if isinstance(data, dict) else data
