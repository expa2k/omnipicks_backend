"""
OmniPicks — API Clients: Bzzoiro (Soccer)
"""
import httpx
import logging
from typing import Dict, Any, List, Optional
from app.config import get_settings

logger = logging.getLogger(__name__)

class BzzoiroClient:
    """Client for Bzzoiro Sports Data API (Soccer)."""
    
    def __init__(self):
        settings = get_settings()
        self.base_url = settings.bzzoiro_base_url or "https://sports.bzzoiro.com"
        self.api_key = settings.bzzoiro_api_key
        self.headers = {
            "Authorization": f"Token {self.api_key}",
            "Accept": "application/json"
        }
    
    async def _request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Generic request handler."""
        if not self.api_key:
            logger.warning("Bzzoiro API key not configured")
            return {}
            
        url = f"{self.base_url}{endpoint}"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.request(method, url, headers=self.headers, params=params, timeout=10.0)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"Bzzoiro API error: {e}")
                return {}

    async def get_leagues(self) -> List[Dict]:
        """Get supported soccer leagues."""
        data = await self._request("GET", "/api/v2/leagues/?limit=200")
        return data.get("results", [])

    async def get_events_by_date(self, date_str: str) -> List[Dict]:
        """Get events for a specific date (YYYY-MM-DD) using binary search on offset."""
        try:
            # 1. Get total count
            data = await self._request("GET", "/api/v2/events/?limit=1")
            count = data.get("count", 0)
            if count == 0:
                return []
                
            low = 0
            high = count - 1
            limit = 100
            found_offset = -1
            
            # Binary search the offset space
            while low <= high:
                mid = (low + high) // 2
                mid_data = await self._request("GET", f"/api/v2/events/?limit={limit}&offset={mid}")
                events = mid_data.get("results", [])
                if not events:
                    break
                    
                mid_date_str = str(events[0].get("event_date", ""))[:10]
                if mid_date_str == date_str:
                    found_offset = mid
                    break
                elif mid_date_str < date_str:
                    # Bzzoiro returns dates descending. A smaller date means we went too far right (offset too high)
                    high = mid - 1
                else:
                    low = mid + 1
                    
            if found_offset != -1:
                # Fetch a generous chunk around the found offset to guarantee we get all games for that day
                chunk_offset = max(0, found_offset - 100)
                final_data = await self._request("GET", f"/api/v2/events/?limit=200&offset={chunk_offset}")
                target_events = [e for e in final_data.get("results", []) if str(e.get("event_date", "")).startswith(date_str)]
                return target_events
                
        except Exception as e:
            logger.error(f"Error binary searching Bzzoiro events: {e}")
            
        return []

    async def get_event_detail(self, event_id: str) -> Dict:
        """Get detailed info for a specific event."""
        data = await self._request("GET", f"/api/v2/events/{event_id}/")
        return data.get("data", {})

    async def get_event_prediction(self, event_id: str) -> Dict:
        """Get Bzzoiro ML prediction for an event."""
        data = await self._request("GET", f"/api/v2/events/{event_id}/prediction/")
        return data

    async def get_live_events(self) -> List[Dict]:
        """Get currently live events."""
        data = await self._request("GET", "/api/v2/events/live/")
        return data.get("data", [])
