import asyncio
import logging
from datetime import date, timedelta
from typing import List, Dict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, update

from app.database import get_engine
from app.services.bzzoiro_client import BzzoiroClient
from app.db_models import TeamHistoricalProfile, Game

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CronSync")

async def sync_team_profiles(session: AsyncSession, bzzoiro_client: BzzoiroClient, team_name: str, team_id: int, event_id: int):
    """Downloads and caches the historical profiles using Bzzoiro."""
    
    # 1. Fetch Lineups for the upcoming event (ALWAYS DO THIS, BEFORE CACHE)
    unavailable_count = 0
    try:
        lineup_data = await bzzoiro_client._request("GET", f"/api/v2/events/{event_id}/lineups/")
        unav = lineup_data.get("unavailable_players")
        if unav:
            # Bzzoiro returns injured players in "home" and "away" lists
            lineups = lineup_data.get("lineups")
            if lineups and lineups.get("home") and lineups["home"].get("team_id") == team_id:
                unavailable_count = len(unav.get("home", []))
            elif lineups and lineups.get("away") and lineups["away"].get("team_id") == team_id:
                unavailable_count = len(unav.get("away", []))
                
        if lineup_data:
            # Save the lineups JSON directly into the Game table so the React frontend can display it
            await session.execute(
                update(Game).where(Game.external_id == str(event_id)).values(raw_data=lineup_data)
            )
            await session.commit()
    except Exception as e:
        logger.warning(f"Could not fetch lineups for event {event_id}: {e}")

    # 2. Check Cache (7 Days) for Historical Profile
    from datetime import datetime
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    
    cache_check = await session.execute(
        select(TeamHistoricalProfile).where(
            TeamHistoricalProfile.team_name == team_name,
            TeamHistoricalProfile.updated_at >= seven_days_ago
        )
    )
    if cache_check.scalars().first():
        logger.info(f"⏭️ Skipping historical profile for {team_name}: Cache is fresh (< 7 days).")
        return
        
    logger.info(f"🔄 Syncing {team_name} historical profile from Bzzoiro...")
    
    # 3. Fetch Last 30 Fixtures from Bzzoiro
    try:
        events_data = await bzzoiro_client._request("GET", f"/api/v2/events/?team_id={team_id}&status=finished&limit=30")
        fixtures = events_data.get("results", [])
    except Exception as e:
        logger.error(f"Error fetching fixtures for {team_name}: {e}")
        return

    if not fixtures:
        logger.warning(f"No recent fixtures found for {team_name}")
        return

    # Helper to calculate stats
    def calculate_stats(games: List[Dict], target_team_id: int):
        stats_by_tier = {1: {"gf":0,"ga":0,"wins":0,"matches":0},
                         2: {"gf":0,"ga":0,"wins":0,"matches":0},
                         3: {"gf":0,"ga":0,"wins":0,"matches":0},
                         4: {"gf":0,"ga":0,"wins":0,"matches":0},
                         5: {"gf":0,"ga":0,"wins":0,"matches":0}}
                         
        def get_tier(name):
            t1 = ["Germany", "France", "Brazil", "Argentina", "Spain", "England", "Portugal", "Netherlands", "Italy", "Belgium"]
            t2 = ["Uruguay", "Colombia", "Croatia", "Morocco", "USA", "Mexico", "Japan", "Senegal"]
            t3 = ["Ecuador", "Côte d'Ivoire", "Australia", "Türkiye", "Serbia", "Switzerland"]
            for t in t1: 
                if t.lower() in name.lower(): return 1
            for t in t2:
                if t.lower() in name.lower(): return 2
            for t in t3:
                if t.lower() in name.lower(): return 3
            return 4

        for g in games:
            home_id = g.get("home_team_id")
            away_id = g.get("away_team_id")
            
            is_home = (home_id == target_team_id)
            opponent = g.get("away_team") if is_home else g.get("home_team")
            
            gf = g.get("home_score") if is_home else g.get("away_score")
            ga = g.get("away_score") if is_home else g.get("home_score")
            
            if gf is None or ga is None:
                continue
                
            opp_tier = get_tier(opponent)
            
            stats_by_tier[opp_tier]["gf"] += gf
            stats_by_tier[opp_tier]["ga"] += ga
            stats_by_tier[opp_tier]["matches"] += 1
            if gf > ga:
                stats_by_tier[opp_tier]["wins"] += 1
                
        return stats_by_tier

    # 4. Process Dimension A (Last 30 games)
    stats_last_year = calculate_stats(fixtures, team_id)
    
    # 5. Process Dimension B (Recent 5 games)
    stats_recent = calculate_stats(fixtures[:5], team_id)
    
    # Clean old profiles
    await session.execute(delete(TeamHistoricalProfile).where(TeamHistoricalProfile.team_name == team_name))
    
    # Insert new profiles
    for tier in range(1, 6):
        sy = stats_last_year[tier]
        if sy["matches"] > 0:
            profile1 = TeamHistoricalProfile(
                team_name=team_name,
                opponent_tier=tier,
                timeframe="last_year",
                avg_goals_for=sy["gf"]/sy["matches"],
                avg_goals_against=sy["ga"]/sy["matches"],
                win_rate=sy["wins"]/sy["matches"],
                matches_played=sy["matches"],
                unavailable_players_count=unavailable_count
            )
            session.add(profile1)
            
        sr = stats_recent[tier]
        if sr["matches"] > 0:
            profile2 = TeamHistoricalProfile(
                team_name=team_name,
                opponent_tier=tier,
                timeframe="recent_5_games",
                avg_goals_for=sr["gf"]/sr["matches"],
                avg_goals_against=sr["ga"]/sr["matches"],
                win_rate=sr["wins"]/sr["matches"],
                matches_played=sr["matches"],
                unavailable_players_count=unavailable_count
            )
            session.add(profile2)
            
    await session.commit()
    logger.info(f"✅ {team_name} synced successfully from Bzzoiro! (Bajas reportadas: {unavailable_count})")


async def run_cron():
    logger.info("Starting OmniPicks Sync Cron Job (Bzzoiro V3.1)...")
    engine = get_engine()
    bzzoiro = BzzoiroClient()
    
    dates_to_check = [date.today(), date.today() + timedelta(days=1)]
    teams_to_sync = {}
    
    target_leagues = ["27", "1", "2", "39", "140", "135", "78", "262", "253"] 
    
    for d in dates_to_check:
        events = await bzzoiro.get_events_by_date(d.isoformat())
        for e in events:
            league = str(e.get("league_id", "1"))
            if league not in target_leagues:
                continue
                
            home = e.get("home_team")
            away = e.get("away_team")
            home_id = e.get("home_team_id")
            away_id = e.get("away_team_id")
            event_id = e.get("id")
            
            if home and home_id: teams_to_sync[home] = {"team_id": home_id, "event_id": event_id}
            if away and away_id: teams_to_sync[away] = {"team_id": away_id, "event_id": event_id}
            
    logger.info(f"Found {len(teams_to_sync)} unique teams playing in the next 48h.")
    
    async with AsyncSession(engine) as session:
        for team_name, info in teams_to_sync.items():
            await sync_team_profiles(session, bzzoiro, team_name, info["team_id"], info["event_id"])
            await asyncio.sleep(1) # Bzzoiro is faster, 1 sec is fine
            
    logger.info("Cron Job Finished!")

if __name__ == "__main__":
    asyncio.run(run_cron())
