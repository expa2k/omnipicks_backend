from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import teams

celtics = [team for team in teams.get_teams() if team['abbreviation'] == 'BOS'][0]
team_id = celtics['id']

try:
    four_factors = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Four Factors', season='2025-26')
    df = four_factors.get_data_frames()[0]
    team_data = df[df['TEAM_ID'] == team_id]
    print("\n--- 4 FACTORS ---")
    print(team_data[['TEAM_NAME', 'EFG_PCT', 'FTA_RATE', 'TM_TOV_PCT', 'OREB_PCT']])
except Exception as e:
    print("Error 4 factors:", e)

