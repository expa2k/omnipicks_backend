import statsapi
import json

def get_h2h_stats(team_a_id, team_b_id, season=2026):
    sched = statsapi.schedule(start_date=f'{season}-01-01', end_date=f'{season}-12-31', team=team_a_id, opponent=team_b_id)
    a_runs = 0
    b_runs = 0
    games_played = 0
    for g in sched:
        if g['status'] == 'Final':
            if g['home_id'] == team_a_id:
                a_runs += g['home_score']
                b_runs += g['away_score']
            else:
                a_runs += g['away_score']
                b_runs += g['home_score']
            games_played += 1
    
    if games_played == 0:
        return 4.5, 4.5 # Default averages
    
    return round(a_runs/games_played, 2), round(b_runs/games_played, 2)

print(get_h2h_stats(137, 112)) # SF vs CHC
