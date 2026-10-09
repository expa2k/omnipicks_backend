import statsapi
from datetime import datetime, timedelta

def get_current_form(team_id, game_date_str):
    end_date = datetime.strptime(game_date_str, "%Y-%m-%d") - timedelta(days=1)
    start_date = end_date - timedelta(days=10)
    sched = statsapi.schedule(start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"), team=team_id)
    
    runs_scored = 0
    runs_allowed = 0
    games_played = 0
    
    # Only take the last 5 finalized games
    sched = [g for g in sched if g['status'] == 'Final']
    for g in sched[-5:]:
        if g['home_id'] == team_id:
            runs_scored += g['home_score']
            runs_allowed += g['away_score']
        else:
            runs_scored += g['away_score']
            runs_allowed += g['home_score']
        games_played += 1
        
    if games_played == 0:
        return 4.5, 4.5
        
    return round(runs_scored/games_played, 2), round(runs_allowed/games_played, 2)

print(get_current_form(137, "2026-06-12"))
