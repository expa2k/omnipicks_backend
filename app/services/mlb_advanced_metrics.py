import statsapi
import datetime
from datetime import timedelta

def get_fip(pitcher_stats):
    """Level 5: Calculates FIP given a pitcher's stats dict."""
    try:
        hr = int(pitcher_stats.get('homeRuns', 0))
        bb = int(pitcher_stats.get('baseOnBalls', 0))
        hbp = int(pitcher_stats.get('hitByPitch', 0))
        so = int(pitcher_stats.get('strikeOuts', 0))
        ip_str = str(pitcher_stats.get('inningsPitched', '0'))
        
        if '.' in ip_str:
            full, part = ip_str.split('.')
            ip = int(full) + (int(part) / 3.0)
        else:
            ip = float(ip_str)
            
        if ip <= 0: return 4.50
        
        fip = ((13 * hr) + (3 * (bb + hbp)) - (2 * so)) / ip + 3.20
        return round(fip, 2)
    except:
        return 4.50

def get_h2h_stats(team_a_id, team_b_id, game_date_str):
    """Level 2: Head to Head Stats"""
    try:
        season = game_date_str.split('-')[0]
        sched = statsapi.schedule(start_date=f'{season}-01-01', end_date=game_date_str, team=team_a_id, opponent=team_b_id)
        a_runs = 0
        b_runs = 0
        games_played = 0
        for g in sched:
            if g['status'] == 'Final' and g['game_date'] < game_date_str:
                if g['home_id'] == team_a_id:
                    a_runs += g['home_score']
                    b_runs += g['away_score']
                else:
                    a_runs += g['away_score']
                    b_runs += g['home_score']
                games_played += 1
        
        if games_played == 0:
            return 4.5, 4.5
        
        return round(a_runs/games_played, 2), round(b_runs/games_played, 2)
    except:
        return 4.5, 4.5

def get_current_form(team_id, game_date_str):
    """Level 3: Current Form (Last 5 games)"""
    try:
        end_date = datetime.datetime.strptime(game_date_str, "%Y-%m-%d") - timedelta(days=1)
        start_date = end_date - timedelta(days=10)
        sched = statsapi.schedule(start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"), team=team_id)
        
        runs_scored = 0
        runs_allowed = 0
        games_played = 0
        
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
    except:
        return 4.5, 4.5

def lookup_team_id(team_name):
    """Convierte un nombre de equipo al ID nativo de statsapi."""
    try:
        results = statsapi.lookup_team(team_name)
        if results:
            return results[0]['id']
    except:
        pass
    return None

def extract_advanced_mlb_features(game_data: dict, game_date_str: str) -> dict:
    features = {}
    
    # Base Data
    home_name = game_data.get("teams", {}).get("home", {}).get("name", "Local")
    away_name = game_data.get("teams", {}).get("away", {}).get("name", "Visitante")
    
    home_id = lookup_team_id(home_name) or game_data.get("teams", {}).get("home", {}).get("id", 1)
    away_id = lookup_team_id(away_name) or game_data.get("teams", {}).get("away", {}).get("id", 2)
    
    game_date = game_date_str if game_date_str else game_data.get("date", "").split("T")[0]
    season = game_date.split("-")[0]
    
    # ---------------------------------------------------------
    # LEVEL 1: GENERAL STATS
    # ---------------------------------------------------------
    try:
        home_hitting = statsapi.get("team_stats", {'teamId': home_id, 'season': season, 'group': 'hitting', 'stats': 'season'})['stats'][0]['splits'][0]['stat']
        away_hitting = statsapi.get("team_stats", {'teamId': away_id, 'season': season, 'group': 'hitting', 'stats': 'season'})['stats'][0]['splits'][0]['stat']
        features['home_runs_scored_per_game'] = round(int(home_hitting['runs']) / max(int(home_hitting['gamesPlayed']), 1), 2)
        features['away_runs_scored_per_game'] = round(int(away_hitting['runs']) / max(int(away_hitting['gamesPlayed']), 1), 2)
        
        home_pitching = statsapi.get("team_stats", {'teamId': home_id, 'season': season, 'group': 'pitching', 'stats': 'season'})['stats'][0]['splits'][0]['stat']
        away_pitching = statsapi.get("team_stats", {'teamId': away_id, 'season': season, 'group': 'pitching', 'stats': 'season'})['stats'][0]['splits'][0]['stat']
        features['home_runs_allowed_per_game'] = round(int(home_pitching['runs']) / max(int(home_pitching['gamesPlayed']), 1), 2)
        features['away_runs_allowed_per_game'] = round(int(away_pitching['runs']) / max(int(away_pitching['gamesPlayed']), 1), 2)
        
        home_bullpen = statsapi.get("team_stats", {'teamId': home_id, 'season': season, 'group': 'pitching', 'stats': 'season', 'split': 'relief'})['stats'][0]['splits'][0]['stat']
        away_bullpen = statsapi.get("team_stats", {'teamId': away_id, 'season': season, 'group': 'pitching', 'stats': 'season', 'split': 'relief'})['stats'][0]['splits'][0]['stat']
        features['home_bullpen_era'] = float(home_bullpen.get('era', 4.5))
        features['away_bullpen_era'] = float(away_bullpen.get('era', 4.5))
    except:
        features['home_runs_scored_per_game'] = 4.5
        features['away_runs_scored_per_game'] = 4.5
        features['home_runs_allowed_per_game'] = 4.5
        features['away_runs_allowed_per_game'] = 4.5
        features['home_bullpen_era'] = 4.5
        features['away_bullpen_era'] = 4.5
        
    try:
        # Extraer Clima y Viento
        game_pk = game_data.get("external_id")
        if game_pk:
            box = statsapi.boxscore_data(int(game_pk))
            box_info = box.get("gameBoxInfo", [])
            for info in box_info:
                if info.get('label') == 'Weather':
                    features['weather_info'] = info.get('value', '')
                elif info.get('label') == 'Wind':
                    features['wind_info'] = info.get('value', '')
    except:
        pass

    # ---------------------------------------------------------
    # LEVEL 2: HEAD-TO-HEAD
    # ---------------------------------------------------------
    h_h2h, a_h2h = get_h2h_stats(home_id, away_id, game_date)
    features['home_h2h_runs'] = h_h2h
    features['away_h2h_runs'] = a_h2h
    
    # ---------------------------------------------------------
    # LEVEL 3: CURRENT FORM (LAST 5 GAMES)
    # ---------------------------------------------------------
    h_form_scored, h_form_allowed = get_current_form(home_id, game_date)
    a_form_scored, a_form_allowed = get_current_form(away_id, game_date)
    features['home_form_runs_scored'] = h_form_scored
    features['home_form_runs_allowed'] = h_form_allowed
    features['away_form_runs_scored'] = a_form_scored
    features['away_form_runs_allowed'] = a_form_allowed

    # ---------------------------------------------------------
    # LEVEL 4 & 5: PROBABLE PITCHERS, FIP, UMPIRE, WEATHER
    # ---------------------------------------------------------
    features[f"Pitcher_{home_name}"] = "Por Confirmar (TBA)"
    features[f"Pitcher_{away_name}"] = "Por Confirmar (TBA)"
    features['home_pitcher_fip'] = 4.50
    features['away_pitcher_fip'] = 4.50
    features['home_pitcher_era'] = 4.50
    features['away_pitcher_era'] = 4.50
    
    try:
        sched = statsapi.schedule(date=game_date, sportId=1)
        target_game = None
        for g in sched:
            if home_name in g['home_name'] or away_name in g['away_name']:
                target_game = g
                break
                
        if target_game:
            home_pitcher_name = target_game.get('home_probable_pitcher', 'Por Confirmar (TBA)')
            away_pitcher_name = target_game.get('away_probable_pitcher', 'Por Confirmar (TBA)')
            
            features[f"Pitcher_{home_name}"] = home_pitcher_name if home_pitcher_name else "Por Confirmar (TBA)"
            features[f"Pitcher_{away_name}"] = away_pitcher_name if away_pitcher_name else "Por Confirmar (TBA)"
            
            # Home Pitcher Advanced Stats
            if home_pitcher_name and home_pitcher_name != 'Por Confirmar (TBA)':
                player = statsapi.lookup_player(home_pitcher_name)
                if player:
                    stats = statsapi.player_stat_data(player[0]['id'], group="[pitching]", type="season")
                    if stats and 'stats' in stats and len(stats['stats']) > 0:
                        pitcher_stats = stats['stats'][0]['stats']
                        features['home_pitcher_fip'] = get_fip(pitcher_stats)
                        features['home_pitcher_era'] = float(pitcher_stats.get('era', '4.50').replace('-.--', '4.50'))
                        
            # Away Pitcher Advanced Stats
            if away_pitcher_name and away_pitcher_name != 'Por Confirmar (TBA)':
                player = statsapi.lookup_player(away_pitcher_name)
                if player:
                    stats = statsapi.player_stat_data(player[0]['id'], group="[pitching]", type="season")
                    if stats and 'stats' in stats and len(stats['stats']) > 0:
                        pitcher_stats = stats['stats'][0]['stats']
                        features['away_pitcher_fip'] = get_fip(pitcher_stats)
                        features['away_pitcher_era'] = float(pitcher_stats.get('era', '4.50').replace('-.--', '4.50'))
                        
            # Umpire, Weather, Wind
            try:
                box = statsapi.boxscore_data(target_game['game_id'])
                if 'gameBoxInfo' in box:
                    for info in box['gameBoxInfo']:
                        label = info.get('label', '')
                        val = info.get('value', '')
                        if label == 'Umpires':
                            features['umpire'] = val.split('.')[0] if '.' in val else val
                        elif label == 'Weather':
                            features['weather'] = val
                        elif label == 'Wind':
                            features['wind'] = val
            except Exception as box_e:
                pass
    except Exception as e:
        print("Error pulling advanced statsapi metrics:", e)
        
    return features
