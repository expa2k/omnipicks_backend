import statsapi
import json

try:
    print(json.dumps(statsapi.get("team_stats", {'teamId': 137, 'season': 2026, 'group': 'hitting', 'stats': 'season'}), indent=2))
except Exception as e:
    print(e)

