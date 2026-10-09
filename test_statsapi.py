import statsapi
from datetime import date, timedelta

target_dates = [
    date(2026, 6, 11).strftime("%Y-%m-%d"),
    date(2026, 6, 12).strftime("%Y-%m-%d"),
    date(2026, 6, 13).strftime("%Y-%m-%d"),
    date(2026, 6, 14).strftime("%Y-%m-%d")
]

for d in target_dates:
    print(f"\n--- Schedule for {d} ---")
    sched = statsapi.schedule(date=d, sportId=1)
    for g in sched:
        home = g['home_name']
        away = g['away_name']
        h_pitcher = g.get('home_probable_pitcher', 'TBA')
        a_pitcher = g.get('away_probable_pitcher', 'TBA')
        if "Cubs" in home or "Cubs" in away or "Giants" in home or "Giants" in away or "Rockies" in home or "Rockies" in away or "Athletics" in home or "Athletics" in away:
            print(f"{away} ({a_pitcher}) @ {home} ({h_pitcher})")

