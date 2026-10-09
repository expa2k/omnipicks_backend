import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

try:
    print("Fetching NFL data directly from nflverse...")
    # Fetch 2024 play by play to compute EPA
    url = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2024.csv.gz"
    pbp = pd.read_csv(url, compression='gzip', low_memory=False)
    
    # Compute EPA per play
    off_epa = pbp.groupby('posteam')['epa'].mean().reset_index()
    def_epa = pbp.groupby('defteam')['epa'].mean().reset_index()
    
    print("\n--- OFFENSIVE EPA ---")
    print(off_epa.head())
    
    print("\n--- DEFENSIVE EPA ---")
    print(def_epa.head())
except Exception as e:
    print("Error:", e)

