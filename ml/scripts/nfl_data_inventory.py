"""
OmniPicks — NFL v3, fase 0: qué datos hay en nflverse y desde cuándo.

Para cada fuente prueba varias temporadas y reporta filas, semanas y algunas columnas clave.
No guarda nada: solo dice qué se puede usar.

Uso:
    python ml/scripts/nfl_data_inventory.py
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import nflreadpy as nfl
import polars as pl

PROBE = [2010, 2012, 2016, 2018, 2020, 2022, 2023, 2024, 2025, 2026]
SOURCES = {
    "participation": (nfl.load_participation, ["offense_players", "defense_players", "offense_personnel", "defenders_in_box",
                                               "defense_coverage_type", "was_pressure", "time_to_throw"]),
    "ftn_charting": (nfl.load_ftn_charting, ["is_play_action", "is_rpo", "n_blitzers", "is_motion", "is_drop", "is_catchable_ball"]),
    "snap_counts": (nfl.load_snap_counts, ["offense_snaps", "defense_snaps", "st_snaps", "position"]),
    "injuries": (nfl.load_injuries, ["report_status", "practice_status", "position"]),
    "rosters_weekly": (nfl.load_rosters_weekly, ["depth_chart_position", "status", "years_exp"]),
    "depth_charts": (nfl.load_depth_charts, ["depth_team", "pos_abb", "position"]),
    "player_stats": (nfl.load_player_stats, ["passing_epa", "receiving_air_yards", "target_share", "def_sacks"]),
}
NO_SEASON = {
    "nextgen_stats": (lambda: nfl.load_nextgen_stats(), ["avg_time_to_throw", "avg_separation", "rush_yards_over_expected"]),
    "pfr_advstats_pass": (lambda: nfl.load_pfr_advstats(stat_type="pass", summary_level="week"), ["times_pressured", "times_blitzed", "passing_bad_throws"]),
    "pfr_advstats_def": (lambda: nfl.load_pfr_advstats(stat_type="def", summary_level="week"), ["def_pressures", "def_missed_tackles", "def_targets"]),
    "officials": (lambda: nfl.load_officials(), ["off_pos", "name"]),
    "draft_picks": (lambda: nfl.load_draft_picks(), ["pick", "round", "gsis_id"]),
    "combine": (lambda: nfl.load_combine(), ["forty", "vertical", "pos"]),
    "contracts": (lambda: nfl.load_contracts(), ["apy", "apy_cap_pct", "gsis_id"]),
}


def cols(df, want):
    return ", ".join(c for c in want if c in df.columns) or "—"


def main():
    print("Fuentes por temporada (filas · última semana):")
    for name, (fn, want) in SOURCES.items():
        found, sample = [], None
        for s in PROBE:
            try:
                df = fn([s])
                if df.height:
                    wk = int(df["week"].max()) if "week" in df.columns and df["week"].max() is not None else None
                    found.append(f"{s}:{df.height:,}" + (f"·s{wk}" if wk else ""))
                    sample = df
            except Exception as e:
                found.append(f"{s}:✗")
            time.sleep(0.2)
        print(f"  {name:16} {'  '.join(found)}")
        if sample is not None:
            print(f"  {'':16} columnas clave: {cols(sample, want)}")
    print("\nFuentes de un solo archivo (rango de temporadas):")
    for name, (fn, want) in NO_SEASON.items():
        try:
            df = fn()
            rng = f"{int(df['season'].min())}–{int(df['season'].max())}" if "season" in df.columns else "—"
            print(f"  {name:18} {df.height:>8,} filas · temporadas {rng} · {cols(df, want)}")
        except Exception as e:
            print(f"  {name:18} ✗ {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    main()
