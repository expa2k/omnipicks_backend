"""
OmniPicks — NBA: ¿qué tan bien proyectamos puntos, rebotes, asistencias y triples por jugador?

Caminando hacia adelante en una temporada (solo con información previa a cada partido):
  · promedio10 : promedio de sus últimos 10 partidos (lo que usaría cualquiera)
  · modelo     : tasa por minuto con descuento (los partidos recientes pesan más, encogida hacia
                 la media de la liga si hay pocos minutos) × minutos esperados con las bajas del día
                 × ritmo del partido
  · techo      : la misma tasa × minutos reales (cuánto se ganaría acertando los minutos)

Se mide el error medio (MAE) en jugadores que jugaron al menos 10 minutos.

Uso:
    python ml/scripts/nba_player_props_eval.py --season 2025-26
"""
import argparse
import logging
import os
import sys
from collections import deque

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy import create_engine, text

from app.config import get_settings
from app.services.nba_lineups import ROTATION_MIN, RosterTracker, expected_minutes
from app.services.nba_props import PROP_STATS, PlayerRates

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("nba_player_props_eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2025-26")
    args = ap.parse_args()
    engine = create_engine(get_settings().mysql_url_sync)
    start_year = int(args.season[:4])
    prev = f"{start_year - 1}-{start_year % 100:02d}"
    with engine.connect() as conn:
        box = pd.read_sql(text(
            "SELECT game_id, game_date, season, team_id, opp_team_id, player_id, minutes, pts, oreb, dreb, ast, fg3m, "
            "fga, fta, tov FROM nba_player_games WHERE season IN (:p, :s) ORDER BY game_date, game_id"),
            conn, params={"p": prev, "s": args.season})
    box["reb"] = box.oreb + box.dreb
    box["game_date"] = pd.to_datetime(box.game_date).dt.date
    # Posesiones por equipo y partido (para el ritmo)
    tp = box.groupby(["game_id", "team_id"]).agg(fga=("fga", "sum"), fta=("fta", "sum"), tov=("tov", "sum"), oreb=("oreb", "sum"))
    tp["poss"] = tp.fga + 0.44 * tp.fta + tp.tov - tp.oreb
    game_poss = tp.groupby("game_id").poss.mean()

    rates = PlayerRates()
    tracker = RosterTracker()
    last10 = {}                                   # jugador → deque de (pts, reb, ast, fg3m)
    team_pace = {}                                # equipo → posesiones medias (con descuento)
    league_pace = [0.0, 0.0]
    rows = []

    for day, dgrp in box.groupby("game_date", sort=True):
        # 1) Predecir cada equipo de cada partido del día
        if dgrp.season.iloc[0] == args.season:
            for (gid, tid), g in dgrp.groupby(["game_id", "team_id"]):
                cands = tracker.candidates(int(tid), day)
                if len(cands) < 8:
                    continue
                act = dict(zip(g.player_id.astype(int), g.minutes))
                outs = {c.player: 0.0 for c in cands if c.player not in act and c.rate >= 0.6 and c.base >= ROTATION_MIN}
                back = {c.player for c in cands if c.player in act and c.rate <= 0.2 and c.base >= ROTATION_MIN}
                mins = expected_minutes(cands, outs, back)[0]
                lp = league_pace[0] / league_pace[1] if league_pace[1] else 99.0
                opp = int(g.opp_team_id.iloc[0])
                pace_factor = ((team_pace.get(int(tid), lp) + team_pace.get(opp, lp)) / 2.0) / lp
                for c, m in zip(cands, mins):
                    if c.player not in act or act[c.player] < 10:
                        continue
                    real = g[g.player_id == c.player].iloc[0]
                    r = rates.rates(c.player)
                    l10 = last10.get(c.player)
                    rec = {"player": c.player, "min_pred": m, "min_real": real.minutes,
                           "team_outs": sum(cc.base for cc in cands if cc.player in outs)}
                    m10 = np.mean([x["min"] for x in l10]) if l10 else np.nan
                    # Minutos: los de las bajas del día, pero anclados a su promedio reciente
                    m_blend = 0.5 * m + 0.5 * m10 + (m - (c.base * c.rate)) * 0.5 if l10 else m
                    for s in PROP_STATS:
                        rec[f"{s}_real"] = real[s]
                        rec[f"{s}_l10min"] = r[s] * m10 * pace_factor if l10 else np.nan
                        rec[f"{s}_blendmin"] = r[s] * m_blend * pace_factor
                        rec[f"{s}_mix"] = 0.5 * r[s] * m_blend * pace_factor + 0.5 * (np.mean([x[s] for x in l10]) if l10 else 0)
                        rec[f"{s}_model"] = r[s] * m * pace_factor
                        rec[f"{s}_ceiling"] = r[s] * real.minutes * pace_factor
                        rec[f"{s}_avg10"] = np.mean([x[s] for x in l10]) if l10 else np.nan
                    rows.append(rec)
        # 2) Actualizar con lo jugado
        for (gid, tid), g in dgrp.groupby(["game_id", "team_id"]):
            tracker.update(day, int(tid), g.player_id.tolist(), g.minutes.tolist())
            for r in g.itertuples():
                if r.minutes > 0:
                    rates.update(int(r.player_id), r.minutes, {s: getattr(r, s) for s in PROP_STATS})
                    last10.setdefault(int(r.player_id), deque(maxlen=10)).append({**{s: getattr(r, s) for s in PROP_STATS}, "min": r.minutes})
            if gid in game_poss.index:
                p = float(game_poss[gid])
                team_pace[int(tid)] = 0.9 * team_pace.get(int(tid), p) + 0.1 * p
                league_pace[0] = 0.997 * league_pace[0] + p
                league_pace[1] = 0.997 * league_pace[1] + 1.0

    df = pd.DataFrame(rows).dropna()
    print(f"\nTemporada {args.season}: {len(df):,} actuaciones (jugadores con 10+ minutos)\n")
    out = {}
    for s in PROP_STATS:
        out[s] = {k: float((df[f"{s}_{k}"] - df[f"{s}_real"]).abs().mean()) for k in ("avg10", "model", "l10min", "blendmin", "mix", "ceiling")}
    t = pd.DataFrame(out).T
    t.columns = ["Promedio 10", "Tasa×min bajas", "Tasa×min prom10", "Tasa×min ajustados", "Mezcla", "Techo"]
    t.index = ["Puntos", "Rebotes", "Asistencias", "Triples"]
    print("Error medio por jugador y partido:")
    print(t.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\nError de minutos: {float((df.min_pred - df.min_real).abs().mean()):.2f}")
    for title, sub in (("Compañeros de una baja de 25+ min", df[df.team_outs >= 25]), ("Sin bajas en su equipo", df[df.team_outs == 0])):
        o = {s: {k: float((sub[f"{s}_{k}"] - sub[f"{s}_real"]).abs().mean()) for k in ("avg10", "mix", "blendmin")} for s in PROP_STATS}
        tt = pd.DataFrame(o).T
        tt.columns = ["Promedio 10", "Mezcla", "Tasa×min ajustados"]
        print(f"\n{title} ({len(sub):,}):\n" + tt.to_string(float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
