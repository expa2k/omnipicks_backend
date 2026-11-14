"""
OmniPicks — MLB v2 en producción (docs/PLAN_MLB.md).

Modelo de carreras por jugador (`mlb_model`): nivel de cada equipo en la temporada + alineación del día
+ abridor rival + parque, localía, noche, ventaja de mano y cansancio del bullpen. Binomial negativa →
moneyline, run line y total.

Para cada juego por jugar: abridores probables y alineaciones de la MLB StatsAPI (si la alineación
todavía no sale, se usa la última del equipo). El histórico: Retrosheet hasta donde haya y la
StatsAPI para lo más reciente. El backtest no encontró ventaja contra el cierre: los picks se dan con
la probabilidad del modelo y su confianza, sin marcarlos como "recomendados".
"""
import logging
import threading
import time
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from app.services.mlb_history import (load_all_games, load_handedness, load_people, retro_team, statsapi_games)
from app.services.mlb_model import MLBParams, fit, game_probs, team_rows

logger = logging.getLogger(__name__)

MODEL_NAME = "mlb_v2"
REFIT_SECONDS = 6 * 3600
BACKTEST = {"games": 15402, "seasons": "2015–2021", "logloss": 0.6849, "logloss_market": 0.6715, "logloss_base": 0.6911}
_lock = threading.Lock()
_state: Dict = {"at": 0.0}


def _season_now(d: date) -> int:
    return d.year if d.month >= 3 else d.year - 1


def _state_now() -> Dict:
    with _lock:
        if time.time() - _state.get("at", 0) > REFIT_SECONDS:
            today = date.today()
            cur = _season_now(today)
            games = load_all_games(cur - 3, cur)
            hand = load_handedness(range(cur - 3, cur + 1))
            rows = team_rows(games, hand)
            f = fit(rows, np.datetime64(today.isoformat()), MLBParams())
            last_lineup, park_of = {}, {}
            for r in games.sort("date").iter_rows(named=True):
                last_lineup[r["home"]] = r["home_lineup"]; last_lineup[r["away"]] = r["away_lineup"]
                park_of[r["home"]] = r["park"]
            ppl = load_people()
            names = {r_: f"{a} {l}" for r_, a, l in zip(ppl["retro_id"].to_list(), ppl["first"].to_list(), ppl["last"].to_list())}
            _state.update({"fit": f, "hand": hand, "last_lineup": last_lineup, "park_of": park_of, "names": names,
                           "at": time.time()})
            logger.info(f"⚾ MLB v2: modelo ajustado (dispersión {f.dispersion:.1f})")
        return _state


def _platoon(lineup: str, sp: str, hand: Dict) -> float:
    th = (hand.get(sp) or {}).get("throws")
    if th not in ("L", "R"):
        return 0.0
    n = sum(1 for b in lineup.split(";") if (hand.get(b) or {}).get("bats") in ("S",) or
            ((hand.get(b) or {}).get("bats") not in (None, th)))
    return n / 9.0


def predict_games_sync(games: List[Tuple[int, str, str, date]]) -> Dict[int, Dict]:
    """[(id, local, visita (nombres o abreviaturas ESPN), fecha)] → {id: resultado}."""
    st = _state_now()
    f, hand = st["fit"], st["hand"]
    by_day: Dict[date, List[Dict]] = {}
    out = {}
    for gid, home, away, day, h_abbr, a_abbr in games:
        if day not in by_day:
            by_day[day] = statsapi_games(day.isoformat(), day.isoformat())
        h, a = retro_team(h_abbr), retro_team(a_abbr)
        sg = next((g for g in by_day[day] if g["home"] == h and g["away"] == a), None)
        if sg is None:
            logger.warning(f"MLB v2: {away} @ {home} ({day}) no está en la StatsAPI")
            continue
        season = _season_now(day)
        info = {}
        rows = {}
        for side, opp in (("home", "away"), ("away", "home")):
            lineup = sg[f"{side}_lineup"] if sg[f"{side}_lineup"].count(";") == 8 else st["last_lineup"].get(sg[side], "")
            sp = sg[f"{opp}_sp_id"]
            rows[side] = {"is_home": 1.0 if side == "home" else 0.0, "night": 1.0 if sg["day_night"] == "N" else 0.0,
                          "park": st["park_of"].get(sg["home"], ""), "lineup": lineup, "sp": sp, "pit": sg[opp], "bat": sg[side],
                          "season": season, "platoon": _platoon(lineup, sp, hand), "fatigue": 0.0}
            info[f"{side}_lineup_official"] = sg[f"{side}_lineup"].count(";") == 8
            info[f"{side}_starter"] = sg[f"{side}_sp"]
            ix = f.index
            sp_own = sg[f"{side}_sp_id"]
            info[f"{side}_starter_rating"] = round(float(f.beta[ix[("sp", sp_own)]]), 3) if ("sp", sp_own) in ix else None
        lh, la = f.rate(rows["home"]), f.rate(rows["away"])
        g = game_probs(lh, la, f.dispersion, run_line=-1.5, total=8.5)
        k = np.arange(31)
        out[gid] = {
            "home_win_prob": round(g["home_win"], 4), "away_win_prob": round(1 - g["home_win"], 4), "draw_prob": None,
            "predicted_home_score": round(lh, 2), "predicted_away_score": round(la, 2),
            "predicted_total": round(lh + la, 2), "predicted_spread": round(la - lh, 2),
            "confidence": round(abs(g["home_win"] - 0.5) * 2, 3),
            "simulation_details": {"model": MODEL_NAME, "expected_runs": {"home": round(lh, 2), "away": round(la, 2)},
                                   "dispersion": round(f.dispersion, 2), "park": rows["home"]["park"],
                                   "platoon": {"home": round(rows["home"]["platoon"], 2), "away": round(rows["away"]["platoon"], 2)},
                                   **info,
                                   "note": "Backtest 2015–2021 sin ventaja contra el cierre: picks por confianza del modelo."},
        }
    return out


def mlb_picks(pred, odds_summary: Optional[Dict], home_name: str, away_name: str, home_abbr: str, away_abbr: str,
              home_runs: Optional[float] = None, away_runs: Optional[float] = None) -> Optional[Dict]:
    """Ganador, run line y total con la probabilidad del modelo y su confianza (mismo formato que NBA/NFL)."""
    from app.services.nba_tracking import market_picks
    sd = getattr(pred, "simulation_details", None) or {}
    if sd.get("model") != MODEL_NAME:
        return None
    o = odds_summary or {}
    sp, tt = o.get("spread") or {}, o.get("total") or {}
    xr = sd.get("expected_runs") or {}
    lh, la, r = float(xr.get("home", 4.4)), float(xr.get("away", 4.4)), float(sd.get("dispersion", 4.4))
    line = sp.get("home_line")
    g = game_probs(lh, la, r, run_line=line if line is not None else -1.5,
                   total=tt.get("line") if tt.get("line") is not None else None)
    return market_picks(pred.home_win_prob, lh - la, lh + la,
                        None if line is None else -line, tt.get("line"),
                        (o.get("best_home_ml") or {}).get("odds"), (o.get("best_away_ml") or {}).get("odds"),
                        sp.get("home_odds"), sp.get("away_odds"), tt.get("over_odds"), tt.get("under_odds"),
                        None if home_runs is None else int(home_runs), None if away_runs is None else int(away_runs),
                        home_abbr, away_abbr, p_home_cover=g.get("home_cover") if line is not None else None,
                        p_over=g.get("over"), winner_recommendable=False,
                        p_home_cover_raw=g.get("home_cover") if line is not None else None, p_over_raw=g.get("over"))
