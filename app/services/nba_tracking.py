"""
OmniPicks — NBA: medición en vivo del modelo.

- Cada vez que se predice un partido ANTES de que empiece, se guarda (o reemplaza) esa
  predicción en nba_prediction_log, junto con la línea de ESPN (DraftKings) de ese momento.
  Así la que queda es la última previa al partido, aunque la PC no esté prendida a la hora
  exacta.
- Al terminar los partidos, se liquidan con el marcador real (tabla nba_games de nba_api,
  o el marcador final de ESPN).
- `performance()` resume acierto, log loss, error del margen y del total, y la comparación
  contra el mercado: quién quedó más cerca del resultado.
"""
import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

from sqlalchemy import and_, select, text
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db_models import Game, GameStatus, NBAPredictionLog

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _market(game: Game) -> Dict:
    odds = (game.raw_data or {}).get("espn_odds") or {}
    spread = odds.get("spread")
    try:
        margin = None if spread is None else -float(spread)          # spread de ESPN = el del local
    except (TypeError, ValueError):
        margin = None
    try:
        total = None if odds.get("over_under") is None else float(odds.get("over_under"))
    except (TypeError, ValueError):
        total = None
    if odds.get("spread_home_line") is not None:
        margin = -float(odds["spread_home_line"])
    if odds.get("total_line") is not None:
        total = float(odds["total_line"])
    return {"market_margin": margin, "market_total": total, "market_provider": odds.get("provider"),
            "ml_home": odds.get("ml_home"), "ml_away": odds.get("ml_away"),
            "spread_home_odds": odds.get("spread_home_odds"), "spread_away_odds": odds.get("spread_away_odds"),
            "over_odds": odds.get("over_odds"), "under_odds": odds.get("under_odds")}


# ══════════════════════════════════════════════════════════════════════
# Qué recomienda el modelo en cada mercado (ganador, spread, total)
# ══════════════════════════════════════════════════════════════════════

DEFAULT_PRICE = -110.0


def _implied(american: Optional[float]) -> Optional[float]:
    if american is None:
        return None
    return 100.0 / (american + 100.0) if american > 0 else -american / (-american + 100.0)


def _decimal(american: float) -> float:
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / -american)


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def pick_tier(edge: Optional[float]) -> str:
    """Mismos niveles que las recomendaciones: recomendado 3-10 pts, revisar >10, sin valor <3."""
    if edge is None:
        return "sin_dato"
    if edge > 0.10:
        return "revisar"
    if edge >= 0.03:
        return "recomendado"
    return "sin_valor"


def _sigmas():
    from app.services.nba_game_model import ModelParams
    p = ModelParams.load()
    return p.sigma_margin or 13.8, p.sigma_total or 18.5


def market_picks(home_win_prob: float, pred_margin: Optional[float], pred_total: Optional[float],
                 market_margin: Optional[float], market_total: Optional[float],
                 ml_home=None, ml_away=None, sp_home_odds=None, sp_away_odds=None, over_odds=None, under_odds=None,
                 home_pts: Optional[int] = None, away_pts: Optional[int] = None,
                 home_abbr: str = "LOC", away_abbr: str = "VIS",
                 p_home_cover: Optional[float] = None, p_over: Optional[float] = None,
                 winner_recommendable: bool = True) -> Dict:
    """Lado que prefiere el modelo en cada mercado, con su probabilidad, la del casino y el valor.

    Si se dan los puntos finales, marca el resultado: "win", "loss" o "push".
    El spread se expresa como en el casino: "TOR +10.5" (línea del lado elegido).
    p_home_cover / p_over: probabilidades ya calculadas (NFL: con números clave y calibradas);
    si no vienen, se usa la normal de la NBA. winner_recommendable=False deja el ganador sin valor.
    """
    sd_m, sd_t = _sigmas()
    done = home_pts is not None and away_pts is not None
    margin = (home_pts - away_pts) if done else None
    out: Dict = {}

    # Ganador
    side = "home" if home_win_prob >= 0.5 else "away"
    prob = home_win_prob if side == "home" else 1.0 - home_win_prob
    price = ml_home if side == "home" else ml_away
    imp = _implied(price)
    out["winner"] = {"side": side, "label": home_abbr if side == "home" else away_abbr, "prob": round(prob, 3),
                     "odds": price, "implied": None if imp is None else round(imp, 3),
                     "edge": None if imp is None else round(prob - imp, 3),
                     "tier": pick_tier(None if imp is None else prob - imp) if winner_recommendable or imp is None else "sin_valor",
                     "result": None if not done else ("win" if (margin > 0) == (side == "home") and margin != 0 else "loss")}

    # Spread: el local cubre si margen + línea_local > 0 (línea_local = −market_margin)
    if pred_margin is not None and market_margin is not None:
        p_home = p_home_cover if p_home_cover is not None else _ncdf((pred_margin - market_margin) / sd_m)
        side = "home" if p_home >= 0.5 else "away"
        prob = p_home if side == "home" else 1.0 - p_home
        line_home = -market_margin
        line = line_home if side == "home" else -line_home
        price = (sp_home_odds if side == "home" else sp_away_odds)
        approx = price is None
        price = DEFAULT_PRICE if price is None else price
        imp = _implied(price)
        res = None
        if done:
            cover = margin + line_home
            res = "push" if cover == 0 else ("win" if (cover > 0) == (side == "home") else "loss")
        out["spread"] = {"side": side, "label": f"{home_abbr if side == 'home' else away_abbr} {line:+g}", "line": line,
                         "prob": round(prob, 3), "odds": price, "odds_approx": approx, "implied": round(imp, 3),
                         "edge": round(prob - imp, 3), "tier": pick_tier(prob - imp), "result": res}

    # Total
    if pred_total is not None and market_total is not None:
        p_over = p_over if p_over is not None else 1.0 - _ncdf((market_total - pred_total) / sd_t)
        side = "over" if p_over >= 0.5 else "under"
        prob = p_over if side == "over" else 1.0 - p_over
        price = over_odds if side == "over" else under_odds
        approx = price is None
        price = DEFAULT_PRICE if price is None else price
        imp = _implied(price)
        res = None
        if done:
            tot = home_pts + away_pts
            res = "push" if tot == market_total else ("win" if (tot > market_total) == (side == "over") else "loss")
        out["total"] = {"side": side, "label": f"{'Over' if side == 'over' else 'Under'} {market_total:g}", "line": market_total,
                        "prob": round(prob, 3), "odds": price, "odds_approx": approx, "implied": round(imp, 3),
                        "edge": round(prob - imp, 3), "tier": pick_tier(prob - imp), "result": res}
    return out


def log_picks(r: NBAPredictionLog) -> Dict:
    if r.model_name == "nfl_v2":
        from app.services.nfl_v2 import nfl_picks
        return nfl_picks(r.home_win_prob, r.pred_margin, r.pred_total, r.market_margin, r.market_total,
                         r.ml_home, r.ml_away, r.spread_home_odds, r.spread_away_odds, r.over_odds, r.under_odds,
                         r.home_pts, r.away_pts, r.home_team, r.away_team)
    from app.services.nba_lineups import team_ids_by_name  # noqa: F401  (nombres → abreviaturas, ver _abbr)
    return market_picks(r.home_win_prob, r.pred_margin, r.pred_total, r.market_margin, r.market_total,
                        r.ml_home, r.ml_away, r.spread_home_odds, r.spread_away_odds, r.over_odds, r.under_odds,
                        r.home_pts, r.away_pts, _abbr(r.home_team), _abbr(r.away_team))


_ABBR_CACHE: Dict[str, str] = {}


def _abbr(name: str) -> str:
    if not _ABBR_CACHE:
        from nba_api.stats.static import teams
        for t in teams.get_teams():
            _ABBR_CACHE[t["full_name"].lower()] = t["abbreviation"]
        _ABBR_CACHE["la clippers"] = "LAC"
    return _ABBR_CACHE.get((name or "").lower(), (name or "?")[:3].upper())


async def log_prediction(db: AsyncSession, game: Game, model_name: str, pred: Dict):
    """Guarda la predicción si el partido todavía no empieza."""
    now = _utcnow()
    if game.game_time is not None and now >= game.game_time:
        return
    if game.status not in (None, GameStatus.SCHEDULED):
        return
    row = {
        "game_id": game.id, "model_name": model_name, "game_date": game.game_date, "game_time": game.game_time,
        "home_team": game.home_team_name, "away_team": game.away_team_name, "predicted_at": now,
        "minutes_before_tip": None if game.game_time is None else round((game.game_time - now).total_seconds() / 60.0, 1),
        "home_win_prob": float(pred["home_win_prob"]),
        "pred_margin": -float(pred["predicted_spread"]) if pred.get("predicted_spread") is not None else 0.0,
        "pred_total": pred.get("predicted_total"), **_market(game),
    }
    stmt = mysql_insert(NBAPredictionLog).values(row)
    update = {k: stmt.inserted[k] for k in row if k not in ("game_id", "model_name")}
    await db.execute(stmt.on_duplicate_key_update(**update))


async def settle(db: AsyncSession) -> int:
    """Llena el marcador de las predicciones cuyos partidos ya terminaron."""
    from app.services.nba_lineups import team_ids_by_name

    pending = (await db.execute(select(NBAPredictionLog).where(and_(
        NBAPredictionLog.settled_at.is_(None), NBAPredictionLog.game_date < date.today() + timedelta(days=1))))).scalars().all()
    if not pending:
        return 0
    ids = team_ids_by_name()
    done = 0
    now = _utcnow()
    for r in pending:
        if r.game_time is not None and now < r.game_time + timedelta(hours=2):
            continue                                        # todavía no termina
        # Fuente: el marcador FINAL de ESPN (estado "final"). nba_api puede publicar parciales.
        g = (await db.execute(select(Game).where(Game.id == r.game_id))).scalar_one_or_none()
        if g is None or g.status != GameStatus.FINISHED or g.home_score is None or g.away_score is None:
            continue
        r.home_pts, r.away_pts, r.settled_at = int(g.home_score), int(g.away_score), _utcnow()
        done += 1
    await db.commit()
    if done:
        logger.info(f"NBA: {done} predicciones liquidadas")
    return done


def _summary(rows: List[NBAPredictionLog]) -> Optional[Dict]:
    if not rows:
        return None
    n = len(rows)
    hits = sum(1 for r in rows if (r.pred_margin > 0) == (r.home_pts > r.away_pts))
    ll = 0.0
    for r in rows:
        p = min(max(r.home_win_prob, 1e-4), 1 - 1e-4)
        ll += -math.log(p if r.home_pts > r.away_pts else 1 - p)
    mae_m = sum(abs(r.pred_margin - (r.home_pts - r.away_pts)) for r in rows) / n
    tot = [r for r in rows if r.pred_total is not None]
    mae_t = sum(abs(r.pred_total - (r.home_pts + r.away_pts)) for r in tot) / len(tot) if tot else None
    mk = [r for r in rows if r.market_margin is not None]
    vs = None
    if mk:
        model_err = sum(abs(r.pred_margin - (r.home_pts - r.away_pts)) for r in mk) / len(mk)
        market_err = sum(abs(r.market_margin - (r.home_pts - r.away_pts)) for r in mk) / len(mk)
        closer = sum(1 for r in mk if abs(r.pred_margin - (r.home_pts - r.away_pts)) < abs(r.market_margin - (r.home_pts - r.away_pts)))
        # Contra el spread: el modelo "elige" el lado que cree que cubre
        ats = [r for r in mk if abs(r.pred_margin - r.market_margin) >= 0.5 and (r.home_pts - r.away_pts) != r.market_margin]
        ats_hits = sum(1 for r in ats if ((r.pred_margin > r.market_margin) == ((r.home_pts - r.away_pts) > r.market_margin)))
        mkt_hits = sum(1 for r in mk if (r.market_margin > 0) == (r.home_pts > r.away_pts))
        vs = {"games": len(mk), "model_mae_margin": round(model_err, 2), "market_mae_margin": round(market_err, 2),
              "model_closer_pct": round(closer / len(mk), 3), "market_winner_pct": round(mkt_hits / len(mk), 3),
              "ats_games": len(ats), "ats_hit_pct": round(ats_hits / len(ats), 3) if ats else None}
    return {"games": n, "winner_pct": round(hits / n, 3), "logloss": round(ll / n, 4),
            "mae_margin": round(mae_m, 2), "mae_total": None if mae_t is None else round(mae_t, 2), "vs_market": vs,
            "markets": _market_records(rows)}


def _market_records(rows: List[NBAPredictionLog]) -> Dict:
    """Récord de cada mercado (todos los picks y solo los recomendados) y ROI apostando 1 unidad."""
    out = {}
    picks = [log_picks(r) for r in rows]
    for m in ("winner", "spread", "total"):
        for scope in ("all", "recommended"):
            sel = [p[m] for p in picks if m in p and p[m]["result"] is not None
                   and (scope == "all" or p[m]["tier"] == "recomendado")]
            w = sum(1 for x in sel if x["result"] == "win")
            l = sum(1 for x in sel if x["result"] == "loss")
            pu = sum(1 for x in sel if x["result"] == "push")
            units = sum((_decimal(x["odds"]) - 1.0) if x["result"] == "win" else (-1.0 if x["result"] == "loss" else 0.0)
                        for x in sel if x.get("odds") is not None)
            staked = sum(1 for x in sel if x.get("odds") is not None and x["result"] != "push")
            out.setdefault(m, {})[scope] = {"w": w, "l": l, "p": pu,
                                            "pct": round(w / (w + l), 3) if w + l else None,
                                            "units": round(units, 2), "roi": round(units / staked, 3) if staked else None}
    rec = [p[m] for p in picks for m in ("winner", "spread", "total")
           if m in p and p[m]["result"] is not None and p[m]["tier"] == "recomendado"]
    w = sum(1 for x in rec if x["result"] == "win"); l = sum(1 for x in rec if x["result"] == "loss")
    units = sum((_decimal(x["odds"]) - 1.0) if x["result"] == "win" else (-1.0 if x["result"] == "loss" else 0.0)
                for x in rec if x.get("odds") is not None)
    staked = sum(1 for x in rec if x.get("odds") is not None and x["result"] != "push")
    out["recommended_total"] = {"w": w, "l": l, "pct": round(w / (w + l), 3) if w + l else None,
                                "units": round(units, 2), "roi": round(units / staked, 3) if staked else None}
    return out


SPORT_MODELS = {"nba": ("nba_players_v2", ("nba_players_v2", "nba_possessions_v1")),
                "nfl": ("nfl_v2", ("nfl_v2",))}


async def performance(db: AsyncSession, days: Optional[int] = None, sport: str = "nba") -> Dict:
    """Rendimiento en vivo de un deporte: predicción previa a cada partido contra el resultado y el mercado."""
    main, models = SPORT_MODELS[sport]
    q = select(NBAPredictionLog).where(and_(NBAPredictionLog.settled_at.is_not(None),
                                            NBAPredictionLog.model_name.in_(models)))
    if days:
        q = q.where(NBAPredictionLog.game_date >= date.today() - timedelta(days=days))
    rows = (await db.execute(q.order_by(NBAPredictionLog.game_date.desc()))).scalars().all()
    if sport == "nfl":
        from app.services.nfl_v2 import abbr as abbr_fn
    else:
        abbr_fn = _abbr
    by_model: Dict[str, List[NBAPredictionLog]] = {}
    for r in rows:
        by_model.setdefault(r.model_name, []).append(r)
    recent = [{
        "date": r.game_date.isoformat(), "matchup": f"{r.away_team} @ {r.home_team}",
        "away_abbr": abbr_fn(r.away_team), "home_abbr": abbr_fn(r.home_team),
        "home_win_prob": round(r.home_win_prob, 3), "pred_margin": round(r.pred_margin, 1),
        "pred_total": None if r.pred_total is None else round(r.pred_total, 1),
        "market_margin": r.market_margin, "market_total": r.market_total, "final": f"{r.away_pts}-{r.home_pts}",
        "margin": r.home_pts - r.away_pts, "hit": (r.pred_margin > 0) == (r.home_pts > r.away_pts),
        "picks": log_picks(r),
    } for r in by_model.get(main, [])[:200]]
    if sport == "nfl":
        from app.services.nfl_v2 import load_params
        bt = (load_params() or {}).get("metrics") or {}
        backtest = {"games": bt.get("games"), "winner_pct": bt.get("winner_pct"), "logloss": None,
                    "mae_margin": bt.get("mae_margin"), "mae_total": bt.get("mae_total"),
                    "mae_market_margin": bt.get("mae_market_margin")} if bt else None
    else:
        from app.services.nba_game_model import ModelParams
        bt = ModelParams.load().metrics or {}
        backtest = {"games": bt.get("n"), "winner_pct": bt.get("acierto"), "logloss": bt.get("logloss"),
                    "mae_margin": bt.get("MAE margen"), "mae_total": bt.get("MAE total")} if bt else None
    pending = (await db.execute(text("SELECT COUNT(*) FROM nba_prediction_log WHERE settled_at IS NULL AND model_name = :m"),
                                {"m": main})).scalar()
    return {"sport": sport, "main_model": main, "models": {k: _summary(v) for k, v in by_model.items()},
            "recent": recent, "backtest": backtest, "pending": pending}
