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
    return {"market_margin": margin, "market_total": total, "market_provider": odds.get("provider")}


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
        h, a = ids.get(r.home_team.lower()), ids.get(r.away_team.lower())
        score = None
        if h and a:
            # Misma fecha exacta (ET en ESPN y en nba_api): hay series de 2 partidos seguidos en la misma sede
            res = (await db.execute(text(
                "SELECT home_pts, away_pts FROM nba_games WHERE home_team_id = :h AND away_team_id = :a "
                "AND game_date = :d AND home_pts IS NOT NULL LIMIT 1"),
                {"h": h, "a": a, "d": r.game_date})).first()
            if res:
                score = (int(res[0]), int(res[1]))
        if score is None:
            g = (await db.execute(select(Game).where(Game.id == r.game_id))).scalar_one_or_none()
            if g is not None and g.status == GameStatus.FINISHED and g.home_score is not None and g.away_score is not None:
                score = (int(g.home_score), int(g.away_score))
        if score is None:
            continue
        r.home_pts, r.away_pts, r.settled_at = score[0], score[1], _utcnow()
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
            "mae_margin": round(mae_m, 2), "mae_total": None if mae_t is None else round(mae_t, 2), "vs_market": vs}


async def performance(db: AsyncSession, days: Optional[int] = None) -> Dict:
    q = select(NBAPredictionLog).where(NBAPredictionLog.settled_at.is_not(None))
    if days:
        q = q.where(NBAPredictionLog.game_date >= date.today() - timedelta(days=days))
    rows = (await db.execute(q.order_by(NBAPredictionLog.game_date.desc()))).scalars().all()
    by_model: Dict[str, List[NBAPredictionLog]] = {}
    for r in rows:
        by_model.setdefault(r.model_name, []).append(r)
    recent = [{
        "date": r.game_date.isoformat(), "matchup": f"{r.away_team} @ {r.home_team}",
        "home_win_prob": round(r.home_win_prob, 3), "pred_margin": round(r.pred_margin, 1),
        "market_margin": r.market_margin, "final": f"{r.away_pts}-{r.home_pts}",
        "margin": r.home_pts - r.away_pts, "hit": (r.pred_margin > 0) == (r.home_pts > r.away_pts),
    } for r in by_model.get("nba_players_v2", [])[:20]]
    from app.services.nba_game_model import ModelParams
    bt = ModelParams.load().metrics or {}
    backtest = {"games": bt.get("n"), "winner_pct": bt.get("acierto"), "logloss": bt.get("logloss"),
                "mae_margin": bt.get("MAE margen"), "mae_total": bt.get("MAE total")} if bt else None
    return {"models": {k: _summary(v) for k, v in by_model.items()}, "recent": recent, "backtest": backtest,
            "pending": (await db.execute(text("SELECT COUNT(*) FROM nba_prediction_log WHERE settled_at IS NULL AND model_name = 'nba_players_v2'"))).scalar()}
