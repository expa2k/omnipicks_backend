"""
OmniPicks — Predictions API Routes
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from datetime import date, datetime, timedelta, timezone
from typing import Optional, List

from app.database import get_db
from app.db_models import (
    Prediction, Pick, Game, OddsSnapshot, SportType, PickResult, GameStatus, MarketType
)

router = APIRouter()


# ── Fechas en hora del Este de EE. UU. (la de la cartelera de NBA, NFL y MLB) ──
from zoneinfo import ZoneInfo
_ET = ZoneInfo("America/New_York")


def et_today() -> date:
    return datetime.now(_ET).date()


def et_bounds_utc(d: date):
    """Inicio y fin del día `d` en hora del Este, como UTC sin zona (así se guarda game_time)."""
    start = datetime(d.year, d.month, d.day, tzinfo=_ET).astimezone(timezone.utc).replace(tzinfo=None)
    return start, start + timedelta(days=1)


async def current_slate_date(db: AsyncSession, sport: Optional[str] = None) -> date:
    """El día que se muestra por defecto: hoy (ET) si quedan partidos por empezar; si no, el
    siguiente día con partidos programados (hasta 3 días adelante)."""
    now = datetime.utcnow()
    for k in range(4):
        d = et_today() + timedelta(days=k)
        start, end = et_bounds_utc(d)
        q = select(func.count(Game.id)).where(and_(Game.game_time >= max(start, now), Game.game_time < end,
                                                    Game.status == GameStatus.SCHEDULED))
        if sport:
            q = q.where(Game.sport == sport)
        if (await db.execute(q)).scalar():
            return d
    return et_today()


def pick_tier(edge: Optional[float]) -> str:
    """Qué tan en serio tomar una diferencia con el mercado (en puntos de probabilidad).
    - recomendado: 3 a 10 puntos de ventaja
    - revisar: más de 10; casi siempre es información que el modelo no tiene (bajas, rotaciones)
    - leve: menos de 3"""
    e = edge or 0.0
    if e > 0.10:
        return "revisar"
    if e >= 0.03:
        return "recomendado"
    return "leve"


@router.get("/today")
async def get_today_predictions(
    sport: Optional[str] = Query(None, description="Filter by sport: nba, nfl, mlb, soccer"),
    db: AsyncSession = Depends(get_db),
):
    """Get all predictions for today's games."""
    today = date.today()
    query = select(Prediction).where(
        func.date(Prediction.created_at) == today
    )
    if sport:
        query = query.where(Prediction.sport == sport)
    query = query.order_by(Prediction.confidence.desc())

    result = await db.execute(query)
    predictions = result.scalars().all()

    return {
        "date": today.isoformat(),
        "count": len(predictions),
        "predictions": [
            {
                "id": p.id,
                "game_id": p.game_id,
                "sport": p.sport.value if p.sport else None,
                "model_name": p.model_name,
                "home_win_prob": round(p.home_win_prob, 4),
                "away_win_prob": round(p.away_win_prob, 4),
                "draw_prob": round(p.draw_prob, 4) if p.draw_prob else None,
                "predicted_home_score": p.predicted_home_score,
                "predicted_away_score": p.predicted_away_score,
                "predicted_total": p.predicted_total,
                "predicted_spread": p.predicted_spread,
                "confidence": round(p.confidence, 4) if p.confidence else None,
                "feature_importances": p.feature_importances,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in predictions
        ],
    }


@router.get("/games")
async def get_game_predictions(
    sport: Optional[str] = Query(None, description="Filter by sport: nba, nfl, mlb, soccer"),
    game_date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
    include_finished: bool = Query(False, description="Incluir los partidos ya empezados o terminados de ese día"),
    db: AsyncSession = Depends(get_db),
):
    """
    CARTELERA COMPLETA: Returns ALL games for the day with predictions, odds, and value picks.
    The mathematical model is the protagonist. Casino odds are secondary.
    """
    if game_date:
        target_date = date.fromisoformat(game_date)
    elif include_finished:
        target_date = et_today()
    else:
        target_date = await current_slate_date(db, sport.upper() if sport else None)

    # 1. Get ALL games for the date using exact ET day boundaries in UTC
    start_utc, end_utc = et_bounds_utc(target_date)
    if include_finished and not game_date:
        # Terminados recientes: de ayer (ET) hasta ahora, para que a medianoche no se pierdan los de anoche
        start_utc = et_bounds_utc(target_date - timedelta(days=1))[0]
        end_utc = datetime.utcnow()
    
    games_query = select(Game).where(and_(Game.game_time >= start_utc, Game.game_time <= end_utc))
    if include_finished:
        games_query = games_query.where(Game.status != GameStatus.CANCELLED)
    else:
        games_query = games_query.where(and_(Game.status == GameStatus.SCHEDULED, Game.game_time > datetime.utcnow()))
    if sport:
        games_query = games_query.where(Game.sport == sport)
    # For MLB, only show actual MLB league games (not minor leagues like PCL, IL, LMB)
    if sport and sport.upper() == "MLB":
        games_query = games_query.where(Game.league == "MLB")
    games_query = games_query.order_by(Game.game_time)

    games_result = await db.execute(games_query)
    games = games_result.scalars().all()

    if not games:
        return {
            "date": target_date.isoformat(),
            "is_today": target_date == et_today(),
            "sport": sport.upper() if sport else "ALL",
            "count": 0,
            "games": [],
        }

    game_ids = [g.id for g in games]

    # 2. Get ALL predictions for these games (batch query)
    preds_result = await db.execute(
        select(Prediction).where(Prediction.game_id.in_(game_ids))
        .order_by(Prediction.created_at.desc())
    )
    all_preds = preds_result.scalars().all()
    # Keep latest prediction per game
    preds_map = {}
    for p in all_preds:
        if p.game_id not in preds_map:
            preds_map[p.game_id] = p

    # 3. Get ALL odds for these games (batch query, latest per sportsbook)
    odds_result = await db.execute(
        select(OddsSnapshot).where(OddsSnapshot.game_id.in_(game_ids))
        .order_by(OddsSnapshot.snapshot_time.desc())
    )
    all_odds = odds_result.scalars().all()
    odds_map = {}
    for o in all_odds:
        if o.game_id not in odds_map:
            odds_map[o.game_id] = []
        # Only keep latest per sportsbook per market
        existing = [(x.sportsbook, x.market) for x in odds_map[o.game_id]]
        if (o.sportsbook, o.market) not in existing:
            odds_map[o.game_id].append(o)

    # 4. Get ALL value picks for these games (batch query)
    picks_result = await db.execute(
        select(Pick).where(Pick.game_id.in_(game_ids))
        .order_by(Pick.expected_value.desc())
    )
    all_picks = picks_result.scalars().all()
    picks_map = {}
    for pk in all_picks:
        if pk.game_id not in picks_map:
            picks_map[pk.game_id] = []
        picks_map[pk.game_id].append(pk)

    # Lo que recomendó el modelo antes del partido (registro congelado), para los ya empezados
    from app.db_models import NBAPredictionLog
    logs = {r.game_id: r for r in (await db.execute(select(NBAPredictionLog).where(and_(
        NBAPredictionLog.game_id.in_(game_ids), NBAPredictionLog.model_name.in_(("nba_players_v2", "nfl_v2", "soccer_v2")))))).scalars().all()}

    # Logos (los manda ESPN; se guardan en la tabla de equipos)
    from app.db_models import Team
    team_ids = {g.home_team_id for g in games} | {g.away_team_id for g in games}
    logos = {t.id: t.logo_url for t in (await db.execute(select(Team).where(Team.id.in_(team_ids)))).scalars().all()}

    # 5. Build response
    response_games = []
    for g in games:
        pred = preds_map.get(g.id)
        odds_list = odds_map.get(g.id, [])
        game_picks = picks_map.get(g.id, [])

        odds_summary = _odds_summary(odds_list)

        game_obj = {
            "id": g.id,
            "sport": g.sport.value if g.sport else None,
            "league": g.league,
            "home_team": g.home_team_name,
            "away_team": g.away_team_name,
            "game_time": g.game_time.isoformat() + "Z" if g.game_time else None,
            "venue": g.venue,
            "status": g.status.value if g.status else None,
            "home_score": g.home_score,
            "away_score": g.away_score,
            "home_logo": logos.get(g.home_team_id) or None,
            "away_logo": logos.get(g.away_team_id) or None,
            "result": _result(g, pred, _frozen(g, logs.get(g.id))),
            "model_picks": _model_picks(g, pred, odds_summary, logs.get(g.id)),
            "lineups": g.raw_data.get("lineups") if g.raw_data and isinstance(g.raw_data, dict) else None,
            # Prediction (protagonist)
            "prediction": {
                "home_win_prob": round(pred.home_win_prob, 4),
                "away_win_prob": round(pred.away_win_prob, 4),
                "draw_prob": round(pred.draw_prob, 4) if pred.draw_prob else None,
                "predicted_home_score": pred.predicted_home_score,
                "predicted_away_score": pred.predicted_away_score,
                "predicted_total": pred.predicted_total,
                "predicted_spread": pred.predicted_spread,
                "confidence": round(pred.confidence, 4) if pred.confidence else None,
                "model_name": pred.model_name,
                "matchup_factors": pred.matchup_factors,
                "simulation_details": pred.simulation_details,
            } if pred else None,
            # Odds (secondary)
            "odds": odds_summary,
            # Value picks (optional list)
            "value_picks": [{
                "id": pk.id,
                "pick_description": pk.pick_description,
                "model_edge": round(pk.model_edge, 4),
                "expected_value": round(pk.expected_value, 4),
                "value_rating": pk.value_rating,
                "best_odds": pk.best_odds,
                "best_sportsbook": pk.best_sportsbook,
                "market": pk.market.value if pk.market else None,
                "model_probability": round(pk.model_probability, 4) if pk.model_probability is not None else None,
                "implied_probability": round(pk.implied_probability, 4) if pk.implied_probability is not None else None,
                "tier": pick_tier(pk.model_edge),
            } for pk in game_picks] if game_picks else [],
        }
        fz = _frozen(g, logs.get(g.id))
        if fz and game_obj["prediction"]:
            # Partido ya empezado: se muestra la predicción congelada antes del inicio (la que se mide)
            game_obj["prediction"].update(fz)
        response_games.append(game_obj)

    return {
        "date": target_date.isoformat(),
        "is_today": target_date == et_today(),
        "sport": sport.upper() if sport else "ALL",
        "count": len(response_games),
        "games": response_games,
    }



@router.get("/picks")
async def get_value_picks(
    sport: Optional[str] = Query(None),
    min_value: Optional[int] = Query(None, description="Minimum value rating (1-5)"),
    min_edge: Optional[float] = Query(None, description="Minimum edge percentage"),
    game_date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
    db: AsyncSession = Depends(get_db),
):
    query = select(Pick, Prediction).join(Game, Pick.game_id == Game.id).join(Prediction, Pick.prediction_id == Prediction.id)
    
    if game_date:
        target_date = date.fromisoformat(game_date)
        query = query.where(Pick.game_date == target_date)
    else:
        target_date = await current_slate_date(db)
        start_utc, end_utc = et_bounds_utc(target_date)
        
        query = query.where(and_(Game.game_time >= start_utc, Game.game_time < end_utc))

    if sport:
        query = query.where(Pick.sport == sport)
    if min_value:
        query = query.where(Pick.value_rating >= min_value)
    if min_edge:
        query = query.where(Pick.model_edge >= min_edge / 100)

    # Only show picks for games that haven't started yet
    now_utc = datetime.utcnow()
    query = query.where(Game.game_time > now_utc)

    query = query.order_by(Pick.value_rating.desc(), Pick.expected_value.desc())
    result = await db.execute(query)
    rows = result.all()
    picks = [r.Pick for r in rows]
    predictions_map = {r.Pick.id: r.Prediction for r in rows}

    # Also fetch associated games
    game_ids = list(set(p.game_id for p in picks))
    games_map = {}
    if game_ids:
        games_result = await db.execute(
            select(Game).where(Game.id.in_(game_ids))
        )
        for g in games_result.scalars().all():
            games_map[g.id] = g

    return {
        "date": target_date.isoformat() if target_date else "ALL",
        "count": len(picks),
        "picks": [
            {
                "id": p.id,
                "game_id": p.game_id,
                "sport": p.sport.value if p.sport else None,
                "market": p.market.value if p.market else None,
                "pick_type": p.pick_description, # Force UI to show description instead of 'home_ml'
                "pick_description": p.pick_description,
                "model_probability": round(p.model_probability, 4),
                "model_edge": round(p.model_edge, 4),
                "expected_value": round(p.expected_value, 4),
                "best_sportsbook": p.best_sportsbook,
                "best_odds": p.best_odds,
                "best_odds_decimal": round(p.best_odds_decimal, 3),
                "implied_probability": round(p.implied_probability, 4),
                "kelly_fraction": round(p.kelly_fraction, 4) if p.kelly_fraction else None,
                "recommended_stake_pct": round(p.recommended_stake_pct, 2) if p.recommended_stake_pct else None,
                "value_rating": p.value_rating,
                "result": p.result.value if p.result else None,
                "profit_loss": p.profit_loss,
                "matchup": f"{games_map[p.game_id].away_team_name} @ {games_map[p.game_id].home_team_name}" if p.game_id in games_map else "Desconocido",
                "game": {
                    "home_team": games_map[p.game_id].home_team_name,
                    "away_team": games_map[p.game_id].away_team_name,
                    "game_time": games_map[p.game_id].game_time.isoformat() + "Z" if games_map[p.game_id].game_time else None,
                    "venue": games_map[p.game_id].venue,
                    "status": games_map[p.game_id].status.value if games_map[p.game_id].status else None,
                } if p.game_id in games_map else None,
                "insights": {
                    "matchup_factors": predictions_map[p.id].matchup_factors if p.id in predictions_map else None,
                    "simulation_details": predictions_map[p.id].simulation_details if p.id in predictions_map else None,
                }
            }
            for p in picks
        ],
    }


@router.get("/dashboard")
async def get_dashboard(
    db: AsyncSession = Depends(get_db),
):
    """Dashboard summary: today's stats across all sports (mismo día que la cartelera)."""
    today = await current_slate_date(db)

    # Count picks by sport
    picks_query = select(
        Pick.sport,
        func.count(Pick.id).label("total"),
        func.sum(func.IF(Pick.value_rating >= 3, 1, 0)).label("high_value"),
    ).where(Pick.game_date == today).group_by(Pick.sport)

    picks_result = await db.execute(picks_query)
    picks_by_sport = {
        row.sport.value: {"total": row.total, "high_value": int(row.high_value or 0)}
        for row in picks_result
    }

    # Count games by sport
    games_query = select(
        Game.sport,
        func.count(Game.id).label("total"),
    ).where(and_(Game.game_date == today, Game.status == GameStatus.SCHEDULED,
                 Game.game_time > datetime.utcnow())).group_by(Game.sport)

    games_result = await db.execute(games_query)
    games_by_sport = {
        row.sport.value: row.total
        for row in games_result
    }

    # Recent performance (last 7 days)
    week_ago = today - timedelta(days=7)
    perf_query = select(
        Pick.sport,
        func.count(Pick.id).label("total"),
        func.sum(func.IF(Pick.result == PickResult.WIN, 1, 0)).label("wins"),
        func.sum(func.IF(Pick.result == PickResult.LOSS, 1, 0)).label("losses"),
        func.sum(Pick.profit_loss).label("total_pl"),
    ).where(
        and_(
            Pick.game_date >= week_ago,
            Pick.result != PickResult.PENDING,
        )
    ).group_by(Pick.sport)

    perf_result = await db.execute(perf_query)
    performance = {}
    for row in perf_result:
        wins = int(row.wins or 0)
        total = row.total
        performance[row.sport.value] = {
            "total_bets": total,
            "wins": wins,
            "losses": int(row.losses or 0),
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            "profit_loss": round(float(row.total_pl or 0), 2),
        }
    
    top_pick = None
    top_pick_game = None
    # Destacado: la mejor diferencia "recomendada" (3-10 pts); las enormes suelen ser información que falta
    top_pick_result = await db.execute(
        select(Pick).where(and_(Pick.game_date == today, Pick.expected_value > 0,
                                Pick.model_edge >= 0.03, Pick.model_edge <= 0.10))
        .order_by(Pick.expected_value.desc()).limit(1)
    )
    tp = top_pick_result.scalar_one_or_none()
    if tp:
        top_pick = tp
        game_result = await db.execute(select(Game).where(Game.id == tp.game_id))
        top_pick_game = game_result.scalar_one_or_none()

    return {
        "date": today.isoformat(),
        "is_today": today == et_today(),
        "games_today": games_by_sport,
        "picks_today": picks_by_sport,
        "weekly_performance": performance,
        "top_pick": {
            "id": top_pick.id,
            "sport": top_pick.sport.value,
            "pick_description": top_pick.pick_description,
            "value_rating": top_pick.value_rating,
            "expected_value": round(top_pick.expected_value, 4),
            "best_odds": top_pick.best_odds,
            "best_sportsbook": top_pick.best_sportsbook,
            "model_edge": round(top_pick.model_edge, 4) if top_pick.model_edge is not None else None,
            "model_probability": round(top_pick.model_probability, 4) if top_pick.model_probability is not None else None,
            "implied_probability": round(top_pick.implied_probability, 4) if top_pick.implied_probability is not None else None,
            "tier": pick_tier(top_pick.model_edge),
            "matchup": f"{top_pick_game.away_team_name} @ {top_pick_game.home_team_name}" if top_pick_game else "Desconocido"
        } if top_pick else None,
    }


@router.get("/history")
async def get_prediction_history(
    sport: Optional[str] = Query(None),
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    """Get historical prediction results for performance tracking."""
    start_date = date.today() - timedelta(days=days)

    query = select(Pick).where(
        and_(
            Pick.game_date >= start_date,
            Pick.result != PickResult.PENDING,
        )
    )
    if sport:
        query = query.where(Pick.sport == sport)

    query = query.order_by(Pick.game_date.desc())
    result = await db.execute(query)
    picks = result.scalars().all()

    # Calculate stats
    total = len(picks)
    wins = sum(1 for p in picks if p.result == PickResult.WIN)
    losses = sum(1 for p in picks if p.result == PickResult.LOSS)
    total_pl = sum(p.profit_loss or 0 for p in picks)

    return {
        "period_start": start_date.isoformat(),
        "period_end": date.today().isoformat(),
        "summary": {
            "total_picks": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            "total_profit_loss": round(total_pl, 2),
            "roi": round(total_pl / total * 100, 2) if total > 0 else 0,
        },
        "picks": [
            {
                "id": p.id,
                "sport": p.sport.value if p.sport else None,
                "pick_description": p.pick_description,
                "value_rating": p.value_rating,
                "model_edge": round(p.model_edge, 4),
                "best_odds": p.best_odds,
                "result": p.result.value if p.result else None,
                "profit_loss": p.profit_loss,
                "game_date": p.game_date.isoformat(),
            }
            for p in picks
        ],
    }


@router.get("/nba/performance")
async def get_nba_performance(
    days: Optional[int] = Query(None, description="Solo los últimos N días"),
    db: AsyncSession = Depends(get_db),
):
    """Rendimiento en vivo del modelo NBA: predicción previa a cada partido contra el resultado y el mercado."""
    from app.services.nba_tracking import performance, settle
    await settle(db)
    return await performance(db, days)


@router.get("/soccer/performance")
async def get_soccer_performance(
    days: Optional[int] = Query(None, description="Solo los últimos N días"),
    db: AsyncSession = Depends(get_db),
):
    """Rendimiento en vivo del fútbol v2: 1X2 previo a cada partido contra el resultado y el mercado."""
    from app.services.nba_tracking import performance, settle
    await settle(db)
    return await performance(db, days, sport="soccer")


@router.get("/nfl/performance")
async def get_nfl_performance(
    days: Optional[int] = Query(None, description="Solo los últimos N días"),
    db: AsyncSession = Depends(get_db),
):
    """Rendimiento en vivo del modelo NFL v2 (mismo formato que /nba/performance)."""
    from app.services.nba_tracking import performance, settle
    await settle(db)
    return await performance(db, days, sport="nfl")



def _dec(american: Optional[float]) -> float:
    if american is None:
        return 0.0
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / -american)


def _odds_summary(odds_list) -> Optional[dict]:
    """Mejor momio por lado (el que más paga), spread, total y favorito del mercado."""
    if not odds_list:
        return None
    by_market = {}
    for o in odds_list:
        by_market.setdefault(o.market, []).append(o)
    out = {"sportsbooks": sorted({o.sportsbook for o in odds_list}),
           "updated_at": max(o.snapshot_time for o in odds_list).isoformat()}      # hora local del servidor (NOW() de MySQL)
    ml = [o for o in by_market.get(MarketType.MONEYLINE, []) if o.home_odds is not None or o.away_odds is not None]
    if ml:
        bh = max((o for o in ml if o.home_odds is not None), key=lambda o: _dec(o.home_odds), default=None)
        ba = max((o for o in ml if o.away_odds is not None), key=lambda o: _dec(o.away_odds), default=None)
        bd = max((o for o in ml if o.draw_odds is not None), key=lambda o: _dec(o.draw_odds), default=None)
        out["best_home_ml"] = {"odds": bh.home_odds, "sportsbook": bh.sportsbook} if bh else None
        out["best_away_ml"] = {"odds": ba.away_odds, "sportsbook": ba.sportsbook} if ba else None
        out["best_draw"] = {"odds": bd.draw_odds, "sportsbook": bd.sportsbook} if bd else None
        if bh and ba:
            out["market_favorite"] = "home" if _dec(bh.home_odds) < _dec(ba.away_odds) else "away"
    sp = [o for o in by_market.get(MarketType.SPREAD, []) if o.spread_home is not None]
    if sp:
        o = next((x for x in sp if x.sportsbook == "DraftKings"), sp[0])
        out["spread"] = {"home_line": o.spread_home, "home_odds": o.spread_home_odds, "away_odds": o.spread_away_odds,
                         "sportsbook": o.sportsbook}
        out.setdefault("market_favorite", "home" if o.spread_home < 0 else "away" if o.spread_home > 0 else None)
    tt = [o for o in by_market.get(MarketType.OVER_UNDER, []) if o.total_line is not None]
    if tt:
        o = next((x for x in tt if x.sportsbook == "DraftKings"), tt[0])
        out["total"] = {"line": o.total_line, "over_odds": o.over_odds, "under_odds": o.under_odds, "sportsbook": o.sportsbook}
    return out



def _frozen(g, log) -> Optional[dict]:
    """Predicción previa al partido (del registro) si el partido ya empezó."""
    if log is None:
        return None
    started = g.status != GameStatus.SCHEDULED or (g.game_time is not None and g.game_time <= datetime.utcnow())
    if not started:
        return None
    m, t = log.pred_margin, log.pred_total
    out = {"home_win_prob": round(log.home_win_prob, 4), "away_win_prob": round(1 - log.home_win_prob, 4),
           "predicted_spread": round(-m, 1), "frozen": True, "frozen_at": log.predicted_at.isoformat() + "Z"}
    if t is not None:
        out.update({"predicted_total": round(t, 1), "predicted_home_score": round((t + m) / 2, 1),
                    "predicted_away_score": round((t - m) / 2, 1)})
    return out


def _result(g, pred, frozen: Optional[dict] = None) -> Optional[dict]:
    """Marcador final y si el modelo acertó al ganador (solo partidos terminados).
    Se mide con la predicción previa al partido cuando existe."""
    if g.status != GameStatus.FINISHED or g.home_score is None or g.away_score is None:
        return None
    out = {"home_score": g.home_score, "away_score": g.away_score,
           "winner": "home" if g.home_score > g.away_score else "away" if g.away_score > g.home_score else "draw"}
    if frozen is not None:
        out["model_pick"] = "home" if frozen["home_win_prob"] >= 0.5 else "away"
        out["hit"] = out["model_pick"] == out["winner"]
        out["margin_error"] = round(abs(-frozen["predicted_spread"] - (g.home_score - g.away_score)), 1)
        return out
    if pred is not None:
        probs = {"home": pred.home_win_prob or 0, "away": pred.away_win_prob or 0, "draw": pred.draw_prob or 0}
        out["model_pick"] = max(probs, key=probs.get)
        out["hit"] = out["model_pick"] == out["winner"]
        if pred.predicted_spread is not None:
            out["margin_error"] = round(abs(-pred.predicted_spread - (g.home_score - g.away_score)), 1)
    return out



def _model_picks(g, pred, odds_summary, log) -> Optional[dict]:
    """Lado que prefiere el modelo en ganador, spread y total, con su valor y, si terminó, el resultado.
    Para partidos ya empezados se usa la predicción congelada antes del inicio. NBA y NFL v2."""
    if g.sport == SportType.NBA:
        from app.services.nba_tracking import _abbr, market_picks

        def picker(*a, home_pts=None, away_pts=None):
            return market_picks(*a, home_pts, away_pts, _abbr(g.home_team_name), _abbr(g.away_team_name))
    elif g.sport == SportType.NFL and (log is not None or (pred is not None and pred.model_name == "nfl_v2")):
        from app.services.nfl_v2 import nfl_picks

        def picker(*a, home_pts=None, away_pts=None):
            return nfl_picks(*a, home_pts, away_pts, g.home_team_name, g.away_team_name)
    else:
        return None
    if log is not None and (g.status != GameStatus.SCHEDULED or (g.game_time and g.game_time <= datetime.utcnow())):
        hp, ap = log.home_pts, log.away_pts
        if hp is None and g.status == GameStatus.FINISHED:
            hp, ap = g.home_score, g.away_score          # ESPN ya tiene el final aunque el registro no se haya liquidado
        picks = picker(log.home_win_prob, log.pred_margin, log.pred_total, log.market_margin, log.market_total,
                       log.ml_home, log.ml_away, log.spread_home_odds, log.spread_away_odds,
                       log.over_odds, log.under_odds, home_pts=hp, away_pts=ap)
        return {**picks, "frozen": True}
    if pred is None or pred.predicted_spread is None:
        return None
    o = odds_summary or {}
    sp, tt = o.get("spread") or {}, o.get("total") or {}
    picks = picker(
        pred.home_win_prob, -pred.predicted_spread, pred.predicted_total,
        None if sp.get("home_line") is None else -sp["home_line"], tt.get("line"),
        (o.get("best_home_ml") or {}).get("odds"), (o.get("best_away_ml") or {}).get("odds"),
        sp.get("home_odds"), sp.get("away_odds"), tt.get("over_odds"), tt.get("under_odds"))
    return {**picks, "frozen": False}
