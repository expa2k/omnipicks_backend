"""
OmniPicks — Data Ingestion Orchestrator
Fetches data from external APIs and saves to the database.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import date, datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.db_models import Team, Game, OddsSnapshot, SportType, GameStatus, MarketType
from app.services.apisports_client import APISportsClient
from app.services.bzzoiro_client import BzzoiroClient
from app.services.odds_client import OddsClient
from app.utils.constants import ODDS_API_KEYS, ODDS_MARKETS_MAP, BZZOIRO_LEAGUES

logger = logging.getLogger(__name__)

class DataIngestionService:
    """Orchestrates pulling data from all APIs and saving to database."""

    odds_changed = 0

    def __init__(self, db: AsyncSession):
        self.db = db
        self.apisports = APISportsClient()
        self.bzzoiro = BzzoiroClient()
        self.odds_client = OddsClient()

    async def ingest_soccer_games(self, target_date: date) -> int:
        """Fetch and save soccer games from API-Sports for top leagues."""
        date_str = target_date.isoformat()
        
        # Valid top leagues and their API-Sports IDs
        VALID_LEAGUES = {
            39: "Premier League",
            140: "La Liga",
            135: "Serie A",
            78: "Bundesliga",
            61: "Ligue 1",
            262: "Liga MX",
            2: "UEFA Champions League",
            3: "UEFA Europa League",
            1: "World Cup",
            9: "Copa America",
            4: "Euro Championship",
        }
        
        games = await self.apisports.get_soccer_games(date_str)
        count = 0
        
        for event in games:
            league_data = event.get("league", {})
            league_id = league_data.get("id")
            
            # 1. Filter only top leagues we care about
            if league_id not in VALID_LEAGUES:
                continue
                
            league_name = VALID_LEAGUES[league_id]
            
            teams = event.get("teams", {})
            home_data = teams.get("home", {})
            away_data = teams.get("away", {})
            
            home_name = home_data.get("name", "Unknown")
            home_id_str = str(home_data.get("id", ""))
            home_logo = home_data.get("logo")
            
            away_name = away_data.get("name", "Unknown")
            away_id_str = str(away_data.get("id", ""))
            away_logo = away_data.get("logo")
            
            # Upsert teams
            home_team = await self._upsert_team(
                sport=SportType.SOCCER,
                external_id=home_id_str,
                name=home_name,
                league=league_name,
                logo_url=home_logo
            )
            
            away_team = await self._upsert_team(
                sport=SportType.SOCCER,
                external_id=away_id_str,
                name=away_name,
                league=league_name,
                logo_url=away_logo
            )
            
            # Status mapping from API-Sports
            status_short = event.get("fixture", {}).get("status", {}).get("short", "NS")
            status_map = {
                "NS": GameStatus.SCHEDULED,
                "TBD": GameStatus.SCHEDULED,
                "1H": GameStatus.LIVE,
                "HT": GameStatus.LIVE,
                "2H": GameStatus.LIVE,
                "ET": GameStatus.LIVE,
                "P": GameStatus.LIVE,
                "FT": GameStatus.FINISHED,
                "AET": GameStatus.FINISHED,
                "PEN": GameStatus.FINISHED,
                "SUSP": GameStatus.POSTPONED,
                "INT": GameStatus.LIVE,
                "PST": GameStatus.POSTPONED,
                "CANC": GameStatus.CANCELLED,
                "ABD": GameStatus.CANCELLED,
                "AWD": GameStatus.FINISHED,
                "WO": GameStatus.FINISHED,
            }
            game_status = status_map.get(status_short, GameStatus.SCHEDULED)
            
            fixture_id = str(event.get("fixture", {}).get("id", ""))
            game_time_utc = self._parse_datetime(event.get("fixture", {}).get("date"))
            season = str(league_data.get("season", ""))
            
            # Scores
            goals = event.get("goals", {})
            home_score = goals.get("home")
            away_score = goals.get("away")
            
            # Upsert game
            await self._upsert_game(
                sport=SportType.SOCCER,
                external_id=fixture_id,
                home_team=home_team,
                away_team=away_team,
                game_date=target_date,
                game_time=game_time_utc,
                league=league_name,
                season=season,
                status=game_status,
                home_score=home_score,
                away_score=away_score,
                raw_data=event
            )
            count += 1
            
        await self.db.commit()
        logger.info(f"Ingested {count} SOCCER games from API-Sports for {date_str}")
        return count

    async def ingest_us_sports_games(self, target_date: date, sport: SportType) -> int:
        """Fetch and save games for NBA, NFL, or MLB.
        NFL uses ESPN as primary (API-Sports can't access current season).
        NBA/MLB use API-Sports.
        """
        date_str = target_date.isoformat()

        if sport == SportType.NFL:
            # ESPN devuelve la semana completa; cada partido se guarda con su fecha real (hora ET)
            from zoneinfo import ZoneInfo
            from app.services.espn_nfl_client import get_nfl_games_espn
            games = await get_nfl_games_espn()

            count = 0
            for game_data in games:
                teams    = game_data.get("teams", {})
                home_data = teams.get("home", {})
                away_data = teams.get("away", {})
                game_info = game_data.get("game", {})

                home_team = await self._upsert_team(
                    sport=sport,
                    external_id=str(home_data.get("id")),
                    name=home_data.get("name", "Unknown"),
                    logo_url=home_data.get("logo"),
                    league="NFL"
                )
                away_team = await self._upsert_team(
                    sport=sport,
                    external_id=str(away_data.get("id")),
                    name=away_data.get("name", "Unknown"),
                    logo_url=away_data.get("logo"),
                    league="NFL"
                )

                game_iso = game_info.get("date", {}).get("time", "")
                game_time_utc = self._parse_datetime(game_iso)
                game_day = target_date
                if game_time_utc is not None:
                    from datetime import timezone
                    game_day = game_time_utc.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("America/New_York")).date()

                scores   = game_data.get("scores", {})
                home_score = scores.get("home", {}).get("total") if isinstance(scores.get("home"), dict) else None
                away_score = scores.get("away", {}).get("total") if isinstance(scores.get("away"), dict) else None

                nfl_game = await self._upsert_game(
                    sport=sport,
                    external_id=str(game_info.get("id")),
                    home_team=home_team,
                    away_team=away_team,
                    game_date=game_day,
                    game_time=game_time_utc,
                    league="NFL",
                    season=str(game_day.year if game_day.month >= 3 else game_day.year - 1),
                    status=self._map_us_status(game_info.get("status", {}).get("short", "NS")),
                    home_score=home_score,
                    away_score=away_score,
                    raw_data=game_data
                )
                await self.db.flush()
                self.odds_changed += await self._store_espn_odds(nfl_game, game_data.get("espn_odds"))
                count += 1

            await self.db.commit()
            logger.info(f"Ingested {count} NFL games from ESPN (semana actual)")
            return count

        if sport == SportType.NBA:
            return await self._ingest_nba_espn(target_date)

        # ── MLB via API-Sports ──────────────────────────────────────────────
        if sport == SportType.MLB:
            games = await self.apisports.get_mlb_games(date_str)
        else:
            return 0

        count = 0
        for game_data in games:
            # Extract basic structure common to API-Sports
            teams = game_data.get("teams", {})
            home_data = teams.get("home", {})
            away_data = teams.get("away", {})
            league_data = game_data.get("league", {})
            league_name = league_data.get("name") if isinstance(league_data, dict) else str(league_data)

            # Filter MLB to only include the actual MLB league, not minor leagues
            if sport == SportType.MLB and league_name != "MLB":
                continue

            # Upsert teams
            home_team = await self._upsert_team(
                sport=sport,
                external_id=str(home_data.get("id")),
                name=home_data.get("name", "Unknown"),
                logo_url=home_data.get("logo"),
                league=league_name
            )
            
            away_team = await self._upsert_team(
                sport=sport,
                external_id=str(away_data.get("id")),
                name=away_data.get("name", "Unknown"),
                logo_url=away_data.get("logo"),
                league=league_name
            )

            # Extract scores based on sport
            scores = game_data.get("scores", {})
            status_data = game_data.get("status", {})
            
            home_score = scores.get("home", {}).get("total") if isinstance(scores.get("home"), dict) else scores.get("home")
            away_score = scores.get("away", {}).get("total") if isinstance(scores.get("away"), dict) else scores.get("away")

            # Determine the correct local US date (Eastern Time = UTC-4)
            from datetime import timedelta
            game_time_utc = self._parse_datetime(game_data.get("date"))
            if game_time_utc:
                local_date = (game_time_utc - timedelta(hours=4)).date()
            else:
                local_date = target_date

            # Upsert game
            await self._upsert_game(
                sport=sport,
                external_id=str(game_data.get("id")),
                home_team=home_team,
                away_team=away_team,
                game_date=local_date,
                game_time=game_time_utc,
                league=league_name,
                season=str(league_data.get("season", "") if isinstance(league_data, dict) else ""),
                status=self._map_us_status(status_data.get("short")),
                home_score=home_score,
                away_score=away_score,
                raw_data=game_data
            )
            count += 1

        await self.db.commit()
        logger.info(f"Ingested {count} {sport.value} games for {date_str}")
        return count

    async def _ingest_nba_espn(self, target_date: date) -> int:
        """NBA desde ESPN (gratis). API-Sports devolvía la liga como "standard"
        y usa "visitors" en vez de "away", así que el filtro anterior descartaba todo."""
        from app.services.espn_nba_client import get_nba_games_espn

        games = await get_nba_games_espn(target_date)
        count = 0
        for g in games:
            home_data = g["teams"]["home"]
            away_data = g["teams"]["away"]
            if not home_data.get("id") or not away_data.get("id"):
                continue

            home_team = await self._upsert_team(
                sport=SportType.NBA, external_id=f"espn_{home_data['id']}",
                name=home_data["name"], short_name=home_data.get("abbreviation"),
                logo_url=home_data.get("logo"), league="NBA",
            )
            away_team = await self._upsert_team(
                sport=SportType.NBA, external_id=f"espn_{away_data['id']}",
                name=away_data["name"], short_name=away_data.get("abbreviation"),
                logo_url=away_data.get("logo"), league="NBA",
            )

            season_start = target_date.year if target_date.month >= 10 else target_date.year - 1
            game = await self._upsert_game(
                sport=SportType.NBA,
                external_id=f"espn_{g['id']}",
                home_team=home_team,
                away_team=away_team,
                game_date=target_date,           # ESPN `dates` ya es el día ET
                game_time=self._parse_datetime(g.get("date")),
                league="NBA",
                season=str(season_start),
                status=self._map_us_status(g["status"]["short"]),
                venue=g.get("venue", {}).get("name") or None,
                home_score=g["scores"]["home"]["total"],
                away_score=g["scores"]["away"]["total"],
                raw_data=g,
            )
            await self.db.flush()
            self.odds_changed += await self._store_espn_odds(game, g.get("espn_odds"))
            count += 1

        await self.db.commit()
        logger.info(f"Ingested {count} NBA games from ESPN for {target_date.isoformat()}")
        return count

    async def ingest_soccer_espn(self, target_date: date) -> int:
        """Fútbol v2: partidos de las 8 competencias de la app desde ESPN (gratis), con marcador y
        momios de DraftKings (1X2 y total). La fecha del partido es la del día ET de la patada inicial."""
        from zoneinfo import ZoneInfo
        from datetime import timezone
        from app.services.espn_nfl_client import SOCCER_V2_COMPETITIONS, get_soccer_games_espn, parse_soccer_event
        count = 0
        for slug, league in SOCCER_V2_COMPETITIONS.items():
            for ev in await get_soccer_games_espn(slug, target_date):
                g = parse_soccer_event(ev)
                if not g or not g["teams"]["home"]["id"] or not g["teams"]["away"]["id"]:
                    continue
                th, ta = g["teams"]["home"], g["teams"]["away"]
                home_team = await self._upsert_team(sport=SportType.SOCCER, external_id=f"espn_{th['id']}", name=th["name"],
                                                    short_name=th.get("abbreviation"), logo_url=th.get("logo"), league=league)
                away_team = await self._upsert_team(sport=SportType.SOCCER, external_id=f"espn_{ta['id']}", name=ta["name"],
                                                    short_name=ta.get("abbreviation"), logo_url=ta.get("logo"), league=league)
                kick = self._parse_datetime(g.get("date"))
                day = kick.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("America/New_York")).date() if kick else target_date
                game = await self._upsert_game(
                    sport=SportType.SOCCER, external_id=f"espn_{g['id']}", home_team=home_team, away_team=away_team,
                    game_date=day, game_time=kick, league=league, season=str(day.year if day.month >= 7 else day.year - 1),
                    status=self._map_us_status(g["status"]["short"]), venue=g["venue"]["name"] or None,
                    home_score=g["scores"]["home"]["total"], away_score=g["scores"]["away"]["total"], raw_data=g)
                await self.db.flush()
                self.odds_changed += await self._store_espn_odds(game, g.get("espn_odds"))
                count += 1
        await self.db.commit()
        logger.info(f"Fútbol (ESPN): {count} partidos para {target_date.isoformat()}")
        return count

    async def ingest_mlb_espn(self, target_date: date) -> int:
        """MLB v2: juegos del día desde ESPN (gratis), con marcador y momios de DraftKings (moneyline,
        run line y total). Mismo parser que el fútbol (estructura de ESPN igual)."""
        import httpx
        from app.services.espn_nfl_client import parse_soccer_event
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get("https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
                                     params={"dates": target_date.strftime("%Y%m%d")})
                events = r.json().get("events", []) if r.status_code == 200 else []
        except Exception as e:
            logger.warning(f"ESPN MLB: sin respuesta ({type(e).__name__}: {e})")
            return 0
        count = 0
        for ev in events:
            g = parse_soccer_event(ev)
            if not g or not g["teams"]["home"]["id"] or not g["teams"]["away"]["id"]:
                continue
            th, ta = g["teams"]["home"], g["teams"]["away"]
            home_team = await self._upsert_team(sport=SportType.MLB, external_id=f"espn_{th['id']}", name=th["name"],
                                                short_name=th.get("abbreviation"), logo_url=th.get("logo"), league="MLB")
            away_team = await self._upsert_team(sport=SportType.MLB, external_id=f"espn_{ta['id']}", name=ta["name"],
                                                short_name=ta.get("abbreviation"), logo_url=ta.get("logo"), league="MLB")
            game = await self._upsert_game(
                sport=SportType.MLB, external_id=f"espn_{g['id']}", home_team=home_team, away_team=away_team,
                game_date=target_date, game_time=self._parse_datetime(g.get("date")), league="MLB",
                season=str(target_date.year), status=self._map_us_status(g["status"]["short"]),
                venue=g["venue"]["name"] or None, home_score=g["scores"]["home"]["total"],
                away_score=g["scores"]["away"]["total"], raw_data=g)
            await self.db.flush()
            self.odds_changed += await self._store_espn_odds(game, g.get("espn_odds"))
            count += 1
        await self.db.commit()
        logger.info(f"MLB (ESPN): {count} juegos para {target_date.isoformat()}")
        return count

    async def _store_espn_odds(self, game: Game, eo: Optional[Dict[str, Any]]) -> int:
        """Guarda las líneas de DraftKings que trae ESPN (moneyline, spread y total).
        Solo inserta una foto nueva si la línea cambió desde la última."""
        if not eo or game.status not in (None, GameStatus.SCHEDULED):
            return 0
        book = "DraftKings"
        wanted = []
        if eo.get("ml_home") is not None and eo.get("ml_away") is not None:
            ml = {"home_odds": eo["ml_home"], "away_odds": eo["ml_away"]}
            if eo.get("ml_draw") is not None:
                ml["draw_odds"] = float(eo["ml_draw"])
            wanted.append((MarketType.MONEYLINE, ml))
        if eo.get("spread_home_line") is not None:
            wanted.append((MarketType.SPREAD, {"spread_home": eo["spread_home_line"], "spread_home_odds": eo.get("spread_home_odds"),
                                              "spread_away_odds": eo.get("spread_away_odds")}))
        if eo.get("total_line") is not None:
            wanted.append((MarketType.OVER_UNDER, {"total_line": eo["total_line"], "over_odds": eo.get("over_odds"),
                                                  "under_odds": eo.get("under_odds")}))
        added = 0
        for market, vals in wanted:
            last = (await self.db.execute(
                select(OddsSnapshot).where(and_(OddsSnapshot.game_id == game.id, OddsSnapshot.sportsbook == book,
                                                OddsSnapshot.market == market))
                .order_by(OddsSnapshot.snapshot_time.desc(), OddsSnapshot.id.desc()).limit(1))).scalar_one_or_none()
            if last is not None and all(getattr(last, k) == v for k, v in vals.items()):
                continue
            snap = OddsSnapshot(game_id=game.id, sport=game.sport, sportsbook=book, market=market, **vals)
            if market == MarketType.MONEYLINE:
                snap.implied_home_prob = _implied(vals["home_odds"])
                snap.implied_away_prob = _implied(vals["away_odds"])
            self.db.add(snap)
            added += 1
        return added

    async def ingest_daily_games(self, target_date: date):
        """Run all daily game ingestion for today only (to save API quotas)."""
        logger.info(f"Starting daily game ingestion for {target_date}")
        
        # Fetch today only
        await self.ingest_soccer_games(target_date)
        await self.ingest_us_sports_games(target_date, SportType.NBA)
        await self.ingest_us_sports_games(target_date, SportType.MLB)
        await self.ingest_us_sports_games(target_date, SportType.NFL)

    async def ingest_odds(self):
        """Fetch current odds from The Odds API for all supported sports."""
        logger.info("Starting odds ingestion")
        count = 0
        
        for sport, keys in ODDS_API_KEYS.items():
            for key in keys:
                try:
                    odds_data = await self.odds_client.get_odds(sport_key=key)
                    # Odds API gives a list of events with bookmakers
                    for event in odds_data:
                        # Attempt to find the matching game in our DB
                        # This requires matching team names since IDs differ across APIs
                        home_name = event.get("home_team")
                        away_name = event.get("away_team")
                        
                        game = await self._find_game_by_teams_approx(sport, home_name, away_name,
                                                                     self._parse_datetime(event.get("commence_time")))
                        if not game:
                            continue
                            
                        # Use DraftKings as the primary source of truth
                        dk_book = next((b for b in event.get("bookmakers", []) if b.get("title") == "DraftKings"), None)
                        if not dk_book and event.get("bookmakers"):
                            dk_book = event.get("bookmakers")[0]
                            
                        if not dk_book:
                            continue
                            
                        sportsbook = dk_book.get("title")
                        
                        for market in dk_book.get("markets", []):
                            market_key = market.get("key")
                            internal_market = ODDS_MARKETS_MAP.get(market_key)
                            
                            if not internal_market:
                                continue
                                
                            outcomes = market.get("outcomes", [])
                            
                            snapshot = OddsSnapshot(
                                game_id=game.id,
                                sport=sport,
                                sportsbook=sportsbook,
                                market=MarketType(internal_market),
                            )
                            
                            if internal_market == "moneyline":
                                for oc in outcomes:
                                    if oc.get("name") == home_name:
                                        snapshot.home_odds = oc.get("price")
                                    elif oc.get("name") == away_name:
                                        snapshot.away_odds = oc.get("price")
                                    elif oc.get("name").lower() == "draw":
                                        snapshot.draw_odds = oc.get("price")
                            
                            elif internal_market == "spread":
                                for oc in outcomes:
                                    if oc.get("name") == home_name:
                                        snapshot.spread_home = oc.get("point")
                                        snapshot.spread_home_odds = oc.get("price")
                                    elif oc.get("name") == away_name:
                                        snapshot.spread_away_odds = oc.get("price")

                            elif internal_market == "over_under":
                                for oc in outcomes:
                                    # Example name: "Over", "Under", point: 2.5
                                    if oc.get("name").lower() == "over":
                                        snapshot.over_odds = oc.get("price")
                                        snapshot.total_line = oc.get("point")
                                    elif oc.get("name").lower() == "under":
                                        snapshot.under_odds = oc.get("price")
                                        snapshot.total_line = oc.get("point")
                                        
                            elif internal_market == "btts":
                                for oc in outcomes:
                                    # name: "Yes", "No"
                                    if oc.get("name").lower() == "yes":
                                        # Map Yes to home_odds, No to away_odds for simplicity in DB, or use over/under.
                                        # Let's map "Yes" to over_odds and "No" to under_odds since it's a binary outcome
                                        snapshot.over_odds = oc.get("price")
                                    elif oc.get("name").lower() == "no":
                                        snapshot.under_odds = oc.get("price")
                                        
                            self.db.add(snapshot)
                            count += 1
                                
                except Exception as e:
                    logger.error(f"Error fetching odds for {key}: {e}")
                    
        await self.db.commit()
        logger.info(f"Ingested {count} new odds snapshots")

    # --- Helpers ---

    async def _upsert_team(self, sport: SportType, external_id: str, name: str, **kwargs) -> Team:
        result = await self.db.execute(
            select(Team).where(and_(Team.sport == sport, Team.external_id == external_id))
        )
        team = result.scalar_one_or_none()
        
        if not team:
            team = Team(sport=sport, external_id=external_id, name=name, **kwargs)
            self.db.add(team)
            await self.db.flush()
        else:
            if sport in (SportType.NFL, SportType.NBA) and name and name != "Unknown" and team.name != name:
                logger.info(f"Equipo {sport.value} {external_id}: nombre corregido '{team.name}' → '{name}'")
                team.name = name
            for k, v in kwargs.items():
                if v is not None:
                    setattr(team, k, v)
        return team

    async def _upsert_game(self, sport: SportType, external_id: str, home_team: Team, away_team: Team, **kwargs) -> Game:
        result = await self.db.execute(
            select(Game).where(and_(Game.sport == sport, Game.external_id == external_id))
        )
        game = result.scalar_one_or_none()
        
        if not game:
            game = Game(
                sport=sport,
                external_id=external_id,
                home_team_id=home_team.id,
                away_team_id=away_team.id,
                home_team_name=home_team.name,
                away_team_name=away_team.name,
                **kwargs
            )
            self.db.add(game)
        else:
            game.home_team_id, game.away_team_id = home_team.id, away_team.id
            game.home_team_name, game.away_team_name = home_team.name, away_team.name
            for k, v in kwargs.items():
                if v is not None:
                    if k == "raw_data" and isinstance(game.raw_data, dict) and isinstance(v, dict):
                        if "lineups" in game.raw_data and "lineups" not in v:
                            v["lineups"] = game.raw_data["lineups"]
                    setattr(game, k, v)
        return game

    async def _find_game_by_teams_approx(self, sport: SportType, home_name: str, away_name: str,
                                         commence: Optional[datetime] = None) -> Optional[Game]:
        """Empareja un evento de The Odds API con un partido nuestro.

        Compara nombres normalizados (sin acentos, con alias como "LA Clippers" = "Los Angeles
        Clippers") y, si viene la hora de inicio, exige que esté a menos de 12 h: hay series de
        dos partidos seguidos entre los mismos equipos.
        """
        from datetime import timedelta
        if not home_name or not away_name:
            return None
        lo = (commence - timedelta(hours=12)) if commence else datetime.utcnow() - timedelta(hours=24)
        hi = (commence + timedelta(hours=12)) if commence else datetime.utcnow() + timedelta(days=3)
        cands = (await self.db.execute(
            select(Game).where(and_(Game.sport == sport, Game.game_time >= lo, Game.game_time <= hi))
            .order_by(Game.game_time.asc()))).scalars().all()
        h, a = _team_key(home_name), _team_key(away_name)
        exact = [g for g in cands if _team_key(g.home_team_name) == h and _team_key(g.away_team_name) == a]
        if exact:
            return min(exact, key=lambda g: abs((g.game_time - commence).total_seconds()) if commence and g.game_time else 0)
        # Respaldo (fútbol, nombres con sufijos distintos): uno contiene al otro
        loose = [g for g in cands if _similar(_team_key(g.home_team_name), h) and _similar(_team_key(g.away_team_name), a)]
        return loose[0] if len(loose) == 1 else None

    def _parse_datetime(self, dt_str) -> Optional[datetime]:
        """Safely parse API ISO datetimes into UTC datetime objects."""
        if not dt_str:
            return None
        try:
            if isinstance(dt_str, dict):
                dt_str = dt_str.get("start")
            if not dt_str:
                return None
            return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        except ValueError:
            return None

    def _map_soccer_status(self, status_str: str) -> GameStatus:
        status_str = str(status_str).lower()
        if status_str in ["not started", "ns", "tbd"]: return GameStatus.SCHEDULED
        if status_str in ["in progress", "live", "halftime", "ht"]: return GameStatus.LIVE
        if status_str in ["finished", "ft", "aet", "pen"]: return GameStatus.FINISHED
        if status_str in ["postponed", "pst"]: return GameStatus.POSTPONED
        if status_str in ["cancelled", "canc"]: return GameStatus.CANCELLED
        return GameStatus.SCHEDULED

    def _map_us_status(self, status_str: str) -> GameStatus:
        status_str = str(status_str).lower()
        if status_str in ["ns", "sched"]: return GameStatus.SCHEDULED
        if status_str in ["q1", "q2", "q3", "q4", "ht", "live", "p"]: return GameStatus.LIVE
        if status_str in ["ft", "aot"]: return GameStatus.FINISHED
        if status_str in ["pst"]: return GameStatus.POSTPONED
        if status_str in ["canc"]: return GameStatus.CANCELLED
        return GameStatus.SCHEDULED



# ── Emparejamiento de nombres entre fuentes ──────────────────────────
_TEAM_ALIASES = {
    "la clippers": "los angeles clippers", "la lakers": "los angeles lakers",
    "ny knicks": "new york knicks", "gs warriors": "golden state warriors",
}


def _team_key(name: Optional[str]) -> str:
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return _TEAM_ALIASES.get(s, s)


def _similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    wa, wb = set(a.split()) - {"fc", "cf", "club", "de", "the"}, set(b.split()) - {"fc", "cf", "club", "de", "the"}
    return bool(wa and wb) and len(wa & wb) / min(len(wa), len(wb)) >= 0.6


def _implied(american: Optional[float]) -> Optional[float]:
    if american is None:
        return None
    return 100.0 / (american + 100.0) if american > 0 else -american / (-american + 100.0)
