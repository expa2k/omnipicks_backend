"""
OmniPicks — FastAPI Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger
import sys
import asyncio
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_pipeline import run_daily_pipeline
from cron_sync_profiles import run_cron

from app.config import get_settings
from app.database import init_db, close_db, check_db_connection
from app.api.routes_predictions import router as predictions_router
from app.api.routes_odds import router as odds_router
from app.api.routes_sports import router as sports_router
from app.api.routes_bankroll import router as bankroll_router


# ── Configure Logging ─────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="DEBUG",
)
logger.add(
    "logs/omnipicks.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
)


# ── Lifespan ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    settings = get_settings()
    logger.info("🚀 OmniPicks starting up...")
    logger.info(f"   Environment: {settings.app_env}")
    logger.info(f"   Debug: {settings.app_debug}")

    # Initialize database
    try:
        await init_db()
        db_ok = await check_db_connection()
        if db_ok:
            logger.info("✅ MySQL connection verified")
        else:
            logger.warning("⚠️  MySQL not available - running without DB")
    except Exception as e:
        logger.warning(f"⚠️  Database init skipped: {e}")

    # Log API key status
    logger.info(f"   Bzzoiro API: {'✅ configured' if settings.bzzoiro_api_key else '❌ missing'}")
    logger.info(f"   API-Sports:  {'✅ configured' if settings.api_sports_key else '❌ missing'}")
    logger.info(f"   Odds API:    {'✅ configured' if settings.odds_api_key else '❌ missing'}")

    logger.info("🏆 OmniPicks is ready!")

    # Ejecutar pipeline automáticamente en segundo plano al arrancar
    logger.info("⏳ Lanzando el pipeline de extracción de datos en segundo plano...")
    asyncio.create_task(run_daily_pipeline())
    
    # Lanzar la sincronización histórica (Similar-Opponent)
    logger.info("⏳ Lanzando la sincronización histórica de perfiles (Cron V3)...")
    asyncio.create_task(run_cron())

    yield

    # Shutdown
    logger.info("🔌 Shutting down OmniPicks...")
    await close_db()
    logger.info("👋 Goodbye!")


# ── Create App ────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="OmniPicks API",
        description="AI-Powered Multi-Sport Prediction System — NBA, NFL, MLB, Soccer",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS for Angular frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            settings.frontend_url,
            "http://localhost:4200",
            "http://localhost:4201",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(predictions_router, prefix="/api/predictions", tags=["Predictions"])
    app.include_router(odds_router, prefix="/api/odds", tags=["Odds"])
    app.include_router(sports_router, prefix="/api/sports", tags=["Sports Data"])
    app.include_router(bankroll_router, prefix="/api/bankroll", tags=["Bankroll"])

    @app.get("/", tags=["Health"])
    async def root():
        return {
            "name": "OmniPicks API",
            "version": "1.0.0",
            "status": "running",
            "sports": ["NBA", "NFL", "MLB", "Soccer"],
            "docs": "/docs",
        }

    @app.get("/health", tags=["Health"])
    async def health_check():
        db_status = await check_db_connection()
        settings = get_settings()
        return {
            "status": "healthy" if db_status else "degraded",
            "database": "connected" if db_status else "disconnected",
            "apis": {
                "bzzoiro": "configured" if settings.bzzoiro_api_key else "missing",
                "api_sports": "configured" if settings.api_sports_key else "missing",
                "odds_api": "configured" if settings.odds_api_key else "missing",
            },
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
    )
