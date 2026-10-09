"""
OmniPicks — Application Configuration
Loads settings from .env file using pydantic-settings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # === MySQL ===
    mysql_host: str = Field(default="localhost")
    mysql_port: int = Field(default=3306)
    mysql_user: str = Field(default="root")
    mysql_password: str = Field(default="")
    mysql_database: str = Field(default="omnipicks")

    # === Bzzoiro Sports Data (Soccer) ===
    bzzoiro_api_key: str = Field(default="")
    bzzoiro_base_url: str = Field(default="https://sports.bzzoiro.com")

    # === API-Sports (NBA, NFL, MLB) ===
    api_sports_key: str = Field(default="")

    # === The Odds API ===
    odds_api_key: str = Field(default="")

    # === Application ===
    app_env: str = Field(default="development")
    app_debug: bool = Field(default=True)
    app_port: int = Field(default=8000)
    app_host: str = Field(default="0.0.0.0")
    frontend_url: str = Field(default="http://localhost:4200")

    @property
    def mysql_url(self) -> str:
        """Async MySQL connection URL for SQLAlchemy."""
        return (
            f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    @property
    def mysql_url_sync(self) -> str:
        """Sync MySQL connection URL for migrations/scripts."""
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    # API-Sports base URLs
    @property
    def api_sports_nba_url(self) -> str:
        return "https://v2.nba.api-sports.io"

    @property
    def api_sports_nfl_url(self) -> str:
        return "https://v1.american-football.api-sports.io"

    @property
    def api_sports_mlb_url(self) -> str:
        return "https://v1.baseball.api-sports.io"

    @property
    def odds_api_url(self) -> str:
        return "https://api.the-odds-api.com/v4"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
