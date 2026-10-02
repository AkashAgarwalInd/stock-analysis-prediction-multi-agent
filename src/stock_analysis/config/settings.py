from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "stock-analysis"
    app_env: Literal["development", "staging", "production", "testing"] = "development"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"

    database_path: Path = Field(default=Path("data/stock_analysis.db"))
    database_echo: bool = False

    gemini_api_key: str | None = None
    gemini_model_primary: str = "gemini-3.5-flash-lite"
    gemini_model_critic: str = "gemini-3.5-flash-lite"
    gemini_model_fallback: str = "gemini-3.5-flash-8b-lite"
    gemini_temperature: float = 0.1
    gemini_max_tokens: int = 8192

    nse_data_cache_dir: Path = Field(default=Path("data/cache"))
    forecast_horizon_days: int = 5
    max_revisions: int = 2
    predictor_max_prob_shift: float = 0.15

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.database_path}"

    @property
    def database_async_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.database_path}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
