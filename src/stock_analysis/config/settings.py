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

    # Phase 7 memory (Plan.md §20, §47)
    memory_max_prior_forecasts: int = 3
    lesson_activation_min_evidence: int = 3
    lesson_event_activation_min_evidence: int = 2
    max_active_lessons_in_prompt: int = 5

    # Plan.md Phase 9 outcome scoring (§14-16)
    outcome_data_delay_minutes: int = 60  # wait after the 15:30 IST close for EOD data
    outcome_max_wait_trading_days: int = 5  # missing prices this long after target -> invalid
    outcome_max_daily_move_pct: float = 35.0  # larger moves suggest an unadjusted split/bonus

    # Plan.md Phases 11-12: postmortem, lessons and bounded calibration (§18-23)
    learning_enabled: bool = True  # False: calibration is computed but not applied to forecasts
    postmortem_temperature: float = 0.2
    lesson_stale_after_days: int = 120  # unconfirmed this long -> retired
    calibration_min_samples: int = 8
    calibration_shrinkage_k: int = 8  # weight n/(n+k) on the estimate, the rest on "no change"
    calibration_window: int = 52  # most recent scored forecasts used per ticker
    calibration_vol_multiplier_min: float = 0.8
    calibration_vol_multiplier_max: float = 1.5
    calibration_bias_max_shift_pct: float = 1.0
    calibration_bias_min_t_stat: float = 2.0
    calibration_bias_min_sign_agreement: float = 0.6

    # Plan.md Phase 13 scorecards (§24): evaluation only; nothing is reweighted automatically
    min_samples_analyst_weights: int = 20
    analyst_weighting_enabled: bool = False  # feature flag; no weighting is implemented yet
    min_samples_decision_evaluation: int = 20  # also: before the track record ranks variants

    # Plan.md Phase 14 probability calibration (§27)
    probability_calibration_buckets: int = Field(default=10, ge=1, le=100)
    min_samples_probability_calibration: int = 20

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.database_path}"

    @property
    def database_async_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.database_path}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
