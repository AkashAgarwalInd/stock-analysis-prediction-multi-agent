from pathlib import Path

import pytest
from pydantic import ValidationError

from stock_analysis.config import Settings, get_settings


class TestSettings:
    def test_default_values(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("APP_ENV", "testing")

        settings = Settings()

        assert settings.app_name == "stock-analysis"
        assert settings.app_env == "testing"
        assert settings.log_level == "INFO"
        assert settings.log_format == "console"
        assert settings.database_path == Path("data/stock_analysis.db")
        assert settings.database_echo is False
        assert settings.gemini_api_key is None
        assert settings.gemini_model_primary == "gemini-3.5-flash-lite"
        assert settings.gemini_model_critic == "gemini-3.5-flash-lite"
        assert settings.gemini_model_fallback == "gemini-3.5-flash-8b-lite"
        assert settings.gemini_temperature == 0.1
        assert settings.gemini_max_tokens == 8192
        assert settings.forecast_horizon_days == 5

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("APP_NAME", "custom-app")
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("LOG_FORMAT", "json")
        monkeypatch.setenv("DATABASE_PATH", "/custom/path/db.sqlite")
        monkeypatch.setenv("DATABASE_ECHO", "true")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("GEMINI_MODEL_PRIMARY", "gemini-1.5-pro")
        monkeypatch.setenv("GEMINI_TEMPERATURE", "0.5")
        monkeypatch.setenv("GEMINI_MAX_TOKENS", "4096")
        monkeypatch.setenv("FORECAST_HORIZON_DAYS", "10")

        settings = Settings()

        assert settings.app_name == "custom-app"
        assert settings.app_env == "production"
        assert settings.log_level == "DEBUG"
        assert settings.log_format == "json"
        assert settings.database_path == Path("/custom/path/db.sqlite")
        assert settings.database_echo is True
        assert settings.gemini_api_key == "test-key"
        assert settings.gemini_model_primary == "gemini-1.5-pro"
        assert settings.gemini_temperature == 0.5
        assert settings.gemini_max_tokens == 4096
        assert settings.forecast_horizon_days == 10

    def test_database_url_properties(self):
        settings = Settings(database_path=Path("/test/db.sqlite"))
        assert settings.database_url == "sqlite:////test/db.sqlite"
        assert settings.database_async_url == "sqlite+aiosqlite:////test/db.sqlite"

    def test_invalid_app_env(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "invalid")

        with pytest.raises(ValidationError):
            Settings()

    def test_invalid_log_format(self, monkeypatch):
        monkeypatch.setenv("LOG_FORMAT", "invalid")

        with pytest.raises(ValidationError):
            Settings()

    def test_get_settings_cached(self, monkeypatch):
        monkeypatch.setenv("APP_NAME", "test-app-1")

        s1 = get_settings()
        assert s1.app_name == "test-app-1"

        monkeypatch.setenv("APP_NAME", "test-app-2")

        s2 = get_settings()
        assert s2.app_name == "test-app-1"
        assert s1 is s2
