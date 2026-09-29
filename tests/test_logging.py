import logging
from io import StringIO
from unittest.mock import patch

from stock_analysis.logging import configure_logging, get_logger, bind_context, clear_context


class TestLogging:
    def test_configure_logging_console(self, monkeypatch):
        monkeypatch.setenv("LOG_FORMAT", "console")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        from stock_analysis.config.settings import get_settings
        get_settings.cache_clear()

        configure_logging()
        logger = get_logger("test")
        assert logger is not None

    def test_configure_logging_json(self, monkeypatch):
        monkeypatch.setenv("LOG_FORMAT", "json")
        monkeypatch.setenv("LOG_LEVEL", "INFO")
        from stock_analysis.config.settings import get_settings
        get_settings.cache_clear()

        configure_logging()
        logger = get_logger("test")
        assert logger is not None

    def test_bind_context(self):
        bind_context(request_id="test-123", user_id="user-456")
        logger = get_logger("test")
        logger.info("test message")
        clear_context()

    def test_get_logger_returns_bound_logger(self):
        logger = get_logger("test.module")
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "error")