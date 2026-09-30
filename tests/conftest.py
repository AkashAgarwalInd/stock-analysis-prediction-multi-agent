import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("APP_ENV", "testing")
os.environ.setdefault("GEMINI_API_KEY", "test-key-12345")


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    from stock_analysis.config.settings import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def temp_db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    yield path
    if path.exists():
        path.unlink()