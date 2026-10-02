from typer.testing import CliRunner

from stock_analysis.cli import app

runner = CliRunner()


def test_cli_status(monkeypatch):
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("DATABASE_PATH", ":memory:")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from stock_analysis.config.settings import get_settings
    get_settings.cache_clear()

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Application Status" in result.output
    assert "stock-analysis" in result.output


def test_cli_config(monkeypatch):
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("DATABASE_PATH", ":memory:")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-12345")
    from stock_analysis.config.settings import get_settings
    get_settings.cache_clear()

    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "Configuration" in result.output
    # API key is masked: *** + last 4 chars
    assert "***2345" in result.output


def test_cli_init(monkeypatch, tmp_path):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("DATABASE_PATH", str(db_path))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from stock_analysis.config.settings import get_settings
    get_settings.cache_clear()

    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert db_path.exists()


def test_cli_config_redaction(monkeypatch):
    """Test that API key is properly masked in config output."""
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("DATABASE_PATH", ":memory:")
    monkeypatch.setenv("GEMINI_API_KEY", "secret-key-1234")
    from stock_analysis.config.settings import get_settings
    get_settings.cache_clear()

    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    # Full key should not be visible
    assert "secret-key-1234" not in result.output
    # Last 4 chars should be visible
    assert "1234" in result.output
