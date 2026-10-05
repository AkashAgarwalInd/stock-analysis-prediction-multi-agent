"""Plan.md §76 final acceptance test, through the CLI exactly as written: backtest,
evaluate, calibration and analyze chain through the app database with no ``--database``."""

from functools import partial

import pytest
from typer.testing import CliRunner

from stock_analysis.cli import app
from stock_analysis.config.settings import get_settings
from stock_analysis.demo import ACCEPTANCE_ITEMS, acceptance_checklist
from tests.backtest_helpers import LAST_TARGET, NOW
from tests.outcome_helpers import FakePriceSource
from tests.phase18.test_demo import SyntheticMarket


@pytest.fixture
def app_db(tmp_path, monkeypatch):
    """The app database, with the working directory in ``tmp_path`` so relative paths
    such as data/backtests/ would land there."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "app.db"
    monkeypatch.setenv("DATABASE_PATH", str(path))
    get_settings.cache_clear()
    return path


@pytest.fixture
def market(monkeypatch):
    """Synthetic history for the backtest and the analysis; the analysis runs at NOW."""
    import stock_analysis.analysis as analysis

    market = SyntheticMarket()
    monkeypatch.setattr("stock_analysis.backtest.runner.load_yfinance_history", market)
    monkeypatch.setattr(
        "stock_analysis.backtest.runner.last_completed_trading_date", lambda now, cal: LAST_TARGET
    )
    monkeypatch.setattr(
        analysis,
        "run_analysis",
        partial(analysis.run_analysis, source=market, run_at=NOW, price_source=FakePriceSource({})),
    )
    return market


def _invoke(*args: str) -> str:
    result = CliRunner().invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return result.output


def test_acceptance_sequence_shares_the_app_database(app_db, market, tmp_path):
    backtest = _invoke("backtest", "--ticker", "RELIANCE.NS", "--weeks", "12", "--no-llm")
    assert "# Backtest: RELIANCE (RELIANCE.NS), 12 weeks" in backtest
    assert f"-Database:`{app_db}`" in "".join(backtest.split())  # wrapping may split the path
    assert not (tmp_path / "data").exists()  # no separate backtest database

    evaluate = _invoke("evaluate", "--ticker", "RELIANCE.NS")
    assert "Forecast track record for RELIANCE" in evaluate
    assert "No scored forecasts yet" not in evaluate

    calibration = _invoke("calibration", "--ticker", "RELIANCE.NS")
    assert "## Calibration versions for RELIANCE" in calibration
    assert "None stored" not in calibration  # the backtest's weekly swings were learned

    report = tmp_path / "analysis.md"
    _invoke("analyze", "RELIANCE", "--no-llm", "-o", str(report))
    checklist = acceptance_checklist(report.read_text())
    assert checklist == {item: True for item, _ in ACCEPTANCE_ITEMS}
    assert "- Calibration v" in report.read_text()


def test_backtest_into_a_database_with_later_history_suggests_a_new_one(app_db, market):
    _invoke("backtest", "--ticker", "RELIANCE", "--weeks", "2", "--no-llm")
    result = CliRunner().invoke(app, ["backtest", "--ticker", "INFY", "--weeks", "4", "--no-llm"])
    output = " ".join(result.output.split())  # undo the console's line wrapping
    assert result.exit_code == 1
    assert "already has history from" in output
    assert "add --database data/backtests/INFY-" in output
    assert not (app_db.parent / "data").exists()  # only suggested, never created


def test_explicit_database_failure_has_no_hint(tmp_path, market):
    path = str(tmp_path / "bt.db")
    _invoke("backtest", "--ticker", "RELIANCE", "--weeks", "2", "--no-llm", "--database", path)
    result = CliRunner().invoke(
        app, ["backtest", "--ticker", "RELIANCE", "--weeks", "2", "--no-llm", "--database", path]
    )
    output = " ".join(result.output.split())
    assert result.exit_code == 1
    assert "already has history from" in output
    assert "add --database" not in output
