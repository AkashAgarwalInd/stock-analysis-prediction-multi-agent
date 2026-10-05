"""Phase 18 "no known future-data leakage": every record a backtest writes is stamped with
the simulated clock, never the wall clock, and no forecast sees data after its as-of."""

from datetime import timedelta

import pytest

from stock_analysis.backtest import run_backtest
from stock_analysis.backtest.runner import later_history
from stock_analysis.database import Database, ForecastSnapshotStore, LLMCallStore
from stock_analysis.snapshots.builder import IST
from tests.backtest_helpers import NOW, RecordingLoader, synthetic_history
from tests.hardening_helpers import stub_factory


@pytest.fixture(scope="module")
def backtest(tmp_path_factory):
    path = tmp_path_factory.mktemp("leak") / "bt.db"
    result = run_backtest(
        "RELIANCE",
        8,
        database_path=path,
        llm_factory=stub_factory(),  # the real factory, so LLM calls are logged
        now=NOW,
        history_loader=RecordingLoader(synthetic_history(weekly_swing_pct=4.0)),
    )
    db = Database(path)
    yield result, db
    db.close()


def test_forecast_dates_follow_the_simulated_clock(backtest):
    result, db = backtest
    store = ForecastSnapshotStore(db)
    for week in result.weeks:
        snap = store.get(week.forecast_id)
        assert snap.made_at == week.forecast_made_at
        assert snap.final_forecast.forecast_date == snap.made_at.astimezone(IST).date().isoformat()
        assert snap.final_forecast.forecast_date == week.as_of_date.isoformat()


def test_nothing_is_stamped_after_the_last_simulated_evaluation(backtest):
    result, db = backtest
    last = max(w.evaluated_at for w in result.weeks)
    assert later_history(db, last + timedelta(seconds=1)) == {}


def test_llm_telemetry_is_kept_out_of_point_in_time_history(backtest):
    """``llm_calls`` rows carry wall-clock times; they must not count as later history, or a
    database could never be extended and point-in-time reads would see them."""
    result, db = backtest
    calls = LLMCallStore(db).calls()
    assert len(calls) == result.llm_usage.calls > 0
    assert all(c.run_id == calls[0].run_id and c.run_id.startswith("backtest-") for c in calls)
    first_forecast = min(w.forecast_made_at for w in result.weeks)
    assert "llm_calls" not in later_history(db, first_forecast)


def test_forecasts_only_see_bars_up_to_their_as_of(backtest):
    """The stored close series behind each quant baseline ends at the week's as-of session."""
    result, db = backtest
    store = ForecastSnapshotStore(db)
    for week in result.weeks:
        snap = store.get(week.forecast_id)
        history = store.get_price_history(snap.data_inputs["price"]["history_sha256"])
        assert history.dates[-1] == week.as_of_date.isoformat()
