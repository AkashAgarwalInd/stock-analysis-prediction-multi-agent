"""Plan.md Phase 10: sequential point-in-time backtest ("done when" 8-12 weeks run and
leakage tests pass)."""

from datetime import date, timedelta

import pytest
from typer.testing import CliRunner

from stock_analysis.backtest import (
    DISABLED_IN_BACKTEST,
    BacktestError,
    LookaheadError,
    backtest_schedule,
    forecast_time,
    render_backtest_report,
    run_backtest,
)
from stock_analysis.database import Database, ForecastSnapshotStore, LearningStore, OutcomeStore
from stock_analysis.market.calendar import get_trading_calendar
from stock_analysis.review.prices import NIFTY_50_SYMBOL
from stock_analysis.schemas.memory import MemoryContext
from tests.backtest_helpers import (
    LAST_TARGET,
    NOW,
    SYMBOL,
    BacktestLLM,
    CountingLLM,
    RecordingLoader,
    synthetic_history,
)

WEEKS = 12


def _run(path, data, *, weeks=WEEKS, llm=None, use_llm=True):
    return run_backtest(
        "RELIANCE",
        weeks,
        database_path=path,
        llm_factory=llm if llm is not None else BacktestLLM(),
        use_llm=use_llm,
        end=LAST_TARGET,
        now=NOW,
        history_loader=RecordingLoader(data),
    )


@pytest.fixture(scope="module")
def swing_run(tmp_path_factory):
    """12 weeks where weekly moves exceed what daily volatility implies."""
    path = tmp_path_factory.mktemp("bt") / "backtest.db"
    llm = BacktestLLM()
    result = _run(path, synthetic_history(weekly_swing_pct=5.0), llm=llm)
    db = Database(path)
    yield result, llm, db
    db.close()


class TestSchedule:
    def test_back_to_back_windows_ending_on_the_target(self):
        cal = get_trading_calendar()
        windows = backtest_schedule(12, LAST_TARGET, horizon=5, calendar=cal)
        assert len(windows) == 12
        assert windows[-1][1] == LAST_TARGET
        for (_, target), (next_as_of, _) in zip(windows, windows[1:], strict=False):
            assert next_as_of == target  # windows share only an endpoint
        for as_of, target in windows:
            assert len(cal.get_trading_days(as_of + timedelta(days=1), target)) == 5

    def test_non_trading_end_rolls_back(self):
        cal = get_trading_calendar()
        saturday = LAST_TARGET + timedelta(days=1)
        assert backtest_schedule(1, saturday, horizon=5, calendar=cal)[0][1] == LAST_TARGET

    def test_end_must_have_completed(self, tmp_path):
        with pytest.raises(BacktestError, match="last completed session"):
            run_backtest(
                "RELIANCE",
                2,
                database_path=tmp_path / "b.db",
                end=date(2026, 10, 2),
                now=NOW,
                history_loader=RecordingLoader(synthetic_history()),
            )


class TestPointInTimeData:
    def test_view_never_returns_later_bars(self):
        data = synthetic_history()
        view = data.as_of(date(2026, 8, 3))
        assert max(b.date for b in view.bars(SYMBOL)) == date(2026, 8, 3)
        assert all(b.date <= date(2026, 8, 3) for b in view.quant_prices(NIFTY_50_SYMBOL))
        assert view.frame(SYMBOL).index.max().date() == date(2026, 8, 3)

    def test_evaluation_prices_stop_at_the_simulated_clock(self):
        source = synthetic_history().price_source()
        with pytest.raises(LookaheadError):
            source.fetch(SYMBOL, date(2026, 8, 3), date(2026, 8, 10))
        source.available_until = date(2026, 8, 10)
        assert source.fetch(SYMBOL, date(2026, 8, 3), date(2026, 8, 10)).bars
        with pytest.raises(LookaheadError):
            source.fetch(SYMBOL, date(2026, 8, 3), date(2026, 8, 11))


class TestBacktestRun:
    def test_simulates_twelve_weeks(self, swing_run):
        result, _, _ = swing_run
        assert len(result.weeks) == WEEKS
        assert [w.status for w in result.weeks] == ["scored"] * WEEKS
        assert all(w.forecast_id for w in result.weeks)

    def test_each_forecast_used_only_data_up_to_its_as_of(self, swing_run):
        result, _, db = swing_run
        store = ForecastSnapshotStore(db)
        for week in result.weeks:
            snap = store.get(week.forecast_id)
            assert snap.as_of_date == week.as_of_date
            assert snap.target_date == week.target_date
            assert snap.made_at == forecast_time(week.as_of_date)
            price = snap.data_inputs["price"]
            assert price["last_date"] == week.as_of_date.isoformat()
            history = store.get_price_history(price["history_sha256"])
            assert max(history.dates) == week.as_of_date.isoformat()
            assert snap.data_inputs["technical_indicators"]["date"] == week.as_of_date.isoformat()
            assert snap.data_inputs["market_context"]["as_of"] == week.as_of_date.isoformat()

    def test_memory_only_holds_earlier_weeks(self, swing_run):
        result, _, db = swing_run
        store = ForecastSnapshotStore(db)
        for k, week in enumerate(result.weeks):
            ctx = MemoryContext.model_validate(store.get(week.forecast_id).data_inputs["memory"])
            assert {f.target_date for f in ctx.prior_forecasts} <= {
                w.target_date for w in result.weeks[:k]
            }
            assert len(ctx.prior_forecasts) == min(k, 3)
            # every earlier week was already scored when this one was forecast
            assert ctx.track_record.forecasts_made == k
            assert ctx.track_record.forecasts_scored == k

    def test_calibration_is_learned_sequentially_not_from_today(self, swing_run):
        result, _, db = swing_run
        history = LearningStore(db).calibration_history("RELIANCE")
        assert history, "weekly swings should eventually trigger calibration"
        # nothing changes before eight independent weeks are scored
        assert all(w.calibration_version == 0 for w in result.weeks[:8])
        assert history[0].n_samples >= 8
        for week in result.weeks:
            usable = [c.version for c in history if c.created_at <= week.forecast_made_at]
            assert week.calibration_version == (max(usable) if usable else 0)
        assert result.weeks[-1].calibration_version > 0
        assert history[0].vol_multiplier > 1.0

    def test_outcomes_are_scored_before_the_next_forecast(self, swing_run):
        result, _, db = swing_run
        outcomes = OutcomeStore(db)
        for week, nxt in zip(result.weeks, result.weeks[1:], strict=False):
            evaluated = outcomes.get(week.forecast_id).evaluated_at
            assert evaluated == week.evaluated_at <= nxt.forecast_made_at

    def test_disabled_analysts_never_reach_the_llm(self, swing_run):
        result, llm, db = swing_run
        assert result.disabled_analysts == DISABLED_IN_BACKTEST
        assert len(llm.analyst_prompts) == 2 * WEEKS  # technical + context only
        assert not any('"news"' in p or '"fundamentals"' in p for p in llm.analyst_prompts)
        snap = ForecastSnapshotStore(db).get(result.weeks[0].forecast_id)
        assert snap.data_inputs["disabled_analysts"] == DISABLED_IN_BACKTEST
        assert snap.analyst_reports["sentiment"]["confidence"] == 0.0
        assert snap.analyst_reports["fundamental"]["stance"] == "neutral"
        # the two enabled analysts still let the predictor adjust the baseline
        assert all(w.adjustment_applied for w in result.weeks)

    def test_learning_ran_each_week(self, swing_run):
        result, llm, db = swing_run
        learning = LearningStore(db)
        assert all(learning.get_postmortem(w.forecast_id) for w in result.weeks)
        assert all(learning.decision_outcomes(w.forecast_id) for w in result.weeks)
        # the LLM is only consulted for weeks that were not ordinary noise
        assert 0 < len(llm.postmortem_prompts) <= WEEKS

    def test_report(self, swing_run):
        result, _, _ = swing_run
        text = render_backtest_report(result)
        assert f"# Backtest: RELIANCE (RELIANCE.NS), {WEEKS} weeks" in text
        assert sum(line.startswith("| 2026-") for line in text.splitlines()) == WEEKS
        assert "volatility multiplier 1.00 →" in text
        assert "sentiment analyst disabled" in text
        assert "not investment advice" in text
        assert "may already know how these dates played out" in text
        assert "## Scorecards for RELIANCE" in text
        assert result.scorecards.n_forecasts == WEEKS
        assert "## Forecast track record for RELIANCE" in text
        assert "| Naive flat |" in text
        assert "## Probability calibration for RELIANCE" in text
        assert result.evaluation.track_records[0].n_forecasts == WEEKS

    def test_report_cells_cannot_break_the_table(self, swing_run):
        result, _, _ = swing_run
        week = result.weeks[0].model_copy(
            update={"forecast_id": None, "status": "no_forecast", "reason": "a | b"}
        )
        text = render_backtest_report(result.model_copy(update={"weeks": [week]}))
        row = next(line for line in text.splitlines() if line.startswith("| 2026-"))
        assert "a / b" in row and row.count("|") == 13


class TestNoLeakage:
    def test_future_prices_cannot_change_earlier_forecasts(self, tmp_path):
        """Tripling every price after week k's target leaves weeks 1..k+1 untouched."""
        schedule = backtest_schedule(10, LAST_TARGET, horizon=5, calendar=get_trading_calendar())
        cutoff = schedule[5][1]  # target of week 6 = as-of of week 7
        clean = _run(tmp_path / "clean.db", synthetic_history(), weeks=10)
        poisoned = _run(tmp_path / "poison.db", synthetic_history(poison_after=cutoff), weeks=10)

        clean_db, poison_db = Database(tmp_path / "clean.db"), Database(tmp_path / "poison.db")
        try:
            a, b = ForecastSnapshotStore(clean_db), ForecastSnapshotStore(poison_db)
            for w_clean, w_poison in zip(clean.weeks[:7], poisoned.weeks[:7], strict=True):
                s1, s2 = a.get(w_clean.forecast_id), b.get(w_poison.forecast_id)
                assert s1.quant_baseline == s2.quant_baseline
                assert s1.final_forecast.p50_price == s2.final_forecast.p50_price
                assert (
                    s1.data_inputs["technical_indicators"] == s2.data_inputs["technical_indicators"]
                )
                assert s1.data_inputs["market_context"] == s2.data_inputs["market_context"]
                assert s1.calibration_version == s2.calibration_version
            # ...while the poisoned future does change what came after it
            assert poisoned.weeks[7].last_close != clean.weeks[7].last_close
        finally:
            clean_db.close()
            poison_db.close()

    def test_data_is_requested_only_up_to_the_last_target(self, tmp_path):
        loader = RecordingLoader(synthetic_history())
        run_backtest(
            "RELIANCE",
            3,
            database_path=tmp_path / "b.db",
            llm_factory=BacktestLLM(),
            end=LAST_TARGET,
            now=NOW,
            history_loader=loader,
        )
        ((symbols, _, end),) = loader.requests
        assert end == LAST_TARGET
        assert set(symbols) == {SYMBOL, NIFTY_50_SYMBOL, "^INDIAVIX"}


class TestSafety:
    def test_rerun_into_the_same_database_is_refused(self, tmp_path):
        path = tmp_path / "b.db"
        _run(path, synthetic_history(), weeks=2)
        with pytest.raises(BacktestError, match="already has history from"):
            _run(path, synthetic_history(), weeks=2)

    def test_history_can_only_be_appended_in_time_order(self, tmp_path):
        """An earlier period after a later one would rewrite what past reads returned."""
        path = tmp_path / "b.db"
        earlier_end = backtest_schedule(3, LAST_TARGET, horizon=5, calendar=get_trading_calendar())[
            0
        ][0]

        def run(end):
            return run_backtest(
                "RELIANCE",
                2,
                database_path=path,
                llm_factory=BacktestLLM(),
                end=end,
                now=NOW,
                history_loader=RecordingLoader(synthetic_history()),
            )

        run(earlier_end)  # older period first: fine
        run(LAST_TARGET)  # then a later one: fine
        with pytest.raises(BacktestError, match="forecast_snapshots"):
            run(earlier_end - timedelta(days=21))  # going back in time is refused

    def test_missing_price_history_fails_fast(self, tmp_path):
        from stock_analysis.backtest import HistoricalMarketData

        with pytest.raises(BacktestError, match="No price history could be loaded"):
            _run(tmp_path / "e.db", HistoricalMarketData({SYMBOL: []}), weeks=2)

    def test_a_failing_week_does_not_stop_the_backtest(self, tmp_path, monkeypatch):
        import stock_analysis.backtest.runner as runner

        schedule = backtest_schedule(3, LAST_TARGET, horizon=5, calendar=get_trading_calendar())
        bad = schedule[1][0]
        real = runner.technical_summary

        def broken(view, symbol):
            if view.as_of_date == bad:
                raise ValueError("indicator failure")
            return real(view, symbol)

        monkeypatch.setattr(runner, "technical_summary", broken)
        result = _run(tmp_path / "f.db", synthetic_history(), weeks=3)
        assert [w.status for w in result.weeks] == ["scored", "no_forecast", "scored"]
        assert result.weeks[1].reason == "forecast failed: indicator failure"
        assert result.errors == [f"{bad}: indicator failure"]

    def test_quant_only_makes_no_llm_calls(self, tmp_path):
        llm = CountingLLM()
        result = _run(tmp_path / "q.db", synthetic_history(), weeks=3, llm=llm, use_llm=False)
        assert llm.calls == 0
        assert [w.status for w in result.weeks] == ["scored"] * 3
        assert not any(w.adjustment_applied for w in result.weeks)
        assert set(result.disabled_analysts) == {"technical", "fundamental", "sentiment", "context"}

    def test_missing_as_of_bar_skips_the_week(self, tmp_path):
        data = synthetic_history()
        schedule = backtest_schedule(3, LAST_TARGET, horizon=5, calendar=get_trading_calendar())
        hole = schedule[1][0]
        data._bars[SYMBOL] = [b for b in data._bars[SYMBOL] if b.date != hole]
        result = _run(tmp_path / "h.db", data, weeks=3)
        assert result.weeks[1].status == "no_forecast"
        assert "no RELIANCE.NS price bar" in result.weeks[1].reason
        assert result.weeks[2].status == "scored"


def test_cli_backtest(tmp_path, monkeypatch):
    from stock_analysis.cli import app

    loader = RecordingLoader(synthetic_history())
    monkeypatch.setattr("stock_analysis.backtest.runner.load_yfinance_history", loader)
    monkeypatch.setattr(
        "stock_analysis.backtest.runner.last_completed_trading_date", lambda now, cal: LAST_TARGET
    )
    report = tmp_path / "report.md"
    result = CliRunner().invoke(
        app,
        [
            "backtest",
            "--ticker",
            "RELIANCE",
            "--weeks",
            "2",
            "--no-llm",
            "--database",
            str(tmp_path / "cli.db"),
            "--output",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "# Backtest: RELIANCE" in result.output
    assert report.read_text().startswith("# Backtest: RELIANCE")


def test_late_data_is_scored_by_a_later_week(tmp_path, monkeypatch):
    """Week 1's target bar arrives late: it is scored at week 2's review and reported so."""
    from stock_analysis.backtest import HistoricalPriceSource

    schedule = backtest_schedule(2, LAST_TARGET, horizon=5, calendar=get_trading_calendar())
    target = schedule[0][1]
    real_fetch = HistoricalPriceSource.fetch
    missed = []

    def late(self, symbol, start, end):
        if symbol == SYMBOL and end == target and not missed:
            missed.append(end)
            return real_fetch(self, symbol, start, end - timedelta(days=1))  # no target bar yet
        return real_fetch(self, symbol, start, end)

    monkeypatch.setattr(HistoricalPriceSource, "fetch", late)
    result = _run(tmp_path / "late.db", synthetic_history(), weeks=2)
    assert missed == [target]
    assert [w.status for w in result.weeks] == ["scored", "scored"]
