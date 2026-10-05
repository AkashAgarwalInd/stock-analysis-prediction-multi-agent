"""Plan.md Phase 17 "done when": ``stock-agents analyze RELIANCE`` produces the end-to-end
demo, whose report visibly contains the 15 parts of the §76 acceptance test."""

import tomllib
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

import stock_analysis.analysis as analysis
from stock_analysis.analysis import collect_inputs, news_summary, run_analysis
from stock_analysis.cli.main import app
from stock_analysis.data.fundamentals import Fundamentals
from stock_analysis.data.market_context import MarketContext
from stock_analysis.data.news import NewsCollection, NewsItem
from stock_analysis.database import LearningStore
from stock_analysis.market.resolver import resolve_nse_ticker
from stock_analysis.review import NIFTY_50_SYMBOL
from stock_analysis.schemas.learning import CalibrationParams
from stock_analysis.schemas.memory import (
    LastReview,
    MemoryContext,
    OutcomeMetrics,
    TrackRecord,
    WindowMetrics,
)
from stock_analysis.schemas.scorecard import HitRate
from stock_analysis.snapshots import find_unsupported_numbers
from stock_analysis.snapshots.report import (
    _NOT_LOADED,
    DISCLAIMER,
    _benchmark_lines,
    _last_review_lines,
)
from stock_analysis.versions import CALIBRATOR_VERSION
from tests.backtest_helpers import BacktestLLM
from tests.forecast_helpers import LAST_BAR, MADE_AT
from tests.outcome_helpers import NOW_MATURED, FakePriceSource, nifty_series, path_series

# Plan.md §76: what the final analysis must visibly contain, in report order
ACCEPTANCE_SECTIONS = [
    "## Last forecast vs actual",  # 1-6: previous forecast, actual, errors, LLM value, cause
    "## System adaptation",  # 7
    "## Current analysis",  # 8-9: regime, analyst perspectives
    "## Quant baseline",  # 10
    "## Final forecast",  # 11
    "## Risks",  # 14
    "## Invalidation triggers",  # 14
    "## Track record",  # 12
    "## Benchmark comparison",  # 13
    "## Disclaimer",  # 15
]


def _bars(n=300):
    rng = np.random.default_rng(1)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    return [
        SimpleNamespace(
            date=LAST_BAR - timedelta(days=n - 1 - i),
            open=Decimal(str(round(c * 0.998, 4))),
            high=Decimal(str(round(c * 1.01, 4))),
            low=Decimal(str(round(c * 0.99, 4))),
            close=Decimal(str(round(c, 4))),
            volume=1_000_000 + i,
        )
        for i, c in enumerate(closes)
    ]


class FakeSource:
    """Live inputs without the network; ``fail`` names parts that raise."""

    def __init__(self, bars=None, fail=()):
        self.bars = _bars() if bars is None else bars
        self.fail = set(fail)
        self.calls: list[str] = []

    def _call(self, part):
        self.calls.append(part)
        if part in self.fail:
            raise RuntimeError(f"{part} source down")

    def prices(self, symbol):
        self._call("prices")
        return self.bars

    def fundamentals(self, symbol):
        self._call("fundamentals")
        return Fundamentals(symbol=symbol, pe_ratio=21.5, sector="Energy")

    def news(self, symbol):
        self._call("news")
        items = [
            NewsItem(
                title=f"Reliance update {i} &amp; outlook",
                url=f"https://example.com/{i}",
                source="Example",
                published_at=datetime(2026, 9, 1 + i),
                summary="Refining margins&nbsp;improved",
            )
            for i in range(12)
        ]
        return NewsCollection(symbol=symbol, items=items)

    def market_context(self, symbol, sector):
        self._call("market_context")
        return MarketContext(symbol=symbol, nifty_50=24000.5, india_vix=14.2, sector_name=sector)


def _rally_prices(snapshot):
    return FakePriceSource(
        {
            snapshot.resolved_symbol: path_series(snapshot, [2.0, 4.0, 6.0, 8.0, 9.0]),
            NIFTY_50_SYMBOL: nifty_series(snapshot, 0.5),
        }
    )


def _analyze(db, run_at, query="reliance", **options):
    options.setdefault("llm_factory", BacktestLLM())
    options.setdefault("source", FakeSource())
    options.setdefault("news_source", None)  # no RSS hindsight news in tests
    options.setdefault("price_source", FakePriceSource({}))
    return run_analysis(query, db, run_at=run_at, **options)


class TestCollectInputs:
    def test_every_input_is_summarized(self):
        source = FakeSource()
        inputs = collect_inputs("RELIANCE", "RELIANCE.NS", "Reliance Industries Ltd", source)
        state = inputs.state
        assert inputs.unavailable == {}
        assert inputs.bars is source.bars
        tech = state.technical_indicators_summary
        assert tech["date"] == LAST_BAR.isoformat()
        assert tech["close"] == pytest.approx(float(source.bars[-1].close))
        assert {"rsi_14", "sma_200", "realized_vol_20"} <= tech.keys()
        assert state.fundamentals_summary["pe_ratio"] == 21.5
        assert state.fundamentals_summary["sector"] == "Energy"
        assert "roe" not in state.fundamentals_summary  # empty fields are dropped
        assert state.market_context_summary["sector_name"] == "Energy"  # sector passed on
        news = state.news_summary
        assert news["article_count"] == 12
        assert len(news["articles"]) == analysis.inputs.MAX_NEWS_ARTICLES
        assert news["articles"][0]["title"] == "Reliance update 11 & outlook"  # newest first
        assert news["articles"][0]["summary"] == "Refining margins improved"  # &nbsp; normalized

    def test_failing_source_is_recorded_not_raised(self):
        source = FakeSource(fail={"fundamentals", "news"})
        inputs = collect_inputs("RELIANCE", "RELIANCE.NS", "Reliance", source)
        assert inputs.state.fundamentals_summary == {
            "unavailable": "RuntimeError: fundamentals source down"
        }
        assert inputs.state.news_summary == {"unavailable": "RuntimeError: news source down"}
        assert inputs.state.market_context_summary["nifty_50"] == 24000.5
        assert inputs.state.technical_indicators_summary["rsi_14"] is not None

    def test_short_history_has_no_indicators(self):
        inputs = collect_inputs("X", "X.NS", "X", FakeSource(bars=_bars(50)))
        assert inputs.state.technical_indicators_summary == {
            "unavailable": "not enough price history (50 daily bars)"
        }

    def test_failed_price_fetch_explains_the_missing_indicators(self):
        inputs = collect_inputs("X", "X.NS", "X", FakeSource(fail={"prices"}))
        assert inputs.bars == []
        assert inputs.state.technical_indicators_summary == {
            "unavailable": "RuntimeError: prices source down"
        }

    def test_disabled_analysts_inputs_are_not_fetched(self):
        source = FakeSource()
        disabled = dict.fromkeys(("technical", "fundamental", "sentiment", "context"), "off")
        inputs = collect_inputs("X", "X.NS", "X", source, disabled_analysts=disabled)
        assert source.calls == ["prices"]  # the quant baseline still needs prices
        state = inputs.state
        assert state.disabled_analysts == disabled
        assert state.technical_indicators_summary is None
        assert state.news_summary is None

    def test_bar_of_a_session_still_open_is_dropped(self):
        during_session = datetime(2026, 9, 29, 5, 0, tzinfo=UTC)  # 10:30 IST on LAST_BAR
        source = FakeSource()
        inputs = collect_inputs("X", "X.NS", "X", source, run_at=during_session)
        assert inputs.bars == source.bars[:-1]
        assert inputs.bars[-1].date == LAST_BAR - timedelta(days=1)
        assert inputs.state.technical_indicators_summary["date"] == inputs.bars[-1].date.isoformat()

    def test_bar_of_a_closed_session_is_kept(self):
        source = FakeSource()
        assert collect_inputs("X", "X.NS", "X", source, run_at=MADE_AT).bars == source.bars

    def test_news_summary_of_empty_collection(self):
        assert news_summary(NewsCollection(symbol="X", items=[]))["articles"] == []


class TestResolve:
    @pytest.mark.parametrize(
        "query, expected",
        [
            ("RELIANCE", ("RELIANCE", "RELIANCE.NS", "Reliance Industries Ltd")),
            ("reliance", ("RELIANCE", "RELIANCE.NS", "Reliance Industries Ltd")),
            ("RELIANCE.NS", ("RELIANCE", "RELIANCE.NS", "Reliance Industries Ltd")),
            ("tata motors", ("TATAMOTORS", "TATAMOTORS.NS", "Tata Motors Ltd")),
            ("xyzfoo", ("XYZFOO", "XYZFOO.NS", "XYZFOO")),
        ],
    )
    def test_ticker_or_company_name(self, query, expected):
        assert resolve_nse_ticker(query) == expected


class TestSecondAnalysis:
    """Analyze, let the forecast mature, analyze again: the demo of Plan.md §56/§73."""

    @pytest.fixture
    def runs(self, migrated_db, store):
        first = _analyze(migrated_db, MADE_AT)
        snapshot = store.get(first.state.forecast_id)
        second = _analyze(migrated_db, NOW_MATURED, price_source=_rally_prices(snapshot))
        return first, second

    def test_report_contains_every_acceptance_section_in_order(self, runs):
        _, second = runs
        report = second.report
        positions = [report.index(section) for section in ACCEPTANCE_SECTIONS]
        assert positions == sorted(positions)
        assert report.rstrip().endswith(DISCLAIMER)

    def test_previous_forecast_vs_actual_with_attribution(self, runs, store, outcome_store):
        first, second = runs
        report = second.report
        outcome = outcome_store.get(first.state.forecast_id)
        last = MemoryContext.model_validate(second.state.memory_context).last_review
        previous = store.get(first.state.forecast_id)
        # 1-4: previous forecast, actual outcome, forecast error, baseline error
        assert f"- Forecast `{first.state.forecast_id}`" in report
        assert "| | Quant baseline | Final forecast | Actual |" in report
        assert last.baseline_p50_price == previous.quant_baseline["p50_price"]
        assert last.baseline_signed_error_pct == outcome.baseline_signed_error_pct
        assert last.baseline_in_80pct_band == outcome.baseline_in_80pct_band
        assert last.baseline_direction_correct == (
            last.baseline_direction == outcome.realized_direction
        )
        assert f"| ₹{last.actual_close:.2f} |" in report
        assert (
            f"| Error vs P50 | {outcome.baseline_signed_error_pct:+.2f}% "
            f"| {outcome.signed_error_pct:+.2f}% | |"
        ) in report
        assert f"| Up/flat/down Brier | {outcome.baseline_loss:.4f} | {outcome.final_loss:.4f}" in (
            report
        )
        assert f"- Market move: Nifty {outcome.nifty_return_pct:+.2f}%" in report
        # 5-6: LLM value added and the postmortem
        assert "- LLM value added:" in report
        assert f"- Primary cause: **{last.primary_cause}**" in report

    def test_current_analysis_baseline_forecast_track_record(self, runs):
        _, second = runs
        report = second.report
        # 8-9: regime and the four analyst perspectives
        assert f"- Market regime: **{second.state.decision.result}**" in report
        for analyst in ("technical", "fundamental", "sentiment", "context"):
            assert f"- **{analyst}**:" in report
        # 12-13: track record and the benchmarks over the same forecasts
        assert "- **8 weeks = 26 weeks = All time**: 1 forecasts" in report
        for variant in ("Naive flat", "Quant uncalibrated", "Quant calibrated", "Final forecast"):
            assert f"| {variant} | " in report
        assert "these differences are not evidence that one variant is better" in report

    def test_benchmarks_come_from_the_track_record(self, runs, store, outcome_store):
        from stock_analysis.learning import build_evaluation

        _, second = runs
        evaluation = build_evaluation(store, outcome_store, as_of=NOW_MATURED, ticker="RELIANCE")
        window = evaluation.track_record.windows[-1]
        metrics = MemoryContext.model_validate(second.state.memory_context)
        stored = metrics.track_record.outcome_metrics.windows[-1]
        assert [v.variant for v in stored.variants] == [v.variant for v in window.variants]
        for mine, theirs in zip(stored.variants, window.variants, strict=True):
            assert mine.direction == theirs.direction
            assert mine.brier == theirs.brier
            assert mine.coverage_80pct == theirs.coverage_80pct

    def test_live_inputs_and_bars_are_what_the_forecast_used(self, runs, store):
        _, second = runs
        snapshot = store.get(second.state.forecast_id)
        source_bars = _bars()
        assert snapshot.last_close == pytest.approx(float(source_bars[-1].close))
        assert snapshot.as_of_date == LAST_BAR
        assert snapshot.data_inputs["fundamentals"]["pe_ratio"] == 21.5
        assert snapshot.data_inputs["technical_indicators"]["date"] == LAST_BAR.isoformat()
        assert snapshot.data_inputs["news"]["article_count"] == 12
        assert store.get_price_history(second.state.price_snapshot["history_sha256"]) is not None

    def test_every_number_traces_to_the_snapshot(self, runs, store):
        _, second = runs
        snapshot = store.get(second.state.forecast_id)
        assert find_unsupported_numbers(second.report, snapshot) == []


class TestAdaptationEffect:
    def test_calibration_effect_on_the_range_is_shown(self, migrated_db, store):
        LearningStore(migrated_db).save_calibration(
            CalibrationParams(
                ticker="RELIANCE",
                version=1,
                vol_multiplier=1.18,
                p50_bias_shift_pct=0.0,
                previous_vol_multiplier=1.0,
                previous_p50_bias_shift_pct=0.0,
                n_samples=8,
                reason=["Realized moves were larger than predicted in 6 of 8 forecasts"],
                evidence={"forecast_ids": []},
                calibrator_version=CALIBRATOR_VERSION,
                created_at=MADE_AT - timedelta(days=1),
            )
        )
        # no review: it would re-estimate this evidence-free calibration back to 1.0
        result = _analyze(migrated_db, MADE_AT, review=False)
        snapshot = store.get(result.state.forecast_id)
        shadow, quant = snapshot.shadow_baseline, snapshot.quant_baseline
        assert quant["p90_price"] - quant["p10_price"] > shadow["p90_price"] - shadow["p10_price"]
        assert (
            f"before calibration ₹{shadow['p10_price']:.2f}–₹{shadow['p90_price']:.2f}, "
            f"after ₹{quant['p10_price']:.2f}–₹{quant['p90_price']:.2f}"
        ) in result.report
        assert find_unsupported_numbers(result.report, snapshot) == []


class TestQuantOnlyAndFailures:
    def test_no_llm_disables_every_analyst(self, migrated_db):
        source = FakeSource()
        result = _analyze(migrated_db, MADE_AT, use_llm=False, llm_factory=None, source=source)
        assert result.report is not None
        assert set(result.state.disabled_analysts) == {
            "technical",
            "fundamental",
            "sentiment",
            "context",
        }
        assert source.calls == ["prices"]
        assert "quant-only run (LLM disabled)" in result.report
        assert result.state.final_forecast["adjustment_applied"] is False

    def test_no_llm_review_uses_rules_only_postmortems(self, migrated_db, store, monkeypatch):
        import stock_analysis.langgraph.runner as runner

        seen = {}
        real = runner.make_pre_run_reviewer

        def spy(*stores, **kwargs):
            seen.update(kwargs)
            return real(*stores, **kwargs)

        monkeypatch.setattr(runner, "make_pre_run_reviewer", spy)
        _analyze(migrated_db, MADE_AT, use_llm=False, llm_factory=None)
        assert seen["llm_factory"] is None

    def test_unavailable_input_is_shown_under_data_quality(self, migrated_db, store):
        result = _analyze(migrated_db, MADE_AT, source=FakeSource(fail={"news"}))
        data_quality = result.report[result.report.index("## Data quality") :]
        assert "- Input not available (news): RuntimeError: news source down" in data_quality
        snapshot = store.get(result.state.forecast_id)
        assert find_unsupported_numbers(result.report, snapshot) == []

    def test_no_prices_means_no_forecast(self, migrated_db, store):
        result = _analyze(migrated_db, MADE_AT, source=FakeSource(fail={"prices"}))
        assert result.report is None
        assert "prices source down" in result.error
        assert result.unavailable_inputs["prices"] == "RuntimeError: prices source down"
        assert store.list_for_ticker("RELIANCE") == []


class TestReportCompatibility:
    """Snapshots stored before Phase 17 hold memory without the new fields."""

    def _context(self, **update):
        review = LastReview(
            forecast_id="f1",
            as_of_date=LAST_BAR,
            target_date=LAST_BAR + timedelta(days=7),
            evaluated_at=NOW_MATURED,
            status="scored",
            last_close=100.0,
            prob_up=0.5,
            prob_flat=0.2,
            prob_down=0.3,
            p10_price=95.0,
            p50_price=101.0,
            p90_price=106.0,
            adjustment_applied=False,
            calibration_version=0,
            predicted_direction="up",
            realized_direction="up",
            direction_correct=True,
            actual_close=103.0,
            signed_error_pct=1.98,
            in_80pct_band=True,
        )
        window = WindowMetrics(
            label="All time", n=1, direction=HitRate(n=1, hits=1), finding="too few"
        )
        track = TrackRecord(
            ticker="RELIANCE",
            forecasts_made=1,
            forecasts_scored=1,
            forecasts_awaiting_outcome=0,
            outcome_metrics=OutcomeMetrics(n_forecasts=1, min_samples=20, windows=[window]),
        )
        return MemoryContext(
            loaded=True,
            last_review=review,
            track_record=track,
            learning_loaded=["last_review", "track_record"],
            **update,
        )

    def test_review_without_baseline_fields_shows_na(self):
        lines = "\n".join(_last_review_lines(self._context()))
        assert "| P50 / close | n/a | ₹101.00 | ₹103.00 |" in lines
        assert "| P10–P90 range | n/a | ₹95.00–₹106.00 (inside) | |" in lines
        assert "| Direction | n/a | UP (P(up) 50.0%) — correct | UP |" in lines

    def test_track_record_without_variants_has_no_benchmarks(self):
        assert _benchmark_lines(self._context()) == [_NOT_LOADED]


@pytest.mark.usefixtures("migrated_db")
class TestAnalyzeCommand:
    @pytest.fixture
    def calls(self, monkeypatch):
        calls = []

        def fake_run(query, db, **kwargs):
            calls.append((query, dict(kwargs)))
            if not kwargs["use_llm"]:
                kwargs["llm_factory"] = None
            return _analyze(db, MADE_AT, query=query, **kwargs)

        monkeypatch.setattr(analysis, "run_analysis", fake_run)
        return calls

    def test_prints_and_writes_the_report(self, temp_db_path, tmp_path, calls):
        out = tmp_path / "reports" / "reliance.md"
        result = CliRunner().invoke(
            app, ["analyze", "RELIANCE", "--database", str(temp_db_path), "-o", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert "Forecast report: RELIANCE" in result.output
        assert calls == [("RELIANCE", {"use_llm": True, "review": True})]
        text = out.read_text()
        assert text.startswith("# Forecast report: RELIANCE (Reliance Industries Ltd)")
        assert DISCLAIMER in text

    def test_flags_are_passed_on(self, temp_db_path, calls):
        result = CliRunner().invoke(
            app,
            ["analyze", "tata motors", "--no-llm", "--no-review", "--database", str(temp_db_path)],
        )
        assert result.exit_code == 0, result.output
        assert calls == [("tata motors", {"use_llm": False, "review": False})]

    def test_warns_when_the_llm_has_no_api_key(self, temp_db_path, calls, monkeypatch):
        from stock_analysis.config.settings import get_settings

        monkeypatch.setenv("GEMINI_API_KEY", "")
        get_settings.cache_clear()
        result = CliRunner().invoke(app, ["analyze", "RELIANCE", "--database", str(temp_db_path)])
        assert result.exit_code == 0, result.output
        assert "Warning: GEMINI_API_KEY is not set" in result.output
        quant_only = CliRunner().invoke(
            app, ["analyze", "RELIANCE", "--no-llm", "--database", str(temp_db_path)]
        )
        assert "GEMINI_API_KEY" not in quant_only.output

    def test_missing_database(self, tmp_path):
        result = CliRunner().invoke(
            app, ["analyze", "RELIANCE", "--database", str(tmp_path / "none.db")]
        )
        assert result.exit_code == 1
        assert "No database at" in result.output

    def test_failed_forecast_exits_nonzero(self, temp_db_path, monkeypatch):
        def no_prices(query, db, **kwargs):
            return _analyze(db, MADE_AT, source=FakeSource(fail={"prices"}), **kwargs)

        monkeypatch.setattr(analysis, "run_analysis", no_prices)
        result = CliRunner().invoke(app, ["analyze", "RELIANCE", "--database", str(temp_db_path)])
        assert result.exit_code == 1
        assert "Input not available (prices): RuntimeError: prices source down" in result.output
        assert "No forecast for RELIANCE.NS" in result.output

    def test_stock_agents_alias(self):
        pyproject = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
        scripts = pyproject["project"]["scripts"]
        assert scripts["stock-agents"] == scripts["stock-analysis"] == "stock_analysis.cli.main:app"
