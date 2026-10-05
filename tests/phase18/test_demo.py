"""Phase 18 "done when": the full demo runs on three stocks (backtest, review, adaptation,
analysis with every §76 item), offline on synthetic NSE history."""

from datetime import date

import pytest

from stock_analysis.backtest import HistoricalMarketData
from stock_analysis.backtest.data import INDIA_VIX_SYMBOL
from stock_analysis.demo import (
    ACCEPTANCE_ITEMS,
    acceptance_checklist,
    render_demo_summary,
    run_demo,
    run_stock_demo,
)
from stock_analysis.review.prices import NIFTY_50_SYMBOL
from stock_analysis.snapshots import DISCLAIMER
from tests.backtest_helpers import NOW, SYMBOL, synthetic_history
from tests.outcome_helpers import FakePriceSource

TICKERS = ("RELIANCE", "INFY", "HDFCBANK")
LAST_SESSION = date(2026, 9, 30)  # last session completed at NOW


class SyntheticMarket:
    """Serves the same synthetic history under any stock symbol: a ``history_loader``
    for the backtest and an ``InputSource`` for the analysis."""

    def __init__(self, **history_options):
        self.data = synthetic_history(weekly_swing_pct=4.0, **history_options)
        self.stock_bars = self.data.as_of(LAST_SESSION).bars(SYMBOL)
        self.price_requests: list[str] = []

    def __call__(self, symbols, start, end):
        source = {
            s: self.data.as_of(end).bars(SYMBOL if s not in (NIFTY_50_SYMBOL, INDIA_VIX_SYMBOL)
                                          else s)
            for s in symbols
        }  # fmt: skip
        return HistoricalMarketData(source)

    def prices(self, symbol):
        self.price_requests.append(symbol)
        return self.stock_bars

    def fundamentals(self, symbol):  # pragma: no cover - quant-only demo never asks
        raise AssertionError("disabled analysts' inputs are not fetched")

    news = market_context = fundamentals


def _demo(tmp_path, tickers=TICKERS, weeks=10, **options):
    market = SyntheticMarket()
    results = run_demo(
        tickers,
        weeks=weeks,
        output_dir=tmp_path,
        use_llm=False,
        now=NOW,
        history_loader=market,
        input_source=market,
        analysis_options={"price_source": FakePriceSource({})},
        **options,
    )
    return market, results


class TestThreeStockDemo:
    @pytest.fixture(scope="class")
    def demo(self, tmp_path_factory):
        return _demo(tmp_path_factory.mktemp("demo"))

    def test_every_stock_passes(self, demo):
        _, results = demo
        assert [r.ticker for r in results] == list(TICKERS)
        for r in results:
            assert r.errors == []
            assert r.passed, r.missing_items

    def test_backtest_review_and_adaptation_ran(self, demo):
        _, results = demo
        for r in results:
            assert r.scored_weeks >= 8
            assert len(r.backtest.weeks) == 10
            # weekly swings beyond daily volatility force a calibration (adaptation)
            assert r.backtest.calibration_history
            assert "## Calibration versions for" in r.reports["calibration"].read_text()

    def test_every_acceptance_item_is_shown(self, demo):
        _, results = demo
        for r in results:
            assert r.checklist == {item: True for item, _ in ACCEPTANCE_ITEMS}
            report = r.reports["analysis"].read_text()
            assert report == r.analysis.report
            assert "- Calibration v" in report  # adaptation from the backtest's learning

    def test_each_stock_has_its_own_database_and_reports(self, demo, tmp_path_factory):
        _, results = demo
        paths = {r.database_path for r in results}
        assert len(paths) == 3 and all(p.exists() for p in paths)
        for r in results:
            assert set(r.reports) == {"backtest", "evaluate", "calibration", "analysis"}
            assert all(p.parent == r.database_path.parent for p in r.reports.values())

    def test_analysis_forecast_uses_its_own_symbol(self, demo):
        market, _ = demo
        assert market.price_requests == ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS"]

    def test_quant_only_demo_makes_no_llm_calls(self, demo):
        _, results = demo
        assert all(r.llm_usage.calls == 0 for r in results)

    def test_summary(self, demo):
        _, results = demo
        summary = render_demo_summary(results)
        for ticker in TICKERS:
            assert f"| {ticker} |" in summary
        assert summary.count("| pass |") == 3
        assert f"{len(ACCEPTANCE_ITEMS)}/{len(ACCEPTANCE_ITEMS)}" in summary
        assert DISCLAIMER in summary


class TestFailures:
    def test_backtest_failure_is_reported_not_raised(self, tmp_path):
        result = run_stock_demo(
            "RELIANCE",
            weeks=4,
            output_dir=tmp_path,
            use_llm=False,
            now=NOW,
            history_loader=lambda *a: HistoricalMarketData({}),
        )
        assert not result.passed
        assert result.errors and result.errors[0].startswith("backtest:")
        assert "FAIL" in render_demo_summary([result])

    def test_unexpected_failure_of_one_stock_does_not_stop_the_others(self, tmp_path, monkeypatch):
        import stock_analysis.demo as demo

        real = demo.run_stock_demo

        def flaky(query, **kwargs):
            if query == "INFY":
                raise RuntimeError("disk full")
            return real(query, **kwargs)

        monkeypatch.setattr(demo, "run_stock_demo", flaky)
        _, results = _demo(tmp_path, tickers=("INFY", "RELIANCE"), weeks=3)
        assert [r.ticker for r in results] == ["INFY", "RELIANCE"]
        assert results[0].errors == ["unexpected RuntimeError: disk full"]
        assert results[1].backtest is not None and not results[1].errors

    def test_checklist_names_missing_items(self):
        checklist = acceptance_checklist("## Quant baseline\n## Final forecast\n" + DISCLAIMER)
        assert checklist["Quant baseline"] and checklist["Disclaimer"]
        assert not checklist["Postmortem"] and not checklist["Track record"]
