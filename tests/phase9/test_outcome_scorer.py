"""Plan.md Phase 9: scoring a stored forecast against synthetic actual prices."""

from datetime import UTC, date, datetime, timedelta

import pytest

from stock_analysis.review import PriceBar, PriceSeries, metrics, score_forecast
from stock_analysis.review.outcome_scorer import is_matured, last_completed_trading_date
from stock_analysis.schemas.outcome import OutcomeStatus
from stock_analysis.snapshots import build_forecast_snapshot
from tests.forecast_helpers import MADE_AT
from tests.outcome_helpers import NOW_MATURED, nifty_series, path_series, window_dates

UP_PATH = [0.5, 1.0, 1.8, 2.4, 3.0]  # ends +3% (realized "up")
DOWN_PATH = [-0.5, -1.0, -1.8, -2.4, -3.0]


def _score(snapshot, stock, nifty=None, now=NOW_MATURED):
    return score_forecast(snapshot, stock, nifty, now=now)


class TestMaturity:
    @pytest.mark.parametrize(
        ("now_utc", "expected"),
        [
            (datetime(2026, 10, 7, 9, 0, tzinfo=UTC), date(2026, 10, 6)),  # 14:30 IST, open
            (datetime(2026, 10, 7, 10, 30, tzinfo=UTC), date(2026, 10, 6)),  # 16:00, data pending
            (datetime(2026, 10, 7, 11, 30, tzinfo=UTC), date(2026, 10, 7)),  # 17:00, published
            (datetime(2026, 10, 2, 12, 0, tzinfo=UTC), date(2026, 10, 1)),  # Gandhi Jayanti
            (datetime(2026, 10, 4, 12, 0, tzinfo=UTC), date(2026, 10, 1)),  # Sunday
        ],
    )
    def test_last_completed_trading_date(self, now_utc, expected):
        assert last_completed_trading_date(now_utc) == expected

    def test_matured_only_after_target_session(self, snapshot):
        assert snapshot.target_date == date(2026, 10, 7)
        assert not is_matured(snapshot.target_date, datetime(2026, 10, 7, 9, 0, tzinfo=UTC))
        assert is_matured(snapshot.target_date, datetime(2026, 10, 7, 11, 30, tzinfo=UTC))

    def test_not_matured_is_unresolved(self, snapshot):
        outcome = _score(
            snapshot, path_series(snapshot, UP_PATH), now=datetime(2026, 10, 7, 9, 0, tzinfo=UTC)
        )
        assert outcome.status == OutcomeStatus.UNRESOLVED
        assert "has not completed" in outcome.invalid_reason


class TestTradingWindow:
    def test_uses_exactly_the_forecast_sessions(self, snapshot):
        outcome = _score(snapshot, path_series(snapshot, UP_PATH))
        assert outcome.status == OutcomeStatus.SCORED
        assert [d.date for d in outcome.daily] == window_dates(snapshot)[1:]
        assert date(2026, 10, 2) not in [d.date for d in outcome.daily]  # holiday skipped

    def test_later_prices_are_never_used(self, snapshot):
        """Point-in-time: a bar after the target session cannot change the evaluation."""
        series = path_series(snapshot, UP_PATH)
        leaked = PriceSeries(
            symbol=series.symbol,
            bars=[*series.bars, PriceBar(date=date(2026, 10, 8), close=snapshot.last_close * 5)],
        )
        clean, with_future = _score(snapshot, series), _score(snapshot, leaked)
        assert with_future.model_dump(exclude={"evaluated_at"}) == clean.model_dump(
            exclude={"evaluated_at"}
        )

    def test_missing_session_is_unresolved_then_invalid(self, snapshot):
        series = path_series(snapshot, UP_PATH)
        gap = PriceSeries(
            symbol=series.symbol, bars=[b for b in series.bars if b.date != date(2026, 10, 5)]
        )
        recent = _score(snapshot, gap)
        assert recent.status == OutcomeStatus.UNRESOLVED
        assert "2026-10-05" in recent.invalid_reason

        much_later = _score(snapshot, gap, now=NOW_MATURED + timedelta(days=14))
        assert much_later.status == OutcomeStatus.INVALID

    def test_missing_as_of_bar_is_unresolved(self, snapshot):
        series = path_series(snapshot, UP_PATH)
        no_base = PriceSeries(symbol=series.symbol, bars=series.bars[1:])
        outcome = _score(snapshot, no_base)
        assert outcome.status == OutcomeStatus.UNRESOLVED
        assert snapshot.as_of_date.isoformat() in outcome.invalid_reason

    def test_unexpected_session_is_invalid(self, snapshot):
        series = path_series(snapshot, UP_PATH)
        extra = PriceSeries(
            symbol=series.symbol,
            bars=[*series.bars, PriceBar(date=date(2026, 10, 2), close=snapshot.last_close)],
        )
        outcome = _score(snapshot, extra)
        assert outcome.status == OutcomeStatus.INVALID
        assert "did not expect: 2026-10-02" in outcome.invalid_reason

    def test_no_data_is_unresolved(self, snapshot):
        outcome = _score(snapshot, PriceSeries(symbol="RELIANCE.NS", error="timeout"))
        assert outcome.status == OutcomeStatus.UNRESOLVED
        assert "timeout" in outcome.invalid_reason


class TestMetrics:
    def test_return_error_direction_and_band(self, snapshot):
        outcome = _score(snapshot, path_series(snapshot, UP_PATH))
        final = snapshot.final_forecast
        basis = snapshot.last_close * 1.03

        assert outcome.actual_return_pct == pytest.approx(3.0)
        assert outcome.actual_close == pytest.approx(basis)
        assert outcome.realized_direction == "up"
        predicted = metrics.predicted_direction(final.prob_up, final.prob_flat, final.prob_down)
        assert outcome.predicted_direction == predicted
        assert outcome.direction_correct is (predicted == "up")
        assert outcome.signed_error_pct == pytest.approx((basis / final.p50_price - 1) * 100)
        assert outcome.abs_error_pct == pytest.approx(abs(outcome.signed_error_pct))
        assert outcome.in_80pct_band is (final.p10_price <= basis <= final.p90_price)

    def test_band_hit_and_miss(self, snapshot):
        final = snapshot.final_forecast
        at_p50 = (final.p50_price / snapshot.last_close - 1) * 100
        hit = _score(snapshot, path_series(snapshot, [0, 0, 0, 0, at_p50]))
        miss = _score(snapshot, path_series(snapshot, [5, 10, 15, 20, 30]))
        assert hit.in_80pct_band is True
        assert hit.pit_percentile == pytest.approx(0.5)
        assert miss.in_80pct_band is False
        assert miss.pit_percentile > 0.99

    def test_brier_pinball_pit_match_definitions(self, snapshot):
        outcome = _score(snapshot, path_series(snapshot, DOWN_PATH))
        final = snapshot.final_forecast
        basis = snapshot.last_close * 0.97
        assert outcome.brier == pytest.approx(
            metrics.brier_score(final.prob_up, final.prob_flat, final.prob_down, "down")
        )
        assert outcome.pinball == pytest.approx(
            metrics.quantile_losses(
                snapshot.last_close, final.p10_price, final.p50_price, final.p90_price, -3.0
            )
        )
        assert outcome.pit_percentile == pytest.approx(
            metrics.pit_percentile(basis, final.p10_price, final.p50_price, final.p90_price)
        )

    def test_volatility_ratio(self, snapshot):
        series = path_series(snapshot, [2, -1, 3, -2, 1])
        outcome = _score(snapshot, series)
        realized = metrics.realized_weekly_vol_pct([b.close for b in series.bars])
        assert outcome.realized_vol_pct == pytest.approx(realized)
        assert outcome.vol_ratio == pytest.approx(realized / snapshot.final_forecast.weekly_vol_pct)

    def test_daily_band_breaches(self, snapshot):
        final_p90 = snapshot.daily_predictions[0].p90_price
        spike = (final_p90 / snapshot.last_close - 1) * 100 + 1.0  # day 1 above its P90
        outcome = _score(snapshot, path_series(snapshot, [spike, 0, 0, 0, 0]))
        assert outcome.daily[0].in_daily_band is False
        assert outcome.daily_band_breaches == 1

    def test_excess_vs_nifty(self, snapshot):
        outcome = _score(snapshot, path_series(snapshot, UP_PATH), nifty_series(snapshot, 1.0))
        assert outcome.nifty_return_pct == pytest.approx(1.0)
        assert outcome.excess_vs_nifty_pct == pytest.approx(2.0)
        assert outcome.beta_adjusted_excess_pct is None  # no beta in the forecast inputs

    def test_missing_nifty_still_scores(self, snapshot):
        outcome = _score(snapshot, path_series(snapshot, UP_PATH), None)
        assert outcome.status == OutcomeStatus.SCORED
        assert outcome.excess_vs_nifty_pct is None
        assert any("Nifty" in c for c in outcome.validity_checks)

    def test_analyst_hits(self, snapshot):
        up = _score(snapshot, path_series(snapshot, UP_PATH))
        down = _score(snapshot, path_series(snapshot, DOWN_PATH))
        assert set(up.analyst_hits.values()) == {True}  # all analysts were bullish
        assert set(down.analyst_hits.values()) == {False}


class TestBaselineVsFinal:
    def test_llm_value_added_is_baseline_minus_final_loss(self, snapshot):
        up = _score(snapshot, path_series(snapshot, UP_PATH))
        down = _score(snapshot, path_series(snapshot, DOWN_PATH))

        for outcome in (up, down):
            assert outcome.loss_metric == "brier"
            assert outcome.baseline_loss == outcome.baseline_brier
            assert outcome.final_loss == outcome.brier
            assert outcome.llm_value_added == pytest.approx(
                outcome.baseline_loss - outcome.final_loss
            )
        # the predictor tilted toward "up": it helps when the stock rises, hurts when it falls
        assert up.llm_value_added > 0
        assert down.llm_value_added < 0
        # quantiles were not adjusted, so quantile losses are identical
        assert up.llm_value_added_pinball == pytest.approx(0.0)

    def test_baseline_scored_on_same_outcome(self, snapshot):
        outcome = _score(snapshot, path_series(snapshot, UP_PATH))
        quant = snapshot.quant_baseline
        assert outcome.baseline_brier == pytest.approx(
            metrics.brier_score(quant["prob_up"], quant["prob_flat"], quant["prob_down"], "up")
        )
        assert outcome.baseline_signed_error_pct == pytest.approx(
            (snapshot.last_close * 1.03 / quant["p50_price"] - 1) * 100
        )

    def test_unadjusted_forecast_adds_nothing(self, guardrail_state):
        snap = build_forecast_snapshot(guardrail_state, made_at=MADE_AT)
        outcome = _score(snap, path_series(snap, UP_PATH))
        assert outcome.llm_value_added == 0.0
        assert outcome.llm_value_added_pinball == 0.0


class TestCorporateActions:
    def test_dividend_in_window_is_recorded(self, snapshot):
        ex_date = window_dates(snapshot)[3]
        series = path_series(
            snapshot, UP_PATH, actions={ex_date: PriceBar(date=ex_date, close=0, dividend=10.0)}
        )
        outcome = _score(snapshot, series)
        assert outcome.status == OutcomeStatus.SCORED
        assert outcome.corporate_actions == [
            {"date": ex_date.isoformat(), "type": "dividend", "amount": 10.0}
        ]
        assert any("Corporate actions" in c for c in outcome.validity_checks)

    def test_unadjusted_split_is_invalid(self, snapshot):
        split_date = window_dates(snapshot)[2]
        series = path_series(
            snapshot,
            [0.5, -49.5, -49.0, -48.8, -48.5],
            actions={split_date: PriceBar(date=split_date, close=0, split=2.0)},
        )
        outcome = _score(snapshot, series)
        assert outcome.status == OutcomeStatus.INVALID
        assert "split not reflected" in outcome.invalid_reason

    def test_unexplained_jump_is_invalid(self, snapshot):
        outcome = _score(snapshot, path_series(snapshot, [0, -45, -45, -45, -45]))
        assert outcome.status == OutcomeStatus.INVALID
        assert "possible unadjusted corporate action" in outcome.invalid_reason

    def test_history_readjusted_after_forecast(self, snapshot):
        """A later dividend scales the whole history; returns and errors must not change."""
        clean = _score(snapshot, path_series(snapshot, UP_PATH))
        rescaled = _score(snapshot, path_series(snapshot, UP_PATH, base=snapshot.last_close * 0.98))

        assert rescaled.adjustment_factor == pytest.approx(0.98)
        assert rescaled.actual_return_pct == pytest.approx(clean.actual_return_pct)
        assert rescaled.actual_close_on_forecast_basis == pytest.approx(
            clean.actual_close_on_forecast_basis
        )
        assert rescaled.signed_error_pct == pytest.approx(clean.signed_error_pct)
        assert rescaled.in_80pct_band == clean.in_80pct_band
        assert any("re-adjusted" in c for c in rescaled.validity_checks)


class TestAttributionAndIntegrity:
    def test_decision_engine_attribution(self, snapshot, guardrail_state):
        adjusted = _score(snapshot, path_series(snapshot, UP_PATH))
        assert adjusted.decision_engine == "rules@0.1.0"
        assert adjusted.decision_engine_enabled is True
        assert adjusted.adjustment_gate_decision == "allow_adjustment"
        assert adjusted.market_regime_decision == "trending_up"
        assert adjusted.adjustment_applied is True

        gated_snap = build_forecast_snapshot(guardrail_state, made_at=MADE_AT)
        gated = _score(gated_snap, path_series(gated_snap, UP_PATH))
        assert gated.decision_engine_enabled is False
        assert gated.adjustment_gate_decision == "no_adjustment"
        assert gated.adjustment_applied is False

    def test_scoring_never_modifies_the_forecast(self, snapshot):
        before = snapshot.model_dump(mode="json")
        _score(snapshot, path_series(snapshot, UP_PATH), nifty_series(snapshot, 1.0))
        assert snapshot.model_dump(mode="json") == before
