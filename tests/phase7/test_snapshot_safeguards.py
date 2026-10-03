"""Phase 7: advice-wording flags, data-quality warnings and prompt rules."""

from datetime import UTC, date, datetime

import pytest

from stock_analysis.langgraph.workflow import build_predictor_prompt
from stock_analysis.market.calendar import TradingCalendar
from stock_analysis.snapshots import (
    build_forecast_snapshot,
    find_unsupported_numbers,
    render_forecast_report,
)
from stock_analysis.snapshots.report import ADVICE_WORDING_NOTE, has_advice_wording
from stock_analysis.versions import ANALYST_PROMPT_FILES, PROMPTS_DIR
from tests.forecast_helpers import MADE_AT

NO_ADVICE_RULE = "never phrase output as a buy, sell or hold recommendation"
CORRECTED_AT = datetime(2026, 9, 30, tzinfo=UTC)


def _with_key_point(state, text):
    report = {**state.technical_report, "key_points": [text]}
    return state.model_copy(update={"technical_report": report})


class TestAdviceWording:
    @pytest.mark.parametrize(
        "text",
        [
            "Strong buy signal on breakout",
            "Technical sell rating after breakdown",
            "We recommend buying on dips",
            "Price target of the move is above resistance",
            "Accumulate on weakness",
        ],
    )
    def test_recommendation_phrasing_is_flagged(self, adjusted_state, text):
        snap = build_forecast_snapshot(_with_key_point(adjusted_state, text), made_at=MADE_AT)
        report = render_forecast_report(snap)

        assert has_advice_wording(snap)
        assert f"- Warning: {ADVICE_WORDING_NOTE}" in report
        assert text in report  # evidence is quoted verbatim, never rewritten
        assert find_unsupported_numbers(report, snap) == []

    @pytest.mark.parametrize(
        "text",
        [
            "Broad sell-off in metals",
            "Support holds near the 50-day average",
            "Buyers absorbed supply",
        ],
    )
    def test_market_vocabulary_is_not_flagged(self, adjusted_state, text):
        snap = build_forecast_snapshot(_with_key_point(adjusted_state, text), made_at=MADE_AT)
        assert not has_advice_wording(snap)
        assert ADVICE_WORDING_NOTE not in render_forecast_report(snap)

    def test_prompts_forbid_recommendations(self):
        for filename in ANALYST_PROMPT_FILES.values():
            assert NO_ADVICE_RULE in (PROMPTS_DIR / filename).read_text()
        assert NO_ADVICE_RULE in build_predictor_prompt({}, [], [], [], [])


class TestDataQualityWarnings:
    def test_standard_forecast_has_no_warnings(self, snapshot):
        assert snapshot.data_quality["warnings"] == []

    def test_calendar_coverage_warning(self, adjusted_state):
        price = {**adjusted_state.price_snapshot, "last_date": "2026-12-28"}
        state = adjusted_state.model_copy(update={"price_snapshot": price})
        snap = build_forecast_snapshot(state, made_at=datetime(2026, 12, 28, 12, tzinfo=UTC))

        assert snap.target_date.year == 2027
        assert any("does not cover 2027" in w for w in snap.data_quality["warnings"])
        report = render_forecast_report(snap)
        assert "does not cover 2027" in report
        assert find_unsupported_numbers(report, snap) == []

    def test_intraday_bar_warning(self, adjusted_state):
        made_at = datetime(2026, 9, 29, 4, 30, tzinfo=UTC)  # 10:00 IST on the as-of day
        snap = build_forecast_snapshot(adjusted_state, made_at=made_at)
        assert any("intraday" in w for w in snap.data_quality["warnings"])

    def test_calendar_covers(self):
        calendar = TradingCalendar()
        assert calendar.covers(date(2026, 12, 31))
        assert not calendar.covers(date(2027, 1, 1))


class TestListForTicker:
    def test_originals_only_by_default(self, store, snapshot):
        store.save(snapshot)
        corrected = store.record_correction(
            snapshot.forecast_id, reason="name fix", corrected_at=CORRECTED_AT
        )

        assert [s.forecast_id for s in store.list_for_ticker("RELIANCE")] == [snapshot.forecast_id]
        assert [
            s.forecast_id for s in store.list_for_ticker("RELIANCE", include_corrections=True)
        ] == [corrected.forecast_id, snapshot.forecast_id]
