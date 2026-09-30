"""Unit tests for Phase 5 analyst Pydantic schemas and GraphState.

pytest used for validation error testing.
"""

import pytest

from stock_analysis.schemas.analyst_reports import (
    AnalystReport,
    AnalystType,
    AnalystStance,
    DecisionResult,
    DecisionType,
    MarketRegime,
    RiskCategory,
)
from stock_analysis.schemas.graph_state import GraphState


class TestAnalystReport:
    """Tests for AnalystReport schema validation."""

    def test_valid_technical_report(self):
        """Valid technical analyst report should pass validation."""
        rpt = AnalystReport(
            analyst=AnalystType.TECHNICAL,
            stance=AnalystStance.BULLISH,
            confidence=0.75,
            key_points=["RSI oversold", "Price above EMA"],
            evidence=["RSI_14=28.5", "price_close=245.3 above EMA_20=240.1"],
            risks=["RSI may reverse"],
            data_gaps=["VWAP not available"],
        )
        assert rpt.analyst == AnalystType.TECHNICAL
        assert rpt.stance == AnalystStance.BULLISH
        assert 0.0 <= rpt.confidence <= 1.0
        assert len(rpt.key_points) <= 5
        assert len(rpt.evidence) <= 10
        assert len(rpt.risks) <= 3

    def test_confidence_bounds(self):
        """Confidence must be between 0.0 and 1.0."""
        # Valid confidence
        rpt = AnalystReport(
            analyst=AnalystType.FUNDAMENTAL,
            stance=AnalystStance.NEUTRAL,
            confidence=0.5,
            key_points=[],
            evidence=[],
            risks=[],
        )
        assert rpt.confidence == 0.5

        # Invalid confidence should raise
        try:
            AnalystReport(
                analyst=AnalystType.SENTIMENT,
                stance=AnalystStance.BEARISH,
                confidence=1.5,  # type: ignore
                key_points=[],
                evidence=[],
                risks=[],
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected

        try:
            AnalystReport(
                analyst=AnalystType.CONTEXT,
                stance=AnalystStance.BULLISH,
                confidence=-0.1,  # type: ignore
                key_points=[],
                evidence=[],
                risks=[],
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected

    def test_key_points_max5(self):
        """Key points must not exceed 5 items."""
        with pytest.raises(Exception):  # pydantic validation error
            AnalystReport(
                analyst=AnalystType.TECHNICAL,
                stance=AnalystStance.BULLISH,
                confidence=0.5,
                key_points=["p1", "p2", "p3", "p4", "p5", "p6"],  # type: ignore
                evidence=[],
                risks=[],
            )

    def test_risks_max3(self):
        """Risks must not exceed 3 items."""
        with pytest.raises(Exception):  # pydantic validation error
            AnalystReport(
                analyst=AnalystType.FUNDAMENTAL,
                stance=AnalystStance.NEUTRAL,
                confidence=0.5,
                key_points=[],
                evidence=[],
                risks=["r1", "r2", "r3", "r4"],  # type: ignore
            )

    def test_data_gaps_default_empty(self):
        """Data gaps should default to empty list."""
        rpt = AnalystReport(
            analyst=AnalystType.TECHNICAL,
            stance=AnalystStance.NEUTRAL,
            confidence=0.5,
            key_points=[],
            evidence=[],
            risks=[],
        )
        assert rpt.data_gaps == []

    def test_all_analyst_types(self):
        """All analyst types should be valid."""
        for at in AnalystType:
            rpt = AnalystReport(
                analyst=at,
                stance=AnalystStance.NEUTRAL,
                confidence=0.5,
                key_points=[],
                evidence=[],
                risks=[],
            )
            assert rpt.analyst == at

    def test_all_stances(self):
        """All stances should be valid."""
        for stance in AnalystStance:
            rpt = AnalystReport(
                analyst=AnalystType.TECHNICAL,
                stance=stance,
                confidence=0.5,
                key_points=[],
                evidence=[],
                risks=[],
            )
            assert rpt.stance == stance

    def test_all_decision_types(self):
        """All decision types should be valid."""
        for dt in DecisionType:
            _ = DecisionResult(
                decision_type=dt,
                result="test",
                confidence=0.5,
                rationale="test",
            )

    def test_all_market_regimes(self):
        """All market regimes should be valid."""
        for mr in MarketRegime:
            r = DecisionResult(
                decision_type=DecisionType.MARKET_REGIME,
                result=mr.value,
                confidence=0.5,
                rationale="test",
            )
            assert r.result == mr.value

    def test_all_risk_categories(self):
        """All risk categories should be valid."""
        for rc in RiskCategory:
            r = DecisionResult(
                decision_type=DecisionType.RISK_CATEGORY,
                result=rc.value,
                confidence=0.5,
                rationale="test",
            )
            assert r.result == rc.value


class TestGraphState:
    """Tests for GraphState schema validation."""

    def test_graph_state_basic(self):
        """GraphState should create with basic fields."""
        state = GraphState(
            symbol="RELIANCE",
            resolved_symbol="RELIANCE",
        )
        assert state.symbol == "RELIANCE"
        assert state.resolved_symbol == "RELIANCE"
        assert state.collectors_complete is False
        assert state.technical_indicators_summary is None
        assert state.fundamentals_summary is None
        assert state.news_summary is None
        assert state.market_context_summary is None
        assert state.technical_report is None
        assert state.fundamental_report is None
        assert state.sentiment_report is None
        assert state.context_report is None
        assert state.decision is None
        assert state.quant_baseline is None
        assert state.final_report is None

    def test_graph_state_with_summaries(self):
        """GraphState should accept summary dicts."""
        state = GraphState(
            symbol="RELIANCE",
            resolved_symbol="RELIANCE",
            technical_indicators_summary={"rsi_14": 35.2},
            fundamentals_summary={"pe_ratio": 25.3},
            news_summary={"article_count": 15},
            market_context_summary={"nifty_50": 22500},
            collectors_complete=True,
        )
        assert state.collectors_complete is True
        assert state.technical_indicators_summary == {"rsi_14": 35.2}
        assert state.fundamentals_summary == {"pe_ratio": 25.3}
        assert state.news_summary == {"article_count": 15}
        assert state.market_context_summary == {"nifty_50": 22500}

    def test_graph_state_extra_forbidden(self):
        """GraphState should forbid extra fields."""
        try:
            GraphState(
                symbol="RELIANCE",
                resolved_symbol="RELIANCE",
                extra_field="should_fail",  # type: ignore
            )
            assert False, "Should have raised ValidationError"
        except Exception:
            pass  # Expected - extra fields forbidden