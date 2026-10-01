"""Unit tests for Phase 5 DecisionEngine and RulesDecisionEngine."""

from stock_analysis.schemas.analyst_reports import (
    AnalystReport,
    AnalystType,
    AnalystStance,
    DecisionResult,
    DecisionType,
    MarketRegime,
    RiskCategory,
    RulesDecisionEngine,
)


class TestRulesDecisionEngine:
    """Tests for RulesDecisionEngine deterministic decisions."""

    def test_default_engine_instance(self):
        """RulesDecisionEngine should be instantiable."""
        engine = RulesDecisionEngine()
        assert engine.name == "rules"

    def test_decide_market_regime_bullish(self):
        """Market regime should classify as trending_up with bullish consensus."""
        engine = RulesDecisionEngine()

        reports = [
            {
                "analyst": "technical",
                "stance": "bullish",
                "confidence": 0.8,
                "key_points": ["RSI bullish"],
                "evidence": ["RSI_14=68"],
                "risks": ["overbought"],
                "data_gaps": [],
            },
            {
                "analyst": "fundamental",
                "stance": "bullish",
                "confidence": 0.7,
                "key_points": ["P/E reasonable"],
                "evidence": ["P/E=25"],
                "risks": ["sector slowdown"],
                "data_gaps": [],
            },
        ]

        result = engine.decide(
            DecisionType.MARKET_REGIME,
            {"analyst_reports": reports},
        )

        assert result.decision_type == DecisionType.MARKET_REGIME
        assert result.result == MarketRegime.TRENDING_UP.value
        assert result.model == "rules"

    def test_decide_market_regime_bearish(self):
        """Market regime should classify as trending_down with bearish consensus."""
        engine = RulesDecisionEngine()

        reports = [
            {
                "analyst": "technical",
                "stance": "bearish",
                "confidence": 0.85,
                "key_points": ["MACD dead cross"],
                "evidence": ["MACD histogram negative"],
                "risks": ["trend reversal"],
                "data_gaps": [],
            },
        ]

        result = engine.decide(
            DecisionType.MARKET_REGIME,
            {"analyst_reports": reports},
        )

        assert result.result == MarketRegime.TRENDING_DOWN.value

    def test_decide_market_regime_high_volatility(self):
        """Market regime should classify as high_volatility with extreme stances."""
        engine = RulesDecisionEngine()

        reports = [
            {
                "analyst": "technical",
                "stance": "strong_bullish",
                "confidence": 0.95,
                "key_points": [],
                "evidence": [],
                "risks": [],
                "data_gaps": [],
            },
            {
                "analyst": "sentiment",
                "stance": "strong_bearish",
                "confidence": 0.9,
                "key_points": [],
                "evidence": [],
                "risks": [],
                "data_gaps": [],
            },
        ]

        result = engine.decide(
            DecisionType.MARKET_REGIME,
            {"analyst_reports": reports},
        )

        assert result.result == MarketRegime.HIGH_VOLATILITY.value

    def test_decide_risk_category_no_gaps(self):
        """Risk category should be low when no critical data gaps."""
        engine = RulesDecisionEngine()

        reports = [
            {
                "analyst": "technical",
                "stance": "bullish",
                "confidence": 0.7,
                "key_points": [],
                "evidence": [],
                "risks": [],
                "data_gaps": [],
            },
        ]

        result = engine.decide(
            DecisionType.RISK_CATEGORY,
            {"analyst_reports": reports},
        )

        assert result.result == RiskCategory.LOW.value

    def test_decide_risk_category_with_critical_gaps(self):
        """Risk category should be high when critical data gaps exist."""
        engine = RulesDecisionEngine()

        reports = [
            {
                "analyst": "technical",
                "stance": "bullish",
                "confidence": 0.7,
                "key_points": [],
                "evidence": [],
                "risks": [],
                "data_gaps": ["critical price data missing"],
            },
        ]

        result = engine.decide(
            DecisionType.RISK_CATEGORY,
            {"analyst_reports": reports},
        )

        assert result.result == RiskCategory.HIGH.value

    def test_decide_analyst_stance_neutral(self):
        """Analyst stance should be neutral when no reports."""
        engine = RulesDecisionEngine()

        result = engine.decide(
            DecisionType.ANALYST_STANCE,
            {"analyst_reports": []},
        )

        assert result.result == AnalystStance.NEUTRAL.value

    def test_decide_analyst_stance_primary(self):
        """Analyst stance should be primary (most confident) stance."""
        engine = RulesDecisionEngine()

        reports = [
            {
                "analyst": "technical",
                "stance": "bullish",
                "confidence": 0.8,
                "key_points": [],
                "evidence": [],
                "risks": [],
                "data_gaps": [],
            },
            {
                "analyst": "fundamental",
                "stance": "neutral",
                "confidence": 0.5,
                "key_points": [],
                "evidence": [],
                "risks": [],
                "data_gaps": [],
            },
        ]

        result = engine.decide(
            DecisionType.ANALYST_STANCE,
            {"analyst_reports": reports},
        )

        assert result.result == AnalystStance.BULLISH.value
        assert result.confidence == 0.8

    def test_decide_unsupported_type(self):
        """Unsupported decision type should degrade gracefully."""
        engine = RulesDecisionEngine()

        result = engine.decide(
            "unsupported_type",
            {"analyst_reports": []},
        )

        assert result.result == MarketRegime.SIDWAYS.value


class TestDecisionResult:
    """Tests for DecisionResult schema."""

    def test_decision_result_basic(self):
        """DecisionResult should validate basic fields."""
        r = DecisionResult(
            decision_type=DecisionType.MARKET_REGIME,
            result="trending_up",
            confidence=0.75,
            rationale="Bullish consensus",
        )
        assert r.decision_type == DecisionType.MARKET_REGIME
        assert r.result == "trending_up"
        assert r.confidence == 0.75
        assert r.rationale == "Bullish consensus"
        assert r.model == "rules"
        assert r.model_version == "0.1.0"

    def test_decision_result_custom_model(self):
        """DecisionResult should accept custom model/version."""
        r = DecisionResult(
            decision_type=DecisionType.RISK_CATEGORY,
            result="high",
            confidence=0.9,
            rationale="Critical gaps",
            model="jev",
            model_version="2.0.0",
        )
        assert r.model == "jev"
        assert r.model_version == "2.0.0"

    def test_decision_result_confidence_bounds(self):
        """Confidence must be 0.0-1.0."""
        # Valid
        r = DecisionResult(
            decision_type=DecisionType.ANALYST_STANCE,
            result="bullish",
            confidence=0.5,
            rationale="test",
        )
        assert 0.0 <= r.confidence <= 1.0

        # Invalid - too high
        try:
            DecisionResult(
                decision_type=DecisionType.MARKET_REGIME,
                result="test",
                confidence=1.5,  # type: ignore
                rationale="test",
            )
            assert False, "Should raise"
        except Exception:
            pass

        # Invalid - too low
        try:
            DecisionResult(
                decision_type=DecisionType.MARKET_REGIME,
                result="test",
                confidence=-0.1,  # type: ignore
                rationale="test",
            )
            assert False, "Should raise"
        except Exception:
            pass

def test_gate_counts_each_analyst_not_each_distinct_stance():
    """Four analysts sharing one stance are four confident analysts, not one."""
    from stock_analysis.schemas.analyst_reports import (
        AdjustmentGateDecision,
        DecisionType,
        RulesDecisionEngine,
    )

    reports = [
        {
            "analyst": analyst,
            "stance": "bullish",
            "confidence": 0.7,
            "key_points": ["Trend is constructive"],
            "evidence": ["Close above 20-day average"],
            "risks": [],
            "data_gaps": [],
        }
        for analyst in ("technical", "fundamental", "sentiment", "context")
    ]
    result = RulesDecisionEngine().decide(
        DecisionType.FORECAST_ADJUSTMENT_GATE,
        {"analyst_reports": reports, "quant_baseline": {"prob_up": 0.5}},
    )
    assert result.result == AdjustmentGateDecision.ALLOW_ADJUSTMENT.value
