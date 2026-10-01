"""Phase 6 Predictor and Critic Tests."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import date

from stock_analysis.schemas.forecast_pipeline import (
    PredictorResult,
    CriticResult,
    CriticFinding,
    CriticCheckType,
    ForecastAdjustment,
    ForecastAdjustmentType,
    FinalForecast,
)
from stock_analysis.langgraph.workflow import (
    build_predictor_prompt,
    critic_node,
    final_forecast_node,
    revision_node,
)
from stock_analysis.schemas.graph_state import GraphState


def _raw_prediction(**overrides) -> dict:
    """Raw predictor output as a plain dict (bypasses PredictorResult validation).

    The critic must be able to judge output that the strict schema would reject.
    """
    base = {
        "prob_up": 0.55,
        "prob_flat": 0.30,
        "prob_down": 0.15,
        "expected_return_pct": 2.5,
        "p10_price": 98.0,
        "p50_price": 102.5,
        "p90_price": 107.0,
        "weekly_vol_pct": 15.0,
        "adjustment_applied": False,
        "adjustments": [],
        "reasoning": "Test",
        "evidence": [],
        "risks": [],
        "quant_baseline_ref": {},
    }
    return {**base, **overrides}


class TestPredictorPrompt:
    """Tests for predictor prompt building."""

    def test_build_predictor_prompt(self):
        """Prompt should include quant baseline and analyst reports."""
        quant = {
            "last_close": 100.0,
            "horizon_trading_days": 5,
            "prob_up": 0.55,
            "prob_flat": 0.30,
            "prob_down": 0.15,
            "expected_return_pct": 2.5,
            "p10_price": 98.0,
            "p50_price": 102.5,
            "p90_price": 107.0,
            "weekly_vol_pct": 15.0,
            "method": "ewma_monte_carlo",
        }
        reports = [
            {
                "analyst": "technical",
                "stance": "bullish",
                "confidence": 0.7,
                "key_points": ["Price above EMA"],
                "evidence": ["Close > EMA20"],
                "risks": ["RSI high"],
                "data_gaps": [],
            }
        ]
        
        prompt = build_predictor_prompt(quant, reports, [], [], [])
        
        assert "QUANT BASELINE" in prompt
        assert "0.55" in prompt  # prob_up
        assert "TECHNICAL ANALYST" in prompt
        assert "BULLISH" in prompt
        assert "MAX_PROB_SHIFT = 0.15" in prompt
        assert "prob_up + prob_flat + prob_down MUST equal 1.0" in prompt
        assert "P10 < P50 < P90" in prompt


class TestCriticNode:
    """Tests for critic validation node."""

    @pytest.fixture
    def valid_predictor_result(self):
        """Valid predictor result that should pass critic."""
        return PredictorResult(
            prob_up=0.55,
            prob_flat=0.30,
            prob_down=0.15,
            expected_return_pct=2.5,
            p10_price=98.0,
            p50_price=102.5,
            p90_price=107.0,
            weekly_vol_pct=15.0,
            adjustment_applied=False,
            adjustments=[],
            reasoning="No adjustment needed",
            evidence=["RSI=55"],
            risks=["Market risk"],
            quant_baseline_ref={
                "prob_up": 0.55,
                "prob_flat": 0.30,
                "prob_down": 0.15,
            },
        )

    def test_critic_passes_valid_result(self, valid_predictor_result):
        """Critic should pass valid predictor result."""
        state = GraphState(
            symbol="RELIANCE",
            resolved_symbol="RELIANCE",
            predictor_result=valid_predictor_result.model_dump(),
            quant_baseline={
                "prob_up": 0.55,
                "prob_flat": 0.30,
                "prob_down": 0.15,
            },
        )
        
        result = critic_node(state)
        
        assert "critic_result" in result
        critic_result = CriticResult(**result["critic_result"])
        assert critic_result.passed is True
        assert critic_result.revision_required is False

    def test_critic_fails_invalid_probabilities(self):
        """Critic should catch invalid probability values."""
        state = GraphState(
            symbol="RELIANCE",
            resolved_symbol="RELIANCE",
            predictor_result=_raw_prediction(prob_up=1.5, prob_down=-0.1),
            quant_baseline={},
        )
        
        result = critic_node(state)
        critic_result = CriticResult(**result["critic_result"])
        
        assert critic_result.passed is False
        assert critic_result.revision_required is True
        # Should have probability validity errors
        errors = [f for f in critic_result.findings if f.severity == "error"]
        assert len(errors) >= 2

    def test_critic_fails_probability_sum(self):
        """Critic should catch probability sum != 1."""
        state = GraphState(
            symbol="RELIANCE",
            resolved_symbol="RELIANCE",
            predictor_result=_raw_prediction(
                prob_up=0.60,
                prob_flat=0.30,
                prob_down=0.20,  # Sum = 1.1
                adjustment_applied=True,
                quant_baseline_ref={
                "prob_up": 0.55,
                "prob_flat": 0.30,
                "prob_down": 0.15,
            },
            ),
            quant_baseline={
                "prob_up": 0.55,
                "prob_flat": 0.30,
                "prob_down": 0.15,
            },
        )
        
        result = critic_node(state)
        critic_result = CriticResult(**result["critic_result"])
        
        assert critic_result.passed is False
        sum_errors = [f for f in critic_result.findings if f.check_type == CriticCheckType.PROBABILITY_SUM]
        assert len(sum_errors) == 1
        assert sum_errors[0].severity == "error"

    def test_critic_fails_price_ordering(self):
        """Critic should catch P10 >= P50 or P50 >= P90."""
        state = GraphState(
            symbol="RELIANCE",
            resolved_symbol="RELIANCE",
            predictor_result=_raw_prediction(p10_price=105.0),  # P10 > P50
            quant_baseline={},
        )
        
        result = critic_node(state)
        critic_result = CriticResult(**result["critic_result"])
        
        assert critic_result.passed is False
        ordering_errors = [f for f in critic_result.findings if f.check_type == CriticCheckType.PRICE_ORDERING]
        assert len(ordering_errors) == 1
        assert ordering_errors[0].severity == "error"

    def test_critic_fails_excessive_adjustment(self):
        """Critic should catch adjustments exceeding MAX_PROB_SHIFT."""
        state = GraphState(
            symbol="RELIANCE",
            resolved_symbol="RELIANCE",
            predictor_result=_raw_prediction(
                prob_up=0.75,  # Shift of 0.20 from baseline 0.55
                prob_flat=0.15,
                prob_down=0.10,
                expected_return_pct=5.0,
                adjustment_applied=True,
                quant_baseline_ref={
                "prob_up": 0.55,
                "prob_flat": 0.30,
                "prob_down": 0.15,
            },
            ),
            quant_baseline={
                "prob_up": 0.55,
                "prob_flat": 0.30,
                "prob_down": 0.15,
            },
        )
        
        result = critic_node(state)
        critic_result = CriticResult(**result["critic_result"])
        
        assert critic_result.passed is False
        excessive_errors = [f for f in critic_result.findings if f.check_type == CriticCheckType.EXCESSIVE_ADJUSTMENT]
        assert len(excessive_errors) >= 1
        assert excessive_errors[0].severity == "error"

    def test_critic_warnings_for_missing_evidence(self):
        """Critic should warn when adjustments lack evidence."""
        result_with_adj = PredictorResult(
            prob_up=0.60,
            prob_flat=0.25,
            prob_down=0.15,
            expected_return_pct=3.0,
            p10_price=98.5,
            p50_price=103.0,
            p90_price=107.5,
            weekly_vol_pct=15.5,
            adjustment_applied=True,
            adjustments=[
                ForecastAdjustment(
                    adjustment_type=ForecastAdjustmentType.PROB_UP,
                    previous_value=0.55,
                    new_value=0.60,
                    reason="Bullish signal",
                    evidence_refs=[],  # Empty evidence
                )
            ],
            reasoning="Adjusted for bullish signal",
            evidence=[],
            risks=[],
            quant_baseline_ref={
                "prob_up": 0.55,
                "prob_flat": 0.30,
                "prob_down": 0.15,
            },
        )
        
        state = GraphState(
            symbol="RELIANCE",
            resolved_symbol="RELIANCE",
            predictor_result=result_with_adj.model_dump(),
            quant_baseline={
                "prob_up": 0.55,
                "prob_flat": 0.30,
                "prob_down": 0.15,
            },
        )
        
        result = critic_node(state)
        critic_result = CriticResult(**result["critic_result"])
        
        # Should pass but have warning
        assert critic_result.passed is True
        evidence_warnings = [f for f in critic_result.findings if f.check_type == CriticCheckType.EVIDENCE_LINKAGE]
        assert len(evidence_warnings) == 1
        assert evidence_warnings[0].severity == "warning"

    def test_critic_warnings_overconfidence(self):
        """Critic should warn on extreme probabilities."""
        result = PredictorResult(
            prob_up=0.95,  # Very high
            prob_flat=0.03,
            prob_down=0.02,
            expected_return_pct=5.0,
            p10_price=98.0,
            p50_price=102.5,
            p90_price=107.0,
            weekly_vol_pct=15.0,
            adjustment_applied=False,
            adjustments=[],
            reasoning="Test",
            evidence=[],
            risks=[],
            quant_baseline_ref={},
        )
        
        state = GraphState(
            symbol="RELIANCE",
            resolved_symbol="RELIANCE",
            predictor_result=result.model_dump(),
            quant_baseline={},
        )
        
        result = critic_node(state)
        critic_result = CriticResult(**result["critic_result"])
        
        assert critic_result.passed is True
        overconf_warnings = [f for f in critic_result.findings if f.check_type == CriticCheckType.OVERCONFIDENCE]
        assert len(overconf_warnings) == 1
        assert overconf_warnings[0].severity == "warning"


class TestRevisionNode:
    """Tests for revision loop node."""

    def test_revision_passes_critic(self):
        """Should not increment when critic passes."""
        state = GraphState(
            symbol="RELIANCE",
            resolved_symbol="RELIANCE",
            critic_result=CriticResult(passed=True, findings=[]).model_dump(),
            forecast_revision_count=0,
        )
        
        result = revision_node(state)
        
        assert result.get("forecast_revision_count") == 0

    def test_revision_max_reached(self):
        """Should fallback to quant when max revisions reached."""
        quant = {
            "prob_up": 0.55,
            "prob_flat": 0.30,
            "prob_down": 0.15,
            "expected_return_pct": 2.5,
            "p10_price": 98.0,
            "p50_price": 102.5,
            "p90_price": 107.0,
            "weekly_vol_pct": 15.0,
            "horizon_trading_days": 5,
        }
        
        state = GraphState(
            symbol="RELIANCE",
            resolved_symbol="RELIANCE",
            quant_baseline=quant,
            critic_result=CriticResult(
                passed=False,
                findings=[CriticFinding(
                    check_type=CriticCheckType.PROBABILITY_SUM,
                    passed=False,
                    message="Sum error",
                    severity="error",
                )],
                revision_required=True,
                revision_guidance="Fix sum",
            ).model_dump(),
            forecast_revision_count=2,  # Already at max
        )
        
        result = revision_node(state)
        
        assert "predictor_result" in result
        assert result.get("forecast_revision_count") == 3
        # Should have fallback predictor result
        pred_result = result["predictor_result"]
        assert pred_result.get("adjustment_applied") is False
        assert "Max revisions" in pred_result.get("reasoning", "")

    def test_revision_increments_count(self):
        """Should increment revision count when critic fails and under max."""
        state = GraphState(
            symbol="RELIANCE",
            resolved_symbol="RELIANCE",
            critic_result=CriticResult(
                passed=False,
                findings=[CriticFinding(
                    check_type=CriticCheckType.PROBABILITY_SUM,
                    passed=False,
                    message="Sum error",
                    severity="error",
                )],
                revision_required=True,
                revision_guidance="Fix sum",
            ).model_dump(),
            forecast_revision_count=0,
        )
        
        result = revision_node(state)
        
        assert result.get("forecast_revision_count") == 1
        # Should not have fallback predictor result yet
        assert "predictor_result" not in result


class TestFinalForecastNode:
    """Tests for final forecast consolidation."""

    def test_final_forecast_from_predictor(self):
        """Should create final forecast from predictor result."""
        pred_result = PredictorResult(
            prob_up=0.60,
            prob_flat=0.25,
            prob_down=0.15,
            expected_return_pct=3.0,
            p10_price=98.5,
            p50_price=103.0,
            p90_price=107.5,
            weekly_vol_pct=15.5,
            adjustment_applied=True,
            adjustments=[
                ForecastAdjustment(
                    adjustment_type=ForecastAdjustmentType.PROB_UP,
                    previous_value=0.55,
                    new_value=0.60,
                    reason="Bullish momentum",
                    evidence_refs=["RSI>70"],
                )
            ],
            reasoning="Adjusted for bullish technicals",
            evidence=["RSI>70", "MACD bullish"],
            risks=["Overbought risk"],
            quant_baseline_ref={
                "prob_up": 0.55,
                "prob_flat": 0.30,
                "prob_down": 0.15,
            },
        )
        
        quant = {
            "prob_up": 0.55,
            "prob_flat": 0.30,
            "prob_down": 0.15,
            "expected_return_pct": 2.5,
            "p10_price": 98.0,
            "p50_price": 102.5,
            "p90_price": 107.0,
            "weekly_vol_pct": 15.0,
            "horizon_trading_days": 5,
        }
        
        state = GraphState(
            symbol="RELIANCE",
            resolved_symbol="RELIANCE",
            predictor_result=pred_result.model_dump(),
            quant_baseline=quant,
            critic_result=CriticResult(passed=True, findings=[]).model_dump(),
            forecast_revision_count=1,
            technical_report={"evidence": ["RSI>70"]},
            fundamental_report={"evidence": ["P/E=25"]},
        )
        
        result = final_forecast_node(state)
        
        assert "final_forecast" in result
        final = FinalForecast(**result["final_forecast"])
        assert final.symbol == "RELIANCE"
        assert final.prob_up == 0.60
        assert final.adjustment_applied is True
        assert len(final.adjustments) == 1
        assert final.critic_passed is True
        assert final.revision_count == 1
        assert final.fallback_to_quant is False
        assert "RSI>70" in final.evidence

    def test_final_forecast_fallback_to_quant(self):
        """Should fallback to quant when predictor fails."""
        state = GraphState(
            symbol="RELIANCE",
            resolved_symbol="RELIANCE",
            predictor_result={"error": "LLM failed"},
            quant_baseline={
                "prob_up": 0.55,
                "prob_flat": 0.30,
                "prob_down": 0.15,
                "expected_return_pct": 2.5,
                "p10_price": 98.0,
                "p50_price": 102.5,
                "p90_price": 107.0,
                "weekly_vol_pct": 15.0,
                "horizon_trading_days": 5,
            },
            critic_result=CriticResult(passed=False, findings=[]).model_dump(),
            forecast_revision_count=0,
        )
        
        result = final_forecast_node(state)
        
        assert "final_forecast" in result
        final = FinalForecast(**result["final_forecast"])
        assert final.fallback_to_quant is True
        assert final.prob_up == 0.55
        assert final.adjustment_applied is False