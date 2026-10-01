"""Phase 6 Forecast Pipeline Schema Tests."""

import pytest
from datetime import date

from stock_analysis.schemas.forecast_pipeline import (
    ForecastAdjustment,
    ForecastAdjustmentType,
    PredictorResult,
    CriticResult,
    CriticFinding,
    CriticCheckType,
    FinalForecast,
)


class TestForecastAdjustment:
    """Tests for ForecastAdjustment schema."""

    def test_valid_adjustment(self):
        """Valid adjustment should pass validation."""
        adj = ForecastAdjustment(
            adjustment_type=ForecastAdjustmentType.PROB_UP,
            previous_value=0.55,
            new_value=0.65,
            reason="Strong technical momentum",
            evidence_refs=["RSI > 70", "MACD bullish crossover"],
        )
        assert adj.adjustment_type == ForecastAdjustmentType.PROB_UP
        assert adj.previous_value == 0.55
        assert adj.new_value == 0.65
        assert adj.reason == "Strong technical momentum"
        assert len(adj.evidence_refs) == 2

    def test_adjustment_magnitude_limit_prob(self):
        """Probability adjustments can shift up to 100%."""
        # This should pass - 50% change is allowed for probabilities
        adj = ForecastAdjustment(
            adjustment_type=ForecastAdjustmentType.PROB_UP,
            previous_value=0.40,
            new_value=0.70,  # 75% increase - allowed for probabilities
            reason="Test",
            evidence_refs=[],
        )
        assert adj.new_value == 0.70

    def test_adjustment_magnitude_limit_non_prob(self):
        """Non-probability adjustments limited to 50% change."""
        # This should fail - 60% change for expected return
        with pytest.raises(ValueError, match="exceeds maximum allowed change"):
            ForecastAdjustment(
                adjustment_type=ForecastAdjustmentType.EXPECTED_RETURN,
                previous_value=2.0,
                new_value=5.0,  # 150% increase - not allowed
                reason="Test",
                evidence_refs=[],
            )

    def test_adjustment_without_evidence_refs(self):
        """Adjustments can have empty evidence_refs (warning, not error)."""
        adj = ForecastAdjustment(
            adjustment_type=ForecastAdjustmentType.PROB_UP,
            previous_value=0.50,
            new_value=0.55,
            reason="Test",
            evidence_refs=[],
        )
        assert adj.evidence_refs == []


class TestPredictorResult:
    """Tests for PredictorResult schema."""

    @pytest.fixture
    def valid_quant_baseline(self):
        return {
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

    def test_valid_predictor_result(self, valid_quant_baseline):
        """Valid predictor result should pass."""
        result = PredictorResult(
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
                    reason="Bullish sentiment",
                    evidence_refs=["Positive news flow"],
                )
            ],
            reasoning="Adjusted for bullish sentiment",
            evidence=["Positive news flow", "RSI > 60"],
            risks=["Market volatility"],
            quant_baseline_ref=valid_quant_baseline,
        )
        assert result.prob_up == 0.60
        assert result.adjustment_applied is True
        assert len(result.adjustments) == 1

    def test_predictor_result_no_adjustment(self, valid_quant_baseline):
        """Predictor result with no adjustment should pass."""
        result = PredictorResult(
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
            reasoning="No adjustment warranted",
            evidence=[],
            risks=[],
            quant_baseline_ref=valid_quant_baseline,
        )
        assert result.adjustment_applied is False
        assert result.adjustments == []

    def test_probability_sum_validation(self, valid_quant_baseline):
        """Probabilities must sum to 1.0."""
        with pytest.raises(ValueError, match="Probabilities must sum to 1.0"):
            PredictorResult(
                prob_up=0.60,
                prob_flat=0.30,
                prob_down=0.20,  # Sum = 1.1
                expected_return_pct=3.0,
                p10_price=98.5,
                p50_price=103.0,
                p90_price=107.5,
                weekly_vol_pct=15.5,
                adjustment_applied=True,
                adjustments=[],
                reasoning="Test",
                evidence=[],
                risks=[],
                quant_baseline_ref=valid_quant_baseline,
            )

    def test_price_ordering_validation(self, valid_quant_baseline):
        """P10 < P50 < P90 must hold."""
        with pytest.raises(ValueError, match="Price ordering violated"):
            PredictorResult(
                prob_up=0.55,
                prob_flat=0.30,
                prob_down=0.15,
                expected_return_pct=2.5,
                p10_price=102.0,  # P10 > P50
                p50_price=101.0,
                p90_price=107.0,
                weekly_vol_pct=15.0,
                adjustment_applied=False,
                adjustments=[],
                reasoning="Test",
                evidence=[],
                risks=[],
                quant_baseline_ref=valid_quant_baseline,
            )

    def test_max_prob_shift_validation(self, valid_quant_baseline):
        """Probability shifts must not exceed MAX_PROB_SHIFT=0.15."""
        with pytest.raises(ValueError, match="exceeds MAX_PROB_SHIFT"):
            PredictorResult(
                prob_up=0.75,  # Shift of 0.20 from 0.55 - exceeds 0.15
                prob_flat=0.15,
                prob_down=0.10,
                expected_return_pct=2.5,
                p10_price=98.0,
                p50_price=102.5,
                p90_price=107.0,
                weekly_vol_pct=15.0,
                adjustment_applied=True,
                adjustments=[],
                reasoning="Test",
                evidence=[],
                risks=[],
                quant_baseline_ref=valid_quant_baseline,
            )

    def test_probability_bounds(self, valid_quant_baseline):
        """Probabilities must be in [0, 1]."""
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            PredictorResult(
                prob_up=-0.1,
                prob_flat=0.6,
                prob_down=0.5,
                expected_return_pct=2.5,
                p10_price=98.0,
                p50_price=102.5,
                p90_price=107.0,
                weekly_vol_pct=15.0,
                adjustment_applied=False,
                adjustments=[],
                reasoning="Test",
                evidence=[],
                risks=[],
                quant_baseline_ref=valid_quant_baseline,
            )


class TestCriticResult:
    """Tests for CriticResult schema."""

    def test_critic_result_passed(self):
        """Critic result with all passed findings."""
        result = CriticResult(
            passed=True,
            findings=[
                CriticFinding(
                    check_type=CriticCheckType.PROBABILITY_VALIDITY,
                    passed=True,
                    message="All probabilities valid",
                    severity="warning",
                ),
                CriticFinding(
                    check_type=CriticCheckType.PRICE_ORDERING,
                    passed=True,
                    message="Price ordering correct",
                    severity="warning",
                ),
            ],
            revision_required=False,
            revision_guidance=None,
        )
        assert result.passed is True
        assert result.revision_required is False

    def test_critic_result_failed(self):
        """Critic result with failed findings."""
        result = CriticResult(
            passed=False,
            findings=[
                CriticFinding(
                    check_type=CriticCheckType.PROBABILITY_SUM,
                    passed=False,
                    message="Probabilities sum to 1.1",
                    severity="error",
                ),
            ],
            revision_required=True,
            revision_guidance="Fix probability sum",
        )
        assert result.passed is False
        assert result.revision_required is True

    def test_critic_auto_sets_revision_flag(self):
        """CriticResult should auto-set revision_required based on findings."""
        result = CriticResult(
            passed=False,
            findings=[
                CriticFinding(
                    check_type=CriticCheckType.PROBABILITY_VALIDITY,
                    passed=False,
                    message="Error found",
                    severity="error",
                ),
                CriticFinding(
                    check_type=CriticCheckType.MISSING_RISKS,
                    passed=False,
                    message="Warning only",
                    severity="warning",
                ),
            ],
        )
        # revision_required should be True because there's an error finding
        assert result.revision_required is True


class TestFinalForecast:
    """Tests for FinalForecast schema."""

    def test_valid_final_forecast(self):
        """Valid final forecast should pass."""
        final = FinalForecast(
            symbol="RELIANCE",
            forecast_date=str(date.today()),
            horizon_trading_days=5,
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
            quant_baseline={"prob_up": 0.55, "method": "ewma_monte_carlo"},
            predictor_reasoning="No adjustment needed",
            critic_passed=True,
            revision_count=0,
            fallback_to_quant=True,
            evidence=["RSI=55", "MACD positive"],
            risks=["Market risk"],
        )
        assert final.symbol == "RELIANCE"
        assert final.fallback_to_quant is True

    def test_final_forecast_prob_sum_validation(self):
        """Final forecast must have probabilities summing to 1."""
        with pytest.raises(ValueError, match="Final probabilities sum to"):
            FinalForecast(
                symbol="RELIANCE",
                forecast_date=str(date.today()),
                horizon_trading_days=5,
                prob_up=0.55,
                prob_flat=0.30,
                prob_down=0.20,  # Sum = 1.05
                expected_return_pct=2.5,
                p10_price=98.0,
                p50_price=102.5,
                p90_price=107.0,
                weekly_vol_pct=15.0,
                adjustment_applied=False,
                adjustments=[],
                quant_baseline={},
                predictor_reasoning="Test",
                critic_passed=True,
                revision_count=0,
                fallback_to_quant=True,
                evidence=[],
                risks=[],
            )

    def test_final_forecast_price_ordering(self):
        """Final forecast must have P10 < P50 < P90."""
        with pytest.raises(ValueError, match="price ordering violated"):
            FinalForecast(
                symbol="RELIANCE",
                forecast_date=str(date.today()),
                horizon_trading_days=5,
                prob_up=0.55,
                prob_flat=0.30,
                prob_down=0.15,
                expected_return_pct=2.5,
                p10_price=105.0,  # P10 > P50
                p50_price=102.5,
                p90_price=107.0,
                weekly_vol_pct=15.0,
                adjustment_applied=False,
                adjustments=[],
                quant_baseline={},
                predictor_reasoning="Test",
                critic_passed=True,
                revision_count=0,
                fallback_to_quant=True,
                evidence=[],
                risks=[],
            )