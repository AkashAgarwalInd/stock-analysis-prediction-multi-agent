import os
import pytest
from unittest.mock import MagicMock, patch

from stock_analysis.llm import get_llm_factory, LLMModel
from stock_analysis.schemas import ForecastLLM


class TestGeminiSmoke:
    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") == "test-key-12345",
        reason="Requires valid GEMINI_API_KEY",
    )
    def test_gemini_structured_output_live(self):
        factory = get_llm_factory()
        factory.initialize()

        result = factory.generate_structured(
            LLMModel.PRIMARY,
            "Generate a simple 1-day forecast for RELIANCE stock. Return only the JSON.",
            ForecastLLM,
            temperature=0.0,
        )

        assert isinstance(result, ForecastLLM)
        assert result.symbol == "RELIANCE"
        assert 0.0 <= result.confidence <= 1.0

    def test_gemini_structured_output_mocked(self, mock_gemini):
        mock_response = MagicMock()
        mock_response.text = '''{
            "symbol": "RELIANCE",
            "forecast_date": "2024-01-15",
            "horizon_days": 5,
            "predictions": [
                {"date": "2024-01-16", "close": "1000.00"},
                {"date": "2024-01-17", "close": "1010.00"}
            ],
            "reasoning": "Test reasoning",
            "confidence": 0.75
        }'''
        mock_gemini["model"].generate_content.return_value = mock_response

        factory = get_llm_factory()
        factory.initialize()

        result = factory.generate_structured(
            LLMModel.PRIMARY,
            "Test prompt",
            ForecastLLM,
        )

        assert isinstance(result, ForecastLLM)
        assert result.symbol == "RELIANCE"
        assert result.confidence == 0.75
        assert len(result.predictions) == 2
        assert result.predictions[0].close == 1000.00