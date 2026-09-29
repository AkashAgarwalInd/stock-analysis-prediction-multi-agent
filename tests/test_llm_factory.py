import pytest
from unittest.mock import MagicMock, patch

from stock_analysis.llm import LLMFactory, get_llm_factory, get_llm, LLMModel
from stock_analysis.llm.models import LLMResponse
from stock_analysis.schemas import ForecastLLM
from stock_analysis.config import get_settings


class TestLLMFactory:
    def test_factory_singleton(self):
        factory1 = get_llm_factory()
        factory2 = get_llm_factory()
        assert factory1 is factory2

    def test_initialize_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from stock_analysis.config.settings import get_settings
        get_settings.cache_clear()

        factory = LLMFactory()
        factory.initialize()
        assert factory._initialized is True
        assert len(factory._models) == 0

    def test_initialize_with_api_key(self, mock_gemini, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        from stock_analysis.config.settings import get_settings
        get_settings.cache_clear()

        factory = LLMFactory()
        factory.initialize()

        assert factory._initialized is True
        assert len(factory._models) == 3
        assert LLMModel.PRIMARY in factory._models
        assert LLMModel.CRITIC in factory._models
        assert LLMModel.FALLBACK in factory._models

    def test_get_model_primary(self, mock_gemini, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        from stock_analysis.config.settings import get_settings
        get_settings.cache_clear()

        factory = LLMFactory()
        factory.initialize()

        model = factory.get_model(LLMModel.PRIMARY)
        assert model is mock_gemini["model"]

    def test_get_model_unknown_raises(self, mock_gemini, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        from stock_analysis.config.settings import get_settings
        get_settings.cache_clear()

        factory = LLMFactory()
        factory.initialize()

        with pytest.raises(ValueError, match="Unknown model role"):
            factory.get_model("invalid_role")

    def test_get_model_without_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from stock_analysis.config.settings import get_settings
        get_settings.cache_clear()

        factory = LLMFactory()
        factory.initialize()

        with pytest.raises(RuntimeError, match="Gemini API key not configured"):
            factory.get_model(LLMModel.PRIMARY)

    def test_generate_text(self, mock_gemini, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        from stock_analysis.config.settings import get_settings
        get_settings.cache_clear()

        mock_response = MagicMock()
        mock_response.text = "Test response"
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        mock_gemini["model"].generate_content.return_value = mock_response

        factory = LLMFactory()
        factory.initialize()

        response = factory.generate_text(LLMModel.PRIMARY, "Test prompt")

        assert isinstance(response, LLMResponse)
        assert response.content == "Test response"
        assert response.model == "gemini-3.5-flash-lite"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20

    def test_generate_structured(self, mock_gemini, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        from stock_analysis.config.settings import get_settings
        get_settings.cache_clear()

        mock_response = MagicMock()
        mock_response.text = '{"symbol": "RELIANCE", "forecast_date": "2024-01-15", "horizon_days": 5, "predictions": [], "reasoning": "test", "confidence": 0.8}'
        mock_gemini["model"].generate_content.return_value = mock_response

        factory = LLMFactory()
        factory.initialize()

        result = factory.generate_structured(
            LLMModel.PRIMARY,
            "Generate forecast",
            ForecastLLM,
        )

        assert isinstance(result, ForecastLLM)
        assert result.symbol == "RELIANCE"
        assert result.confidence == 0.8

    def test_generate_structured_empty_response_raises(self, mock_gemini, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        from stock_analysis.config.settings import get_settings
        get_settings.cache_clear()

        mock_response = MagicMock()
        mock_response.text = None
        mock_gemini["model"].generate_content.return_value = mock_response

        factory = LLMFactory()
        factory.initialize()

        with pytest.raises(RuntimeError, match="LLM returned empty response"):
            factory.generate_structured(LLMModel.PRIMARY, "Test", ForecastLLM)


class TestGetLLM:
    def test_get_llm_returns_model(self, mock_gemini, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        from stock_analysis.config.settings import get_settings
        get_settings.cache_clear()

        model = get_llm(LLMModel.PRIMARY)
        assert model is mock_gemini["model"]

    def test_get_llm_different_roles(self, mock_gemini, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        from stock_analysis.config.settings import get_settings
        get_settings.cache_clear()

        primary = get_llm(LLMModel.PRIMARY)
        critic = get_llm(LLMModel.CRITIC)
        fallback = get_llm(LLMModel.FALLBACK)

        assert primary is mock_gemini["model"]
        assert critic is mock_gemini["model"]
        assert fallback is mock_gemini["model"]