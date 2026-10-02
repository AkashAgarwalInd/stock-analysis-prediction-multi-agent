import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from stock_analysis.llm import LLMFactory, LLMModel, get_llm_factory
from stock_analysis.llm import factory as factory_module
from stock_analysis.llm.models import LLMResponse


class Answer(BaseModel):
    verdict: str
    score: float


def _model(text: str | None = None, error: Exception | None = None) -> MagicMock:
    model = MagicMock()
    model.model_name = "fake-model"
    if error is not None:
        model.generate_content.side_effect = error
    else:
        model.generate_content.return_value = MagicMock(text=text)
    return model


def _factory(**models: MagicMock) -> LLMFactory:
    """Factory with injected models, skipping real SDK initialisation."""
    factory = LLMFactory()
    factory._models = {LLMModel(role): model for role, model in models.items()}
    factory._initialized = True
    return factory


class TestLLMFactory:
    def test_singleton(self):
        assert get_llm_factory() is get_llm_factory()

    def test_models_are_single_definition(self):
        assert factory_module.LLMModel is LLMModel
        assert factory_module.LLMResponse is LLMResponse

    def test_get_model_without_api_key_raises(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "")
        factory = LLMFactory()
        with pytest.raises(RuntimeError, match="Gemini API key not configured"):
            factory.get_model(LLMModel.PRIMARY)

    def test_key_added_later_is_picked_up(self, monkeypatch):
        from stock_analysis.config.settings import get_settings

        monkeypatch.setenv("GEMINI_API_KEY", "")
        get_settings.cache_clear()
        factory = LLMFactory()
        factory.initialize()
        assert not factory._initialized

        monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
        get_settings.cache_clear()
        factory.initialize()
        assert factory._initialized
        assert set(factory._models) == set(LLMModel)

    def test_generate_structured_sends_json_config_and_schema(self):
        primary = _model(json.dumps({"verdict": "ok", "score": 0.5}))
        result = _factory(primary=primary).generate_structured(
            LLMModel.PRIMARY, "Rate it.", Answer, temperature=0.3, max_tokens=123
        )

        assert result == Answer(verdict="ok", score=0.5)
        (prompt,), kwargs = primary.generate_content.call_args
        assert "Rate it." in prompt
        assert '"verdict"' in prompt  # JSON schema appended
        config = kwargs["generation_config"]
        assert config.response_mime_type == "application/json"
        assert config.temperature == 0.3
        assert config.max_output_tokens == 123

    def test_generate_structured_defaults_come_from_settings(self):
        from stock_analysis.config.settings import get_settings

        primary = _model(json.dumps({"verdict": "ok", "score": 1}))
        _factory(primary=primary).generate_structured(LLMModel.PRIMARY, "x", Answer)

        config = primary.generate_content.call_args.kwargs["generation_config"]
        assert config.temperature == get_settings().gemini_temperature
        assert config.max_output_tokens == get_settings().gemini_max_tokens

    def test_generate_structured_empty_response_raises(self):
        with pytest.raises(RuntimeError, match="empty response"):
            _factory(primary=_model(None)).generate_structured(LLMModel.PRIMARY, "x", Answer)

    def test_schema_violation_raises_validation_error_without_fallback(self):
        primary = _model('{"verdict": "ok"}')
        fallback = _model(json.dumps({"verdict": "ok", "score": 1}))
        with pytest.raises(ValidationError):
            _factory(primary=primary, fallback=fallback).generate_structured(
                LLMModel.PRIMARY, "x", Answer
            )
        fallback.generate_content.assert_not_called()

    def test_api_error_retries_on_fallback_model(self):
        primary = _model(error=ConnectionError("503"))
        fallback = _model(json.dumps({"verdict": "backup", "score": 0.1}))
        result = _factory(primary=primary, fallback=fallback).generate_structured(
            LLMModel.PRIMARY, "x", Answer
        )
        assert result.verdict == "backup"

    def test_fallback_failure_is_raised(self):
        broken = _model(error=ConnectionError("503"))
        with pytest.raises(ConnectionError):
            _factory(primary=broken, fallback=_model(error=ConnectionError("503"))).generate_text(
                LLMModel.PRIMARY, "x"
            )

    def test_generate_text_returns_content_and_usage(self):
        primary = _model("hello")
        primary.generate_content.return_value.usage_metadata = MagicMock(
            prompt_token_count=3, candidates_token_count=2
        )
        response = _factory(primary=primary).generate_text(LLMModel.PRIMARY, "hi")

        assert response.content == "hello"
        assert response.model == "fake-model"
        assert response.usage == {"prompt_tokens": 3, "completion_tokens": 2}
        config = primary.generate_content.call_args.kwargs["generation_config"]
        assert not getattr(config, "response_mime_type", None)  # plain text, not JSON

    def test_generate_text_without_api_key_is_empty(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "")
        response = LLMFactory().generate_text(LLMModel.PRIMARY, "hi")
        assert response.content == ""
