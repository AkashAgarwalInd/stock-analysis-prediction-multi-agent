import json
from typing import Any, TypeVar

from pydantic import BaseModel

from stock_analysis.config import get_settings
from stock_analysis.llm.models import LLMModel, LLMResponse
from stock_analysis.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

__all__ = ["LLMFactory", "LLMModel", "LLMResponse", "get_llm", "get_llm_factory"]


class LLMFactory:
    """Creates Gemini models per role and runs structured / text generation."""

    def __init__(self) -> None:
        self._models: dict[LLMModel, Any] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Build one Gemini model per role. A no-op until an API key is configured."""
        if self._initialized:
            return

        settings = get_settings()
        if not settings.gemini_api_key:
            logger.warning("gemini_api_key_not_set")
            return

        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        self._models = {
            LLMModel.PRIMARY: genai.GenerativeModel(settings.gemini_model_primary),
            LLMModel.CRITIC: genai.GenerativeModel(settings.gemini_model_critic),
            LLMModel.FALLBACK: genai.GenerativeModel(settings.gemini_model_fallback),
        }
        self._initialized = True
        logger.info("llm_factory_initialized", roles=[r.value for r in self._models])

    def get_model(self, role: LLMModel) -> Any:
        """Return the Gemini model for ``role``; raises if no API key is configured."""
        self.initialize()
        if not self._models:
            raise RuntimeError("Gemini API key not configured")
        return self._models[role]

    @staticmethod
    def _generation_config(
        temperature: float | None, max_tokens: int | None, json_mode: bool
    ) -> Any:
        import google.generativeai as genai

        settings = get_settings()
        kwargs: dict[str, Any] = {
            "temperature": settings.gemini_temperature if temperature is None else temperature,
            "max_output_tokens": settings.gemini_max_tokens if max_tokens is None else max_tokens,
        }
        if json_mode:
            kwargs["response_mime_type"] = "application/json"
        return genai.GenerationConfig(**kwargs)

    def _generate(
        self,
        role: LLMModel,
        prompt: str,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
    ) -> tuple[Any, Any]:
        """Call the model for ``role``; on an API error retry once on the FALLBACK model."""
        config = self._generation_config(temperature, max_tokens, json_mode)
        model = self.get_model(role)
        try:
            return model, model.generate_content(prompt, generation_config=config)
        except Exception as exc:  # SDK raises many unrelated error types
            if role == LLMModel.FALLBACK:
                raise
            logger.warning("llm_call_failed_using_fallback", role=role.value, error=str(exc))
            fallback = self.get_model(LLMModel.FALLBACK)
            return fallback, fallback.generate_content(prompt, generation_config=config)

    def generate_structured(
        self,
        role: LLMModel,
        prompt: str,
        response_schema: type[T],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Generate JSON and validate it against ``response_schema``.

        The schema is appended to the prompt instead of passed as Gemini's
        ``response_schema`` because the SDK rejects free-form ``dict`` fields.
        Schema violations surface as ``pydantic.ValidationError``.
        """
        schema = json.dumps(response_schema.model_json_schema())
        full_prompt = f"{prompt}\n\nRespond with JSON only, matching this JSON schema:\n{schema}"
        logger.debug("llm_generate_structured", role=role.value)

        _, response = self._generate(role, full_prompt, temperature, max_tokens, json_mode=True)
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("LLM returned empty response")
        return response_schema.model_validate_json(text)

    def generate_text(
        self,
        role: LLMModel,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate plain text. Returns an empty response when no API key is configured."""
        try:
            self.get_model(role)
        except RuntimeError:
            return LLMResponse(content="", model="", usage={})

        logger.debug("llm_generate_text", role=role.value)
        model, response = self._generate(role, prompt, temperature, max_tokens, json_mode=False)

        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            content=response.text or "",
            model=getattr(model, "model_name", "unknown"),
            usage={
                "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                "completion_tokens": getattr(usage, "candidates_token_count", 0),
            }
            if usage
            else None,
        )


_factory_instance: LLMFactory | None = None


def get_llm_factory() -> LLMFactory:
    """Get or create the LLM factory singleton."""
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = LLMFactory()
    return _factory_instance


def get_llm(role: LLMModel) -> Any:
    """Get an LLM model by role."""
    return get_llm_factory().get_model(role)


class DisabledLLM:
    """Stands in for the LLM in a quant-only run (every analyst disabled); any call is a bug."""

    def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("LLM calls are disabled for this run")
