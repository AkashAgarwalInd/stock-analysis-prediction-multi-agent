import logging
from enum import Enum
from typing import Any, TypeVar, Generic

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Any)


class LLMModel(str, Enum):
    PRIMARY = "primary"
    CRITIC = "critic"
    FALLBACK = "fallback"


class LLMResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    content: str
    model: str
    usage: dict[str, Any] | None = None
    metadata: dict[str, Any] = {}


class LLMFactory:
    _instance: "LLMFactory | None" = None

    def __init__(self):
        self._settings = None
        self._models: dict[LLMModel, Any] = {}
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return

        try:
            from stock_analysis.config import get_settings
            self._settings = get_settings()
        except Exception:
            self._settings = None

        if not self._settings or not self._settings.gemini_api_key:
            logger.warning("gemini_api_key_not_set", message="Gemini API key not configured")
            self._initialized = True
            return

        import google.generativeai as genai

        genai.configure(api_key=self._settings.gemini_api_key)

        self._models[LLMModel.PRIMARY] = genai.GenerativeModel(
            self._settings.gemini_model_primary
        )
        self._models[LLMModel.CRITIC] = genai.GenerativeModel(
            self._settings.gemini_model_critic
        )
        self._models[LLMModel.FALLBACK] = genai.GenerativeModel(
            self._settings.gemini_model_fallback
        )

        self._initialized = True
        logger.info("llm_factory_initialized", models=list(self._models.keys()))

    def get_model(self, role: LLMModel) -> Any:
        if not self._initialized:
            self.initialize()

        if role not in self._models:
            if not self._settings or not self._settings.gemini_api_key:
                raise RuntimeError("Gemini API key not configured")
            raise ValueError(f"Unknown model role: {role}")

        return self._models[role]

    def generate_structured(
        self,
        role: LLMModel,
        prompt: str,
        response_schema: type[T],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        try:
            model = self.get_model(role)
        except RuntimeError:
            raise

        try:
            from google.generativeai.types import GenerateContentConfig
        except ImportError:
            GenerateContentConfig = None

        config_kwargs: dict[str, Any] = {}
        if GenerateContentConfig is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        if temperature is not None:
            config_kwargs["temperature"] = temperature
        else:
            try:
                from stock_analysis.config import get_settings
                config_kwargs["temperature"] = get_settings().gemini_temperature
            except Exception:
                config_kwargs["temperature"] = 0.1

        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        else:
            try:
                from stock_analysis.config import get_settings
                config_kwargs["max_output_tokens"] = get_settings().gemini_max_tokens
            except Exception:
                config_kwargs["max_output_tokens"] = 8192

        logger.debug("llm_generate_structured", role=role.value)

        if GenerateContentConfig is not None:
            response = model.generate_content(prompt, generation_config=GenerateContentConfig(**config_kwargs))
        else:
            response = model.generate_content(prompt)

        if getattr(response, "text", None) is None:
            raise RuntimeError("LLM returned empty response")

        return response_schema.model_validate_json(response.text)

    def generate_text(
        self,
        role: LLMModel,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        try:
            model = self.get_model(role)
        except RuntimeError:
            raise LLMResponse(content="", model="", usage={})

        try:
            from google.generativeai.types import GenerateContentConfig
        except ImportError:
            GenerateContentConfig = None

        config_kwargs: dict[str, Any] = {}
        if GenerateContentConfig is not None:
            config_kwargs["response_mime_type"] = "application/json"

        if temperature is not None:
            config_kwargs["temperature"] = temperature
        else:
            try:
                from stock_analysis.config import get_settings
                config_kwargs["temperature"] = get_settings().gemini_temperature
            except Exception:
                config_kwargs["temperature"] = 0.1

        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        else:
            try:
                from stock_analysis.config import get_settings
                config_kwargs["max_output_tokens"] = get_settings().gemini_max_tokens
            except Exception:
                config_kwargs["max_output_tokens"] = 8192

        logger.debug("llm_generate_text", role=role.value)

        if GenerateContentConfig is not None:
            response = model.generate_content(prompt, generation_config=GenerateContentConfig(**config_kwargs))
        else:
            response = model.generate_content(prompt)

        return LLMResponse(
            content=response.text or "",
            model=getattr(model, "model_name", "unknown"),
            usage={
                "prompt_tokens": getattr(getattr(response, "usage_metadata", None), "prompt_token_count", 0),
                "completion_tokens": getattr(getattr(response, "usage_metadata", None), "candidates_token_count", 0),
            }
            if getattr(response, "usage_metadata", None)
            else None,
        )


# Public API
get_llm_factory = None  # will be set when module is imported via get_llm_factory()
get_llm = None  # will be set when module is imported via get_llm()


def get_llm_factory_func() -> LLMFactory:
    """Get or create the LLM factory singleton."""
    from stock_analysis.llm import get_llm_factory as _get_llm_factory
    return _get_llm_factory()


def get_llm_func(role: LLMModel) -> Any:
    """Get an LLM model by role."""
    factory = get_llm_factory_func()
    return factory.get_model(role)