from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict


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