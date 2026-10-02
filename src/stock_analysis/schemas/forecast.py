from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PricePoint(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    date: date
    open: Decimal | None = None
    high: Decimal | None = None
    low: Decimal | None = None
    close: Decimal
    volume: int | None = None


class ForecastBaseline(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    forecast_date: date
    horizon_days: int
    predictions: list[PricePoint]
    method: str
    metadata: dict = Field(default_factory=dict)


class ForecastLLM(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    forecast_date: date
    horizon_days: int
    predictions: list[PricePoint]
    reasoning: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    adjustments: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


class ForecastFinal(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    forecast_date: date
    horizon_days: int
    predictions: list[PricePoint]
    baseline_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    llm_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_weights(self) -> "ForecastFinal":
        total = self.baseline_weight + self.llm_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError("baseline_weight and llm_weight must sum to 1.0")
        return self


class ForecastBase(BaseModel):
    symbol_id: int
    forecast_date: date
    horizon_days: int = 5
    baseline_forecast_json: str
    llm_forecast_json: str | None = None
    final_forecast_json: str
    confidence_score: float | None = Field(None, ge=0.0, le=1.0)
    status: Literal["pending", "completed", "evaluated", "archived"] = "pending"


class ForecastCreate(ForecastBase):
    pass


class ForecastUpdate(BaseModel):
    llm_forecast_json: str | None = None
    final_forecast_json: str | None = None
    confidence_score: float | None = Field(None, ge=0.0, le=1.0)
    status: Literal["pending", "completed", "evaluated", "archived"] | None = None


class Forecast(ForecastBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: str
