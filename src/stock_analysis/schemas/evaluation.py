from datetime import date

from pydantic import BaseModel, ConfigDict


class ForecastEvaluationBase(BaseModel):
    forecast_id: int
    actual_prices_json: str
    baseline_metrics_json: str
    llm_metrics_json: str | None = None
    final_metrics_json: str
    evaluation_date: date


class ForecastEvaluationCreate(ForecastEvaluationBase):
    pass


class ForecastEvaluation(ForecastEvaluationBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: str
