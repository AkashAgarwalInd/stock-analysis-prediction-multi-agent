from datetime import date
from pydantic import BaseModel, ConfigDict


class PostmortemBase(BaseModel):
    forecast_id: int
    analysis_json: str
    lessons_learned: str | None = None
    calibration_adjustment_json: str | None = None


class PostmortemCreate(PostmortemBase):
    pass


class Postmortem(PostmortemBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: str