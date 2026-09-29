from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class AuditLogBase(BaseModel):
    entity_type: str = Field(..., max_length=50)
    entity_id: int
    action: str = Field(..., max_length=50)
    changes_json: str | None = None
    user_id: str | None = Field(None, max_length=100)


class AuditLogCreate(AuditLogBase):
    pass


class AuditLog(AuditLogBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime