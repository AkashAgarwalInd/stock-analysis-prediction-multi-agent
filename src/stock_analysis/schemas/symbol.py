from pydantic import BaseModel, ConfigDict, Field


class SymbolBase(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20, pattern=r"^[A-Z0-9\.\-]+$")
    name: str = Field(..., min_length=1, max_length=200)
    sector: str | None = Field(None, max_length=100)
    industry: str | None = Field(None, max_length=100)
    exchange: str = Field(default="NSE", max_length=20)
    is_active: bool = True


class SymbolCreate(SymbolBase):
    pass


class SymbolUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    sector: str | None = Field(None, max_length=100)
    industry: str | None = Field(None, max_length=100)
    exchange: str | None = Field(None, max_length=20)
    is_active: bool | None = None


class Symbol(SymbolBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: str
    updated_at: str