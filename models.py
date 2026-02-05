"""Optional Pydantic models for portfolio API/validation.

Currently tools use inline schemas in tools.py. These models are available for
shared validation, API boundaries, or future use (e.g. REST API)."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import date

AssetClass = Literal["Stock", "Bond", "Cash", "ETF"]
RiskProfile = Literal["Conservative", "Moderate", "Aggressive"]

class DailyRow(BaseModel):
    Date: date
    StockValue: float = Field(ge=0)
    BondValue: float = Field(ge=0)
    ETFValue: float = Field(ge=0)
    CashValue: float = Field(ge=0)

class EditAction(BaseModel):
    action: Literal["add", "update", "remove"]
    Date: date
    StockValue: Optional[float] = Field(default=None, ge=0)
    BondValue: Optional[float] = Field(default=None, ge=0)
    CashValue: Optional[float] = Field(default=None, ge=0)
    ETFValue: Optional[float] = Field(default=None, ge=0)

class PerformanceRequest(BaseModel):
    start: date
    end: date
    asset_class: Optional[AssetClass] = None  # None => total

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v, info):
        start = info.data.get("start")
        if start and v < start:
            raise ValueError("end must be >= start")
        return v

class TrendRequest(BaseModel):
    include_asset_classes: Optional[list[AssetClass]] = None  # None => held classes
    ask_to_recommend_risk_change: bool = True
