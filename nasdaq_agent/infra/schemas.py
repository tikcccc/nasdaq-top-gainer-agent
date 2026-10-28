"""Validated data shapes that cross tool boundaries."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Quality / risk flags attached to screener rows.
FLAG_DERIVATIVE = "derivative"          # warrant, right, unit, preferred...
FLAG_PENNY = "penny"                    # price < $1
FLAG_LOW_VOLUME = "low_volume"
FLAG_MICRO_CAP = "micro_cap"
FLAG_INCONSISTENT = "inconsistent"      # price / change / pct don't reconcile
FLAG_STALE = "stale"
# Any of these leaves the selected stock unverified as the top gainer in scope.
UNVERIFIED_FLAGS = frozenset({"data_conflict", "possible_split_artifact", FLAG_STALE, "source_conflict",
                              "ranking_unverified", "eligibility_unconfirmed", "unverified_quote"})


class GainerRow(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    symbol: str
    name: str = ""
    price: float = Field(gt=0)
    pct_change: float                   # percent units: 182.82 means +182.82%
    net_change: float | None = None
    volume: int | None = Field(default=None, ge=0)
    market_cap: float | None = None
    exchange: str | None = None
    quote_type: str | None = None
    market_time: str | None = None      # ISO timestamp of the quote, when the source provides one
    flags: list[str] = []
    source: str = ""

    @field_validator("symbol")
    @classmethod
    def _symbol(cls, v: str) -> str:
        v = v.strip().upper()
        if not v or len(v) > 10:
            raise ValueError("bad symbol")
        return v


class Headline(BaseModel):
    id: str
    title: str
    publisher: str = ""
    published: str | None = None
    link: str | None = None
    source: str = ""
    symbol: str = ""
    trade_date: str = ""


Trend = Literal["up", "down", "sideways"]


class AnalysisResult(BaseModel):
    """What the agent-generated analysis code must produce. Returns are fractions (0.05 = +5%)."""

    model_config = ConfigDict(allow_inf_nan=False)
    symbol: str
    days: int = Field(ge=1)
    daily_returns: list[float]
    avg_daily_return: float
    cumulative_return: float
    trend: Trend
    trend_slope: float
    volatility_daily: float = Field(ge=0)
    volatility_annualized: float = Field(ge=0)
    max_drawdown: float = Field(ge=-1, le=0)
    benchmark: dict[str, float]          # e.g. {"GSPC": 0.004, "IXIC": 0.011} cumulative returns
    excess_return: float | None          # null when GSPC is unavailable
    notes: str | None = None

    @field_validator("trend", mode="before")
    @classmethod
    def _trend(cls, v: str) -> str:
        v = str(v).strip().lower()
        return {"upward": "up", "uptrend": "up", "downward": "down", "downtrend": "down", "flat": "sideways"}.get(v, v)

    @field_validator("benchmark", mode="before")
    @classmethod
    def _benchmark_keys(cls, v: dict) -> dict:
        return {str(k).lstrip("^").upper(): float(x) for k, x in dict(v).items()}

    @model_validator(mode="after")
    def _shape(self) -> "AnalysisResult":
        if len(self.daily_returns) != self.days:
            raise ValueError(f"daily_returns has {len(self.daily_returns)} values but days={self.days}")
        if any(r < -1 for r in self.daily_returns):
            raise ValueError("daily_returns cannot be below -100%; use fractions")
        return self


class SentimentResult(BaseModel):
    method: Literal["jev", "llm", "none"]
    n_headlines: int = 0
    score: float | None = None            # 0 (very negative) .. 4 (very positive)
    label: str = "no news"
    confidence: float | None = None
    catalyst: str | None = None
    catalyst_confidence: float | None = None
    dilution_risk: float | None = None
    uncertain: bool = True
    headline_ids: list[str] = []
