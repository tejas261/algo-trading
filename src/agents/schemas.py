"""Pydantic schemas for structured LLM outputs in the trading workflow.

These models are used with LangChain's ``with_structured_output`` so the
LLM returns validated, typed data instead of free-form text.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Strategy / signal
# ------------------------------------------------------------------

class SignalDecision(BaseModel):
    """Structured output from the signal-analysis LLM call.

    ``signal_type`` drives downstream logic.  ``confidence_note`` is a
    human-readable summary the LLM produces -- it is logged but never
    used for actual trading decisions.
    """

    signal_type: str = Field(
        description="Trade signal: LONG, SHORT, or NO_SIGNAL.",
    )
    confidence_note: str = Field(
        description=(
            "LLM-generated qualitative summary of signal confidence. "
            "For audit/logging only -- NOT used for trading decisions."
        ),
    )
    indicators_summary: dict = Field(
        default_factory=dict,
        description="Key indicator values that informed the signal.",
    )


# ------------------------------------------------------------------
# Risk
# ------------------------------------------------------------------

class RiskDecisionSchema(BaseModel):
    """Structured output from the risk-assessment LLM call."""

    approved: bool = Field(
        description="Whether the trade passes all risk checks.",
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="List of reasons for approval or rejection.",
    )
    position_size: float = Field(
        ge=0,
        description="Approved position size in base units.",
    )
    risk_per_trade_pct: float = Field(
        ge=0,
        le=100,
        description="Percentage of equity risked on this trade.",
    )


# ------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------

class ExecutionRequest(BaseModel):
    """Order request sent to the execution adapter."""

    order_type: str = Field(
        description="Order type: MARKET, LIMIT, STOP, or STOP_LIMIT.",
    )
    symbol: str
    side: str = Field(description="LONG or SHORT.")
    quantity: float = Field(gt=0)
    price: float | None = Field(
        default=None,
        description="Limit price (None for market orders).",
    )
    stop_price: float | None = Field(
        default=None,
        description="Stop trigger price (None unless stop order).",
    )
    idempotency_key: str = Field(
        description="Unique key to prevent duplicate order submission.",
    )


class ExecutionResult(BaseModel):
    """Result returned after an order is submitted / filled."""

    success: bool
    order_id: str | None = None
    fill_price: float | None = None
    fill_quantity: float | None = None
    error: str | None = None
    timestamp: str = Field(
        description="ISO-8601 timestamp of the execution event.",
    )


# ------------------------------------------------------------------
# Alerts / monitoring
# ------------------------------------------------------------------

class AlertEvent(BaseModel):
    """Structured alert emitted by the monitoring node."""

    level: str = Field(
        description="Severity level: INFO, WARNING, or CRITICAL.",
    )
    event_type: str = Field(
        description="Machine-readable event category (e.g. 'drawdown_breach').",
    )
    message: str
    data: dict | None = Field(
        default=None,
        description="Optional payload with contextual data.",
    )
    timestamp: str = Field(
        description="ISO-8601 timestamp of the alert.",
    )


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------

class DailyReport(BaseModel):
    """End-of-day summary report."""

    date: str = Field(description="Report date in YYYY-MM-DD format.")
    symbol: str
    mode: str = Field(description="Execution mode: backtest, paper, or live.")
    trades_today: int = Field(ge=0)
    pnl_today: float
    equity: float
    drawdown_pct: float = Field(ge=0)
    open_positions: int = Field(ge=0)
    alerts: list[str] = Field(default_factory=list)
    summary: str = Field(
        description="LLM-generated natural-language summary of the day.",
    )
