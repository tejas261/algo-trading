"""Performance metrics computation for backtesting results.

All functions are pure (stateless) and operate on lists/sequences.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from src.models.trade import Trade, TradeStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Annualization factor inference
# ------------------------------------------------------------------

_TIMEFRAME_MULTIPLIERS: dict[str, dict[str, int]] = {
    # crypto: 365 days * 24 hours = 8760 hours/year
    # equity: 252 days * 6.5 hours = 1638 hours/year
    "1m":  {"crypto": 525_600, "equity": 98_280},   # minutes in a year
    "5m":  {"crypto": 105_120, "equity": 19_656},
    "15m": {"crypto": 35_040,  "equity": 6_552},
    "1h":  {"crypto": 8_760,   "equity": 1_638},
    "4h":  {"crypto": 2_190,   "equity": 410},      # 8760/4, 1638/4 rounded
    "1d":  {"crypto": 365,     "equity": 252},
}


def infer_periods_per_year(timeframe: str = "1h", market_type: str = "crypto") -> int:
    """Infer annualization factor from timeframe and market type.

    crypto markets: 365 days, 24 hours
    equity markets: 252 days, 6.5 hours

    Args:
        timeframe: Candle/bar timeframe string (e.g. "1m", "5m", "15m",
            "1h", "4h", "1d").
        market_type: Either ``"crypto"`` or ``"equity"``.

    Returns:
        Number of periods in one year for the given timeframe and market.

    Raises:
        ValueError: If *timeframe* or *market_type* is not recognised.
    """
    tf = timeframe.lower().strip()
    mt = market_type.lower().strip()

    if tf not in _TIMEFRAME_MULTIPLIERS:
        raise ValueError(
            f"Unknown timeframe {timeframe!r}. "
            f"Supported: {sorted(_TIMEFRAME_MULTIPLIERS)}"
        )
    entry = _TIMEFRAME_MULTIPLIERS[tf]

    if mt not in entry:
        raise ValueError(
            f"Unknown market_type {market_type!r}. Supported: {sorted(entry)}"
        )

    return entry[mt]


def compute_sharpe(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute the annualized Sharpe ratio.

    Args:
        returns: Periodic returns (e.g. daily percentage returns as decimals).
        risk_free_rate: Annualized risk-free rate (e.g. 0.05 for 5%).
        periods_per_year: Number of return periods in a year (252 for daily).

    Returns:
        Annualized Sharpe ratio, or 0.0 if insufficient data.
    """
    if len(returns) < 2:
        return 0.0

    rfr_per_period = risk_free_rate / periods_per_year
    excess = [r - rfr_per_period for r in returns]
    mean_excess = sum(excess) / len(excess)
    variance = sum((x - mean_excess) ** 2 for x in excess) / (len(excess) - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0

    if std == 0.0:
        return 0.0

    return (mean_excess / std) * math.sqrt(periods_per_year)


def compute_sortino(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute the annualized Sortino ratio.

    Uses downside deviation (only negative excess returns) instead of
    total standard deviation.

    Args:
        returns: Periodic returns.
        risk_free_rate: Annualized risk-free rate.
        periods_per_year: Number of return periods in a year.

    Returns:
        Annualized Sortino ratio, or 0.0 if insufficient data.
    """
    if len(returns) < 2:
        return 0.0

    rfr_per_period = risk_free_rate / periods_per_year
    excess = [r - rfr_per_period for r in returns]
    mean_excess = sum(excess) / len(excess)

    downside = [x for x in excess if x < 0]
    if not downside:
        return 0.0

    downside_variance = sum(x ** 2 for x in downside) / len(downside)
    downside_std = math.sqrt(downside_variance)

    if downside_std == 0.0:
        return 0.0

    return (mean_excess / downside_std) * math.sqrt(periods_per_year)


def compute_max_drawdown(equity_curve: Sequence[float]) -> float:
    """Compute the maximum drawdown from an equity curve.

    Args:
        equity_curve: Sequence of equity values in chronological order.

    Returns:
        Maximum drawdown as a positive percentage (e.g. 15.5 for 15.5%).
        Returns 0.0 for empty or single-point curves.
    """
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0

    for equity in equity_curve:
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = ((peak - equity) / peak) * 100.0
            max_dd = max(max_dd, dd)

    return max_dd


def compute_calmar(
    returns: Sequence[float],
    max_dd: float,
    periods_per_year: int = 252,
) -> float:
    """Compute the Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Periodic returns.
        max_dd: Maximum drawdown as a positive percentage.
        periods_per_year: Number of return periods in a year.

    Returns:
        Calmar ratio, or 0.0 if max drawdown is zero.
    """
    if max_dd <= 0.0 or len(returns) == 0:
        return 0.0

    mean_return = sum(returns) / len(returns)
    annualized_return = mean_return * periods_per_year
    return (annualized_return / max_dd) * 100.0


def compute_win_rate(trades: Sequence[Trade]) -> float:
    """Compute win rate as a percentage.

    Only closed trades are considered.  A "win" is a trade with
    positive realized PnL.

    Args:
        trades: Sequence of Trade objects.

    Returns:
        Win rate as a percentage (e.g. 55.0 for 55%).
    """
    closed = [t for t in trades if t.status == TradeStatus.CLOSED]
    if not closed:
        return 0.0

    winners = sum(1 for t in closed if t.realized_pnl > 0)
    return (winners / len(closed)) * 100.0


def compute_profit_factor(trades: Sequence[Trade]) -> float:
    """Compute profit factor (gross profits / gross losses).

    Args:
        trades: Sequence of Trade objects.

    Returns:
        Profit factor.  Returns 0.0 if there are no losses,
        and 0.0 if there are no closed trades.
    """
    closed = [t for t in trades if t.status == TradeStatus.CLOSED]
    if not closed:
        return 0.0

    gross_profit = sum(t.realized_pnl for t in closed if t.realized_pnl > 0)
    gross_loss = abs(sum(t.realized_pnl for t in closed if t.realized_pnl <= 0))

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def compute_expectancy(trades: Sequence[Trade]) -> float:
    """Compute expectancy (average PnL per trade).

    Args:
        trades: Sequence of Trade objects.

    Returns:
        Average realized PnL per closed trade.
    """
    closed = [t for t in trades if t.status == TradeStatus.CLOSED]
    if not closed:
        return 0.0

    total_pnl = sum(t.realized_pnl for t in closed)
    return total_pnl / len(closed)


def compute_all_metrics(
    trades: Sequence[Trade],
    equity_curve: Sequence[float],
    daily_returns: Sequence[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    timeframe: str | None = None,
    market_type: str | None = None,
) -> dict[str, Any]:
    """Compute a comprehensive set of performance metrics.

    Args:
        trades: All trades from the backtest.
        equity_curve: Equity values in chronological order.
        daily_returns: Daily return percentages (as decimals, e.g. 0.01 = 1%).
        risk_free_rate: Annualized risk-free rate.
        periods_per_year: Number of return observations per year.  This is
            used as the default but can be overridden by providing
            *timeframe* and *market_type*.
        timeframe: Optional candle timeframe (e.g. ``"1h"``, ``"1d"``).
            When provided together with *market_type*, the annualization
            factor is inferred via :func:`infer_periods_per_year`.
        market_type: Optional market type (``"crypto"`` or ``"equity"``).

    Returns:
        Dictionary with all computed metrics.
    """
    # If timeframe + market_type supplied, infer the annualization factor
    # (explicit periods_per_year still wins when caller sets it to a
    # non-default value).
    if timeframe is not None and market_type is not None:
        periods_per_year = infer_periods_per_year(timeframe, market_type)

    closed = [t for t in trades if t.status == TradeStatus.CLOSED]
    winners = [t for t in closed if t.realized_pnl > 0]
    losers = [t for t in closed if t.realized_pnl <= 0]

    max_dd = compute_max_drawdown(equity_curve)
    sharpe = compute_sharpe(daily_returns, risk_free_rate, periods_per_year)
    sortino = compute_sortino(daily_returns, risk_free_rate, periods_per_year)
    calmar = compute_calmar(daily_returns, max_dd, periods_per_year)
    win_rate = compute_win_rate(trades)
    profit_factor = compute_profit_factor(trades)
    expectancy = compute_expectancy(trades)

    # Average win/loss percentages
    win_pcts = [t.return_pct for t in winners if t.return_pct is not None]
    loss_pcts = [t.return_pct for t in losers if t.return_pct is not None]
    avg_win_pct = sum(win_pcts) / len(win_pcts) if win_pcts else 0.0
    avg_loss_pct = sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0.0

    # Total return
    total_return_pct = 0.0
    if len(equity_curve) >= 2 and equity_curve[0] > 0:
        total_return_pct = ((equity_curve[-1] - equity_curve[0]) / equity_curve[0]) * 100.0

    # Consecutive streaks
    max_consec_wins = 0
    max_consec_losses = 0
    cur_w = cur_l = 0
    for t in closed:
        if t.realized_pnl > 0:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        max_consec_wins = max(max_consec_wins, cur_w)
        max_consec_losses = max(max_consec_losses, cur_l)

    # Average trade duration
    durations = [
        t.duration.total_seconds() / 86400.0
        for t in closed
        if t.duration is not None
    ]
    avg_duration_days = sum(durations) / len(durations) if durations else None

    return {
        "total_return_pct": round(total_return_pct, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "total_trades": len(closed),
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate_pct": round(win_rate, 2),
        "avg_win_pct": round(avg_win_pct, 4),
        "avg_loss_pct": round(avg_loss_pct, 4),
        "profit_factor": round(profit_factor, 4) if not math.isinf(profit_factor) else None,
        "expectancy": round(expectancy, 4),
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        "avg_trade_duration_days": round(avg_duration_days, 2) if avg_duration_days is not None else None,
    }
