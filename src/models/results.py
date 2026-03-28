"""Backtest results and performance metrics models."""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Optional

from pydantic import BaseModel, Field, computed_field

from src.models.portfolio import DailyReturn
from src.models.trade import Trade, TradeStatus


class PerformanceMetrics(BaseModel):
    """Summary statistics for a backtest run."""

    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float

    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: Optional[float] = None
    expectancy: float

    avg_trade_duration_days: Optional[float] = None
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0


class MonthlyReturn(BaseModel):
    """Monthly return summary."""

    model_config = {"frozen": True}

    year: int
    month: int
    return_pct: float
    start_equity: float
    end_equity: float


class BacktestResults(BaseModel):
    """Complete output of a backtest run."""

    strategy_name: str
    symbol: str = ""
    start_date: date
    end_date: date
    initial_capital: float = Field(gt=0)
    final_equity: float

    trade_log: list[Trade] = Field(default_factory=list)
    equity_curve: list[DailyReturn] = Field(default_factory=list)
    daily_returns: list[float] = Field(default_factory=list)
    monthly_returns: list[MonthlyReturn] = Field(default_factory=list)

    metrics: Optional[PerformanceMetrics] = None

    def compute_metrics(self, risk_free_rate_annual: float = 0.0) -> PerformanceMetrics:
        """Derive :class:`PerformanceMetrics` from the trade log and equity curve."""
        closed_trades = [t for t in self.trade_log if t.status == TradeStatus.CLOSED]
        total = len(closed_trades)
        winners = [t for t in closed_trades if t.realized_pnl > 0]
        losers = [t for t in closed_trades if t.realized_pnl <= 0]

        win_count = len(winners)
        loss_count = len(losers)
        win_rate = (win_count / total * 100) if total > 0 else 0.0

        # Average win / loss percentages
        win_pcts = [t.return_pct for t in winners if t.return_pct is not None]
        loss_pcts = [t.return_pct for t in losers if t.return_pct is not None]
        avg_win = sum(win_pcts) / len(win_pcts) if win_pcts else 0.0
        avg_loss = sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0.0

        # Profit factor
        gross_profit = sum(t.realized_pnl for t in winners)
        gross_loss = abs(sum(t.realized_pnl for t in losers))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

        # Expectancy
        avg_pnl = sum(t.realized_pnl for t in closed_trades) / total if total > 0 else 0.0

        # Total & annualized return
        total_return = ((self.final_equity - self.initial_capital) / self.initial_capital) * 100
        num_days = (self.end_date - self.start_date).days or 1
        years = num_days / 365.25
        annualized = ((self.final_equity / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0.0

        # Max drawdown from equity curve
        max_dd = self._max_drawdown()

        # Sharpe / Sortino
        sharpe = self._sharpe(risk_free_rate_annual)
        sortino = self._sortino(risk_free_rate_annual)
        calmar = (annualized / max_dd) if max_dd > 0 else None

        # Consecutive wins / losses
        max_consec_w, max_consec_l = self._consecutive_streaks(closed_trades)

        # Average duration
        durations = [t.duration.total_seconds() / 86400 for t in closed_trades if t.duration is not None]
        avg_dur = sum(durations) / len(durations) if durations else None

        self.metrics = PerformanceMetrics(
            total_return_pct=round(total_return, 4),
            annualized_return_pct=round(annualized, 4),
            max_drawdown_pct=round(max_dd, 4),
            sharpe_ratio=round(sharpe, 4) if sharpe is not None else None,
            sortino_ratio=round(sortino, 4) if sortino is not None else None,
            calmar_ratio=round(calmar, 4) if calmar is not None else None,
            total_trades=total,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate_pct=round(win_rate, 2),
            avg_win_pct=round(avg_win, 4),
            avg_loss_pct=round(avg_loss, 4),
            profit_factor=round(profit_factor, 4) if profit_factor is not None else None,
            expectancy=round(avg_pnl, 4),
            avg_trade_duration_days=round(avg_dur, 2) if avg_dur is not None else None,
            max_consecutive_wins=max_consec_w,
            max_consecutive_losses=max_consec_l,
        )
        return self.metrics

    def compute_monthly_returns(self) -> list[MonthlyReturn]:
        """Aggregate daily equity curve into monthly returns."""
        if not self.equity_curve:
            return []

        monthly: list[MonthlyReturn] = []
        current_month: Optional[tuple[int, int]] = None
        month_start_eq: float = self.initial_capital

        for dr in self.equity_curve:
            key = (dr.date.year, dr.date.month)
            if current_month is None:
                current_month = key
            elif key != current_month:
                # Close previous month
                prev = self.equity_curve[self.equity_curve.index(dr) - 1]
                ret = ((prev.equity - month_start_eq) / month_start_eq * 100) if month_start_eq else 0.0
                monthly.append(MonthlyReturn(
                    year=current_month[0],
                    month=current_month[1],
                    return_pct=round(ret, 4),
                    start_equity=month_start_eq,
                    end_equity=prev.equity,
                ))
                month_start_eq = prev.equity
                current_month = key

        # Final month
        if current_month and self.equity_curve:
            last = self.equity_curve[-1]
            ret = ((last.equity - month_start_eq) / month_start_eq * 100) if month_start_eq else 0.0
            monthly.append(MonthlyReturn(
                year=current_month[0],
                month=current_month[1],
                return_pct=round(ret, 4),
                start_equity=month_start_eq,
                end_equity=last.equity,
            ))

        self.monthly_returns = monthly
        return monthly

    # ----- private helpers -----

    def _max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0].equity
        max_dd = 0.0
        for dr in self.equity_curve:
            if dr.equity > peak:
                peak = dr.equity
            dd = ((peak - dr.equity) / peak) * 100 if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    def _sharpe(self, rfr_annual: float) -> Optional[float]:
        if len(self.daily_returns) < 2:
            return None
        rfr_daily = rfr_annual / 252
        excess = [r - rfr_daily for r in self.daily_returns]
        mean = sum(excess) / len(excess)
        var = sum((x - mean) ** 2 for x in excess) / (len(excess) - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        if std == 0:
            return None
        return (mean / std) * math.sqrt(252)

    def _sortino(self, rfr_annual: float) -> Optional[float]:
        if len(self.daily_returns) < 2:
            return None
        rfr_daily = rfr_annual / 252
        excess = [r - rfr_daily for r in self.daily_returns]
        mean = sum(excess) / len(excess)
        downside = [x for x in excess if x < 0]
        if not downside:
            return None
        down_var = sum(x ** 2 for x in downside) / len(downside)
        down_std = math.sqrt(down_var)
        if down_std == 0:
            return None
        return (mean / down_std) * math.sqrt(252)

    @staticmethod
    def _consecutive_streaks(trades: list[Trade]) -> tuple[int, int]:
        max_w = max_l = cur_w = cur_l = 0
        for t in trades:
            if t.realized_pnl > 0:
                cur_w += 1
                cur_l = 0
            else:
                cur_l += 1
                cur_w = 0
            max_w = max(max_w, cur_w)
            max_l = max(max_l, cur_l)
        return max_w, max_l
