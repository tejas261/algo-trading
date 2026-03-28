"""Report generation: CSV trade logs, equity curves, JSON summaries, and charts."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from src.models.trade import Trade, TradeStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Generates backtest reports in various formats.

    All methods are stateless and can be called independently or via
    :meth:`generate_full_report` for a complete output bundle.
    """

    def generate_trade_log_csv(
        self,
        trades: Sequence[Trade],
        path: str | Path,
    ) -> Path:
        """Write a CSV trade log.

        Columns: trade_id, symbol, side, status, entry_price, entry_quantity,
        entry_timestamp, exit_price, exit_timestamp, realized_pnl,
        return_pct, total_commission, duration_hours, tag.

        Args:
            trades: Sequence of Trade objects.
            path: Output file path.

        Returns:
            The resolved output path.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "trade_id",
            "symbol",
            "side",
            "status",
            "entry_price",
            "entry_quantity",
            "entry_timestamp",
            "exit_price",
            "avg_exit_price",
            "exit_timestamp",
            "realized_pnl",
            "return_pct",
            "total_commission",
            "duration_hours",
            "tag",
            "exit_reason",
            "funding_paid",
            "effective_slippage_bps",
            "liquidation_price",
        ]

        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for trade in trades:
                duration_hrs = None
                if trade.duration is not None:
                    duration_hrs = round(trade.duration.total_seconds() / 3600.0, 2)

                writer.writerow({
                    "trade_id": str(trade.trade_id),
                    "symbol": trade.symbol,
                    "side": trade.side.value,
                    "status": trade.status.value,
                    "entry_price": round(trade.entry_price, 6),
                    "entry_quantity": round(trade.entry_quantity, 6),
                    "entry_timestamp": trade.entry_timestamp.isoformat(),
                    "exit_price": round(trade.exit_price, 6) if trade.exit_price else "",
                    "avg_exit_price": round(trade.avg_exit_price, 6) if trade.avg_exit_price else "",
                    "exit_timestamp": trade.exit_timestamp.isoformat() if trade.exit_timestamp else "",
                    "realized_pnl": round(trade.realized_pnl, 4),
                    "return_pct": round(trade.return_pct, 4) if trade.return_pct is not None else "",
                    "total_commission": round(trade.total_commission, 4),
                    "duration_hours": duration_hrs if duration_hrs is not None else "",
                    "tag": trade.tag,
                    "exit_reason": trade.metadata.get("exit_reason", ""),
                    "funding_paid": round(trade.metadata.get("funding_paid", 0.0), 4),
                    "effective_slippage_bps": round(trade.metadata.get("effective_slippage_bps", 0.0), 4),
                    "liquidation_price": round(trade.metadata.get("liquidation_price", 0.0), 2) if trade.metadata.get("liquidation_price") else "",
                })

        logger.info("Trade log CSV written: %s (%d trades)", out, len(trades))
        return out

    def generate_equity_curve_csv(
        self,
        equity_curve: Sequence[tuple[datetime, float]],
        path: str | Path,
    ) -> Path:
        """Write equity curve to CSV.

        Columns: timestamp, equity.

        Args:
            equity_curve: Sequence of (timestamp, equity) tuples.
            path: Output file path.

        Returns:
            The resolved output path.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "equity"])
            for ts, eq in equity_curve:
                writer.writerow([ts.isoformat(), round(eq, 4)])

        logger.info("Equity curve CSV written: %s (%d points)", out, len(equity_curve))
        return out

    def generate_summary_json(
        self,
        metrics: dict[str, Any],
        config: dict[str, Any],
        path: str | Path,
    ) -> Path:
        """Write a JSON summary of metrics and configuration.

        Args:
            metrics: Performance metrics dictionary.
            config: Strategy/backtest configuration.
            path: Output file path.

        Returns:
            The resolved output path.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "generated_at": datetime.utcnow().isoformat(),
            "config": config,
            "metrics": metrics,
        }

        with open(out, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("Summary JSON written: %s", out)
        return out

    def generate_equity_chart(
        self,
        equity_curve: Sequence[tuple[datetime, float]],
        path: str | Path,
    ) -> Path:
        """Generate an equity curve chart as a PNG image.

        Args:
            equity_curve: Sequence of (timestamp, equity) tuples.
            path: Output image path.

        Returns:
            The resolved output path.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if not equity_curve:
            logger.warning("Empty equity curve; skipping chart generation")
            return out

        timestamps = [ts for ts, _ in equity_curve]
        equities = [eq for _, eq in equity_curve]

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(timestamps, equities, linewidth=1.2, color="#2196F3")
        ax.fill_between(timestamps, equities, alpha=0.1, color="#2196F3")

        ax.set_title("Equity Curve", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

        # Draw initial capital reference line
        if equities:
            ax.axhline(y=equities[0], color="gray", linestyle="--", alpha=0.5, label="Initial Capital")
            ax.legend(loc="upper left")

        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)

        logger.info("Equity chart saved: %s", out)
        return out

    def generate_drawdown_chart(
        self,
        equity_curve: Sequence[tuple[datetime, float]],
        path: str | Path,
    ) -> Path:
        """Generate a drawdown chart as a PNG image.

        Args:
            equity_curve: Sequence of (timestamp, equity) tuples.
            path: Output image path.

        Returns:
            The resolved output path.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if not equity_curve:
            logger.warning("Empty equity curve; skipping drawdown chart")
            return out

        timestamps = [ts for ts, _ in equity_curve]
        equities = [eq for _, eq in equity_curve]

        # Compute drawdown series
        peak = equities[0]
        drawdowns: list[float] = []
        for eq in equities:
            if eq > peak:
                peak = eq
            dd_pct = ((peak - eq) / peak) * 100.0 if peak > 0 else 0.0
            drawdowns.append(-dd_pct)  # negative for visual convention

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(timestamps, drawdowns, 0, alpha=0.4, color="#F44336")
        ax.plot(timestamps, drawdowns, linewidth=0.8, color="#D32F2F")

        ax.set_title("Drawdown", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

        # Annotate max drawdown
        if drawdowns:
            min_dd = min(drawdowns)
            min_idx = drawdowns.index(min_dd)
            ax.annotate(
                f"Max DD: {min_dd:.2f}%",
                xy=(timestamps[min_idx], min_dd),
                xytext=(10, -20),
                textcoords="offset points",
                fontsize=9,
                color="#D32F2F",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#D32F2F"),
            )

        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)

        logger.info("Drawdown chart saved: %s", out)
        return out

    def generate_full_report(
        self,
        trades: Sequence[Trade],
        equity_curve: Sequence[tuple[datetime, float]],
        daily_returns: Sequence[float],
        config: dict[str, Any],
        output_dir: str | Path,
        risk_free_rate: float = 0.0,
    ) -> Path:
        """Generate a complete report bundle in the output directory.

        Creates:
            - ``trade_log.csv``
            - ``equity_curve.csv``
            - ``summary.json``
            - ``equity_chart.png``
            - ``drawdown_chart.png``

        Args:
            trades: All trades from the backtest.
            equity_curve: Sequence of (timestamp, equity) tuples.
            daily_returns: Daily return values.
            config: Strategy/backtest configuration dict.
            output_dir: Directory to write all report files into.
            risk_free_rate: Annualized risk-free rate for metrics.

        Returns:
            The output directory path.
        """
        from src.engine.metrics import compute_all_metrics

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Compute metrics
        equity_values = [eq for _, eq in equity_curve]
        metrics = compute_all_metrics(
            trades=trades,
            equity_curve=equity_values,
            daily_returns=daily_returns,
            risk_free_rate=risk_free_rate,
        )

        # Generate all outputs
        self.generate_trade_log_csv(trades, out_dir / "trade_log.csv")
        self.generate_equity_curve_csv(equity_curve, out_dir / "equity_curve.csv")
        self.generate_summary_json(metrics, config, out_dir / "summary.json")
        self.generate_equity_chart(equity_curve, out_dir / "equity_chart.png")
        self.generate_drawdown_chart(equity_curve, out_dir / "drawdown_chart.png")

        logger.info("Full report generated in: %s", out_dir)
        return out_dir
