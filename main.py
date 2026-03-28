"""CLI entry point for the algo-trading framework.

Provides subcommands for backtesting, paper trading, live trading,
and running the full agentic LangGraph workflow.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

BASE_CONFIG = "config/base_config.yaml"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file and return its contents as a dict."""
    p = Path(path)
    if not p.is_file():
        logger.warning("Config file not found: %s", p)
        return {}
    with open(p) as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins on leaves)."""
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _apply_common(config: dict[str, Any]) -> dict[str, Any]:
    """Propagate ``common:`` shortcut values into the sections that need them."""
    common = config.get("common", {})
    if not common:
        return config

    symbol = common.get("symbol")
    timeframe = common.get("timeframe")
    market_type = common.get("market_type")
    instrument_type = common.get("instrument_type")
    leverage = common.get("leverage")
    initial_capital = common.get("initial_capital")
    currency = common.get("currency")

    # data section
    data = config.setdefault("data", {})
    if symbol:
        data.setdefault("default_symbol", symbol)
    if timeframe:
        data.setdefault("default_timeframe", timeframe)
    if market_type:
        data.setdefault("market_type", market_type)

    # strategy section
    strategy = config.setdefault("strategy", {})
    if timeframe:
        strategy.setdefault("timeframe", timeframe)
    if market_type:
        strategy.setdefault("market_type", market_type)
    if instrument_type:
        strategy.setdefault("instrument_type", instrument_type)
    if leverage is not None:
        strategy.setdefault("leverage", leverage)

    # perpetual settings → strategy
    perp = config.get("perpetual", {})
    for key in ("initial_margin_fraction", "maintenance_margin_fraction",
                "liquidation_fee_bps", "funding_mode", "funding_bps_per_8h",
                "enable_funding_costs"):
        if key in perp:
            strategy.setdefault(key, perp[key])

    # risk section
    risk = config.setdefault("risk", {})
    if leverage is not None:
        risk.setdefault("leverage", leverage)

    # paper section
    paper = config.setdefault("paper", {})
    if initial_capital is not None:
        paper.setdefault("initial_capital", initial_capital)
    if currency:
        paper.setdefault("currency", currency)

    return config


def load_all_configs(strategy_config_path: str) -> dict[str, Any]:
    """Load base_config.yaml, then deep-merge the strategy config on top.

    The ``common:`` section in base_config provides shortcut values
    (symbol, timeframe, initial_capital, etc.) that are propagated into
    the sections that need them, so you only set them once.
    """
    strategy_path = Path(strategy_config_path)
    config_dir = strategy_path.parent

    # 1. Load base config
    base = _load_yaml(config_dir / Path(BASE_CONFIG).name)

    # 2. Deep-merge strategy-specific overrides on top
    override = _load_yaml(strategy_path)
    merged = _deep_merge(base, override)

    # 3. Propagate common values into relevant sections
    merged = _apply_common(merged)

    return merged


def _parse_date(value: str | None) -> datetime | None:
    """Parse a YYYY-MM-DD string into a datetime, or return None."""
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise click.BadParameter(
            f"Invalid date format '{value}'. Expected YYYY-MM-DD."
        )


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.option(
    "--config",
    "config_path",
    type=click.Path(),
    default="config/app_config.yaml",
    show_default=True,
    help="Path to the main application config YAML.",
)
@click.option(
    "--verbose/--no-verbose",
    default=False,
    show_default=True,
    help="Enable verbose (DEBUG-level) logging.",
)
@click.pass_context
def cli(ctx: click.Context, config_path: str, verbose: bool) -> None:
    """AlgoTrader -- quantitative trading framework."""
    import logging

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    ctx.obj["verbose"] = verbose
    ctx.obj["config"] = load_all_configs(config_path)


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--data", "data_path", required=True, type=click.Path(exists=True), help="Path to OHLCV CSV file.")
@click.option("--symbol", default=None, help="Trading symbol (default: from config).")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config).")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD).")
@click.option("--end", default=None, help="End date (YYYY-MM-DD).")
@click.option("--output-dir", default=None, help="Directory for reports (default: from config).")
@click.option("--initial-capital", default=None, type=float, help="Starting equity (default: from config).")
@click.pass_context
def backtest(
    ctx: click.Context,
    data_path: str,
    symbol: str | None,
    timeframe: str | None,
    start: str | None,
    end: str | None,
    output_dir: str | None,
    initial_capital: float | None,
) -> None:
    """Run a historical backtest on CSV data."""
    from src.adapters.data.csv_data_adapter import CsvDataAdapter
    from src.engine.backtester import Backtester
    from src.engine.reporting import ReportGenerator
    from src.strategy.strategy import TrendBreakoutStrategy

    config = ctx.obj["config"]
    common = config.get("common", {})
    data_config: dict[str, Any] = config.get("data", {})

    # Resolve defaults from config (CLI flags override)
    symbol = symbol or common.get("symbol") or data_config.get("default_symbol", "BTCUSDT")
    timeframe = timeframe or common.get("timeframe") or data_config.get("default_timeframe", "4h")
    output_dir = output_dir or config.get("app", {}).get("output_dir", "output")
    initial_capital = initial_capital or common.get("initial_capital") or 10_000.0

    start_dt = _parse_date(start)
    end_dt = _parse_date(end)

    # ---- Load data --------------------------------------------------------
    csv_path = Path(data_path).resolve()
    adapter = CsvDataAdapter(
        data_dir=csv_path.parent,
        file_map={(symbol, timeframe): str(csv_path)},
        gap_detection=data_config.get("gap_detection", "warn"),
        market_type=data_config.get("market_type", "crypto"),
    )
    adapter.connect()

    click.echo(f"Loading data from {csv_path} ...")
    df = adapter.fetch_ohlcv(symbol, timeframe, start=start_dt, end=end_dt)
    adapter.disconnect()
    click.echo(f"Loaded {len(df)} candles  [{df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}]")

    # ---- Run strategy -----------------------------------------------------
    strategy_config: dict[str, Any] = config.get("strategy", {})
    indicator_cfg = strategy_config.get("indicators", {})

    strategy = TrendBreakoutStrategy(config=indicator_cfg)
    click.echo("Computing indicators and signals ...")
    df = strategy.run(df)

    # ---- Prepare configs for backtester -----------------------------------
    risk_config: dict[str, Any] = config.get("risk", {})
    broker_key = config.get("execution", {}).get("adapter", "paper")
    broker_section: dict[str, Any] = config.get(broker_key, {})
    cost_config: dict[str, Any] = broker_section.get("cost_model", {})

    exit_cfg = strategy_config.get("exit", {})
    instrument_cfg: dict[str, Any] = config.get("instrument", {})

    bt_strategy_config: dict[str, Any] = {
        "strategy_name": strategy_config.get("name", "trend_breakout_v2"),
        "symbol": symbol,
        "exit_mode": exit_cfg.get("exit_mode", "trend_following"),
        "stop_loss_atr_mult": float(exit_cfg.get("initial_stop_atr_multiple", 2.0)),
        "trailing_stop_atr_mult": float(exit_cfg.get("trailing_stop_atr_multiple", 3.0)),
        "atr_column": "atr_14",
        "atr_mean_column": "atr_14_sma_50",
        "signal_column": "signal",
        # Perpetual / liquidation modeling
        "instrument_type": strategy_config.get("instrument_type", "spot"),
        "leverage": float(strategy_config.get("leverage", 1.0)),
        "initial_margin_fraction": float(strategy_config.get("initial_margin_fraction", 1.0)),
        "maintenance_margin_fraction": float(strategy_config.get("maintenance_margin_fraction", 0.005)),
        "liquidation_fee_bps": float(strategy_config.get("liquidation_fee_bps", 50)),
        # Funding cost modeling
        "funding_mode": strategy_config.get("funding_mode", "constant"),
        "funding_column_name": strategy_config.get("funding_column_name", "funding_rate"),
        # Metrics annualization
        "timeframe": timeframe,
        "market_type": strategy_config.get("market_type", "crypto"),
    }

    # Merge instrument spec into cost_config for CostModel.from_config()
    cost_config_full: dict[str, Any] = {**cost_config}
    cost_config_full["instrument_type"] = strategy_config.get("instrument_type", "spot")
    cost_config_full["min_notional"] = instrument_cfg.get("min_notional", 10.0)
    cost_config_full["min_qty"] = instrument_cfg.get("min_qty", 0.00001)
    cost_config_full["qty_step"] = instrument_cfg.get("qty_step", 0.00001)
    cost_config_full["price_tick"] = instrument_cfg.get("price_tick", 0.01)
    cost_config_full["assumed_daily_volume"] = instrument_cfg.get("assumed_daily_volume", 1_000_000_000.0)

    # ---- Run backtester ---------------------------------------------------
    click.echo(f"Running backtest (capital={initial_capital:,.0f}) ...")
    bt = Backtester(
        df=df,
        strategy_config=bt_strategy_config,
        risk_config=risk_config,
        cost_config=cost_config_full,
        initial_capital=initial_capital,
    )
    results = bt.run()

    # ---- Generate reports -------------------------------------------------
    click.echo(f"Generating reports in {output_dir}/ ...")
    reporter = ReportGenerator()
    portfolio_tracker = bt._portfolio  # noqa: SLF001
    equity_curve = portfolio_tracker.get_equity_curve()

    report_config = {
        "symbol": symbol,
        "timeframe": timeframe,
        "initial_capital": initial_capital,
        "strategy": bt_strategy_config,
        "risk": risk_config,
        "costs": cost_config_full,
    }

    reporter.generate_full_report(
        trades=results.trade_log,
        equity_curve=equity_curve,
        daily_returns=results.daily_returns,
        config=report_config,
        output_dir=output_dir,
    )

    # ---- Print summary ----------------------------------------------------
    _print_backtest_summary(results, initial_capital)


def _build_target_list(strategy_config: dict[str, Any]) -> list[dict[str, float]] | None:
    """Convert strategy exit config into the target format expected by Backtester."""
    exit_cfg = strategy_config.get("exit", {})
    partial_exits = exit_cfg.get("partial_exits", [])
    trailing = exit_cfg.get("trailing", {})

    if not partial_exits and not trailing:
        return None

    targets: list[dict[str, float]] = []
    for pe in partial_exits:
        targets.append({
            "atr_mult": float(pe.get("target_r", 2.0)),
            "pct": float(pe.get("pct", 0.0)) * 100,  # convert fraction -> pct
        })

    # Trailing portion as the final target (large ATR mult so trailing stop governs)
    if trailing:
        trailing_pct = float(trailing.get("pct", 0.0)) * 100
        targets.append({
            "atr_mult": 5.0,  # far target; trailing stop handles the actual exit
            "pct": trailing_pct,
        })

    # Normalise percentages to 100
    total = sum(t["pct"] for t in targets)
    if total > 0 and abs(total - 100.0) > 1e-6:
        for t in targets:
            t["pct"] = (t["pct"] / total) * 100.0

    return targets if targets else None


def _print_backtest_summary(results: Any, initial_capital: float) -> None:
    """Print a formatted summary of backtest results to the console."""
    m = results.metrics
    click.echo("")
    click.echo("=" * 60)
    click.echo("  BACKTEST RESULTS")
    click.echo("=" * 60)
    click.echo(f"  Strategy      : {results.strategy_name}")
    click.echo(f"  Symbol        : {results.symbol}")
    click.echo(f"  Period        : {results.start_date} -> {results.end_date}")
    click.echo(f"  Initial Capital: {initial_capital:>14,.2f}")
    click.echo(f"  Final Equity  : {results.final_equity:>14,.2f}")
    click.echo("-" * 60)

    if m is None:
        click.echo("  No trades were executed.")
        click.echo("=" * 60)
        return

    click.echo(f"  Total Return  : {m.total_return_pct:>+10.2f} %")
    click.echo(f"  Ann. Return   : {m.annualized_return_pct:>+10.2f} %")
    click.echo(f"  Max Drawdown  : {m.max_drawdown_pct:>10.2f} %")
    click.echo(f"  Sharpe Ratio  : {_fmt_ratio(m.sharpe_ratio)}")
    click.echo(f"  Sortino Ratio : {_fmt_ratio(m.sortino_ratio)}")
    click.echo(f"  Calmar Ratio  : {_fmt_ratio(m.calmar_ratio)}")
    click.echo("-" * 60)
    click.echo(f"  Total Trades  : {m.total_trades:>10d}")
    click.echo(f"  Win Rate      : {m.win_rate_pct:>10.2f} %")
    click.echo(f"  Avg Win       : {m.avg_win_pct:>+10.2f} %")
    click.echo(f"  Avg Loss      : {m.avg_loss_pct:>+10.2f} %")
    click.echo(f"  Profit Factor : {_fmt_ratio(m.profit_factor)}")
    click.echo(f"  Expectancy    : {m.expectancy:>+10.2f}")
    click.echo(f"  Max Consec W  : {m.max_consecutive_wins:>10d}")
    click.echo(f"  Max Consec L  : {m.max_consecutive_losses:>10d}")

    if m.avg_trade_duration_days is not None:
        click.echo(f"  Avg Duration  : {m.avg_trade_duration_days:>10.1f} days")

    click.echo("=" * 60)


def _fmt_ratio(value: float | None) -> str:
    """Format an optional ratio for display."""
    if value is None:
        return "       N/A"
    return f"{value:>10.4f}"


# ---------------------------------------------------------------------------
# paper
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--data", "data_path", required=True, type=click.Path(exists=True), help="Path to OHLCV CSV file.")
@click.option("--symbol", default="NIFTY", show_default=True, help="Trading symbol.")
@click.option("--timeframe", default="1h", show_default=True, help="Candle timeframe.")
@click.pass_context
def paper(ctx: click.Context, data_path: str, symbol: str, timeframe: str) -> None:
    """Run a paper-trading session using the LangGraph agent."""
    from src.agents.graph import run_trading_cycle

    config = ctx.obj["config"]

    click.echo("Starting paper trading session ...")
    click.echo(f"  Symbol    : {symbol}")
    click.echo(f"  Timeframe : {timeframe}")
    click.echo(f"  Data      : {data_path}")
    click.echo("")

    run_trading_cycle(
        mode="paper",
        symbol=symbol,
        timeframe=timeframe,
        data_path=data_path,
        config=config,
    )


# ---------------------------------------------------------------------------
# live
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--symbol", default="NIFTY", show_default=True, help="Trading symbol.")
@click.option("--timeframe", default="1h", show_default=True, help="Candle timeframe.")
@click.option("--broker", default="paper", show_default=True, help="Broker adapter to use (paper/delta/groww).")
@click.pass_context
def live(ctx: click.Context, symbol: str, timeframe: str, broker: str) -> None:
    """Run in live trading mode with human approval for trades.

    WARNING: This mode places real orders through the configured broker.
    Ensure you understand the risks before proceeding.
    """
    from src.agents.graph import run_trading_cycle

    config = ctx.obj["config"]

    click.echo("")
    click.secho("=" * 60, fg="red", bold=True)
    click.secho("  WARNING: LIVE TRADING MODE", fg="red", bold=True)
    click.secho("  Real orders will be placed via the broker.", fg="red", bold=True)
    click.secho("  All trades require human approval.", fg="red")
    click.secho("=" * 60, fg="red", bold=True)
    click.echo("")
    click.echo(f"  Symbol    : {symbol}")
    click.echo(f"  Timeframe : {timeframe}")
    click.echo(f"  Broker    : {broker}")
    click.echo("")

    if not click.confirm("Do you want to proceed with live trading?"):
        click.echo("Aborted.")
        raise SystemExit(0)

    # Override broker in config
    config.setdefault("execution", {})["adapter"] = broker

    run_trading_cycle(
        mode="live",
        symbol=symbol,
        timeframe=timeframe,
        config=config,
        human_approval=True,
    )


# ---------------------------------------------------------------------------
# agent
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--mode", required=True, type=click.Choice(["backtest", "paper", "live"]), help="Execution mode.")
@click.option("--symbol", default="NIFTY", show_default=True, help="Trading symbol.")
@click.option("--data", "data_path", default=None, type=click.Path(), help="Path to OHLCV CSV file (required for backtest/paper).")
@click.option("--timeframe", default="1h", show_default=True, help="Candle timeframe.")
@click.pass_context
def agent(ctx: click.Context, mode: str, symbol: str, data_path: str | None, timeframe: str) -> None:
    """Run the full agentic LangGraph workflow.

    Delegates to ``run_trading_cycle`` with the specified mode.
    """
    from src.agents.graph import run_trading_cycle

    config = ctx.obj["config"]

    if mode in ("backtest", "paper") and data_path is None:
        raise click.UsageError("--data is required for backtest and paper modes.")

    click.echo(f"Starting agent in {mode} mode ...")
    click.echo(f"  Symbol    : {symbol}")
    click.echo(f"  Timeframe : {timeframe}")
    if data_path:
        click.echo(f"  Data      : {data_path}")
    click.echo("")

    human_approval = mode == "live"

    run_trading_cycle(
        mode=mode,
        symbol=symbol,
        timeframe=timeframe,
        data_path=data_path,
        config=config,
        human_approval=human_approval,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
