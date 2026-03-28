from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import requests


@dataclass
class BacktestConfig:
    symbol: str = "BTCUSD"           # Delta India product symbol, e.g. BTCUSD or ETHUSD
    resolution: str = "1h"
    start: str = "2024-01-01"
    end: str = "2026-01-01"
    initial_capital_inr: float = 10000.0
    leverage: float = 5.0             # IMPORTANT: target/SL below are on margin/trade amount, not raw price
    lot_size: float = 0.001           # BTCUSD on Delta India is 0.001 BTC per lot; ETH often 0.01 ETH.
    lots: int = 1
    target_return_on_margin: float = 0.50   # +50% on trade margin
    stop_return_on_margin: float = 0.30     # -30% on trade margin
    taker_fee_rate: float = 0.0005          # 0.05%
    gst_on_fees: float = 0.18
    slippage_rate: float = 0.0005           # 0.05% each side as default
    funding_rate_per_8h: float = 0.0        # set non-zero if you want to approximate funding cost
    use_mark_price_for_exits: bool = False
    risk_guard_max_margin_fraction: float = 0.25  # cap margin allocation per new sequence to avoid suicidal sizing


class DeltaHistoricalClient:
    BASE_URL = "https://api.india.delta.exchange/v2/history/candles"

    def fetch_candles(self, symbol: str, resolution: str, start: str, end: str) -> pd.DataFrame:
        start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
        end_ts = int(pd.Timestamp(end, tz="UTC").timestamp())

        step_seconds = {
            "1m": 60,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "6h": 21600,
            "1d": 86400,
            "1w": 604800,
        }[resolution]

        max_candles = 2000
        chunk_span = step_seconds * max_candles
        out: List[Dict] = []
        cursor = start_ts

        while cursor < end_ts:
            chunk_end = min(cursor + chunk_span, end_ts)
            params = {
                "symbol": symbol,
                "resolution": resolution,
                "start": cursor,
                "end": chunk_end,
            }
            r = requests.get(self.BASE_URL, params=params, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"}, timeout=30)
            r.raise_for_status()
            payload = r.json()
            if not payload.get("success"):
                raise RuntimeError(f"Delta API returned failure: {payload}")
            rows = payload.get("result", [])
            out.extend(rows)
            if not rows:
                break
            last_time = int(rows[-1]["time"])
            cursor = last_time + step_seconds
            time.sleep(0.15)

        if not out:
            raise RuntimeError("No candles returned. Check symbol/resolution/date range.")

        df = pd.DataFrame(out)
        df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["timestamp", "open", "high", "low", "close", "volume"]]


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_trend"] = df["close"].ewm(span=200, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ADX
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    df["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()
    df["atr"] = atr

    df["vol_ma20"] = df["volume"].rolling(20).mean()
    return df


def strong_bullish_signal(row: pd.Series) -> bool:
    return bool(
        row["close"] > row["ema_trend"] and
        row["ema_fast"] > row["ema_slow"] and
        row["rsi"] >= 58 and
        row["macd_hist"] > 0 and
        row["adx"] >= 20 and
        row["volume"] >= row["vol_ma20"]
    )


def strong_bearish_signal(row: pd.Series) -> bool:
    return bool(
        row["close"] < row["ema_trend"] and
        row["ema_fast"] < row["ema_slow"] and
        row["rsi"] <= 42 and
        row["macd_hist"] < 0 and
        row["adx"] >= 20 and
        row["volume"] >= row["vol_ma20"]
    )


def notional_usd(entry_price: float, lot_size: float, lots: int) -> float:
    return entry_price * lot_size * lots


def margin_required_usd(entry_price: float, lot_size: float, lots: int, leverage: float) -> float:
    return notional_usd(entry_price, lot_size, lots) / leverage


def fee_usd(notional_value_usd: float, fee_rate: float, gst_on_fees: float) -> float:
    base = notional_value_usd * fee_rate
    return base * (1 + gst_on_fees)


def funding_cost_usd(entry_price: float, exit_price: float, lot_size: float, lots: int, candles_held: int, rate_per_8h: float) -> float:
    if rate_per_8h == 0:
        return 0.0
    avg_notional = ((entry_price + exit_price) / 2.0) * lot_size * lots
    intervals = candles_held / 8.0  # for 1h candles
    return avg_notional * rate_per_8h * intervals


def backtest(df: pd.DataFrame, cfg: BacktestConfig, usd_inr: float = 83.0) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    df = compute_indicators(df)

    capital_inr = cfg.initial_capital_inr
    available_inr = capital_inr
    trades: List[Dict] = []
    equity_curve: List[Dict] = []

    in_position = False
    side: Optional[str] = None
    pending_side: Optional[str] = None
    entry_price = 0.0
    entry_time = None
    current_margin_usd = 0.0
    next_target_usd = 0.0
    current_stop_usd = 0.0
    sequence_id = 0
    ladder_step = 0
    entry_fee_usd = 0.0
    entry_slipped_price = 0.0

    for i in range(220, len(df)):
        row = df.iloc[i]

        if not in_position:
            if pending_side is not None:
                side = pending_side
                pending_side = None
                raw_entry = float(row["open"])
                slipped_entry = raw_entry * (1 + cfg.slippage_rate if side == "long" else 1 - cfg.slippage_rate)
                margin_usd = margin_required_usd(slipped_entry, cfg.lot_size, cfg.lots, cfg.leverage)
                margin_inr = margin_usd * usd_inr
                max_margin_allowed = capital_inr * cfg.risk_guard_max_margin_fraction
                if margin_inr > max_margin_allowed:
                    equity_curve.append({"timestamp": row["timestamp"], "equity_inr": capital_inr})
                    continue

                entry_fee_usd = fee_usd(notional_usd(slipped_entry, cfg.lot_size, cfg.lots), cfg.taker_fee_rate, cfg.gst_on_fees)
                entry_time = row["timestamp"]
                entry_price = raw_entry
                entry_slipped_price = slipped_entry
                current_margin_usd = margin_usd
                next_target_usd = current_margin_usd * (1 + cfg.target_return_on_margin)
                current_stop_usd = current_margin_usd * (1 - cfg.stop_return_on_margin)
                sequence_id += 1
                ladder_step = 0
                in_position = True

                equity_curve.append({"timestamp": row["timestamp"], "equity_inr": capital_inr})
                continue

            long_sig = strong_bullish_signal(row)
            short_sig = strong_bearish_signal(row)
            if long_sig != short_sig:
                pending_side = "long" if long_sig else "short"

            equity_curve.append({"timestamp": row["timestamp"], "equity_inr": capital_inr})
            continue

        # evaluate exit/laddering inside the current candle using high/low extremes
        candles_held = max(1, int((row["timestamp"] - entry_time) / pd.Timedelta(hours=1)))
        raw_high = float(row["high"])
        raw_low = float(row["low"])

        high_exec = raw_high * (1 - cfg.slippage_rate)
        low_exec = raw_low * (1 + cfg.slippage_rate)

        if side == "long":
            best_unrealized_margin = current_margin_usd * (1 + cfg.leverage * ((high_exec - entry_slipped_price) / entry_slipped_price))
            worst_unrealized_margin = current_margin_usd * (1 + cfg.leverage * ((low_exec - entry_slipped_price) / entry_slipped_price))
        else:
            best_unrealized_margin = current_margin_usd * (1 + cfg.leverage * ((entry_slipped_price - low_exec) / entry_slipped_price))
            worst_unrealized_margin = current_margin_usd * (1 + cfg.leverage * ((entry_slipped_price - high_exec) / entry_slipped_price))

        # Conservative assumption: if both threshold and stop hit in the same candle, stop wins.
        if worst_unrealized_margin <= current_stop_usd:
            # derive exit price corresponding to current stop threshold
            required_return = (current_stop_usd / current_margin_usd) - 1.0
            price_return = required_return / cfg.leverage
            if side == "long":
                stop_price = entry_slipped_price * (1 + price_return)
                exit_price = stop_price * (1 - cfg.slippage_rate)
            else:
                stop_price = entry_slipped_price * (1 - price_return)
                exit_price = stop_price * (1 + cfg.slippage_rate)

            exit_notional = notional_usd(exit_price, cfg.lot_size, cfg.lots)
            exit_fee_usd = fee_usd(exit_notional, cfg.taker_fee_rate, cfg.gst_on_fees)
            gross_pnl_usd = current_stop_usd - current_margin_usd
            funding_usd = funding_cost_usd(entry_slipped_price, exit_price, cfg.lot_size, cfg.lots, candles_held, cfg.funding_rate_per_8h)
            net_pnl_usd = gross_pnl_usd - entry_fee_usd - exit_fee_usd - funding_usd
            capital_inr += net_pnl_usd * usd_inr

            trades.append({
                "sequence_id": sequence_id,
                "ladder_steps_hit": ladder_step,
                "side": side,
                "entry_time": entry_time,
                "exit_time": row["timestamp"],
                "entry_price": entry_slipped_price,
                "exit_price": exit_price,
                "entry_margin_usd": current_margin_usd,
                "stop_margin_usd": current_stop_usd,
                "gross_pnl_usd": gross_pnl_usd,
                "net_pnl_usd": net_pnl_usd,
                "net_pnl_inr": net_pnl_usd * usd_inr,
                "capital_after_inr": capital_inr,
                "reason": "stop_loss"
            })

            in_position = False
            side = None
            pending_side = None
            entry_price = 0.0
            current_margin_usd = 0.0
            next_target_usd = 0.0
            current_stop_usd = 0.0
            ladder_step = 0
            equity_curve.append({"timestamp": row["timestamp"], "equity_inr": capital_inr})
            continue

        if best_unrealized_margin >= next_target_usd:
            # ratchet the tracked trade amount upward without closing the position
            current_margin_usd = next_target_usd
            next_target_usd = current_margin_usd * (1 + cfg.target_return_on_margin)
            current_stop_usd = current_margin_usd * (1 - cfg.stop_return_on_margin)
            ladder_step += 1

        equity_curve.append({"timestamp": row["timestamp"], "equity_inr": capital_inr})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    summary = summarize_results(trades_df, equity_df, cfg)
    return trades_df, equity_df, summary


def summarize_results(trades_df: pd.DataFrame, equity_df: pd.DataFrame, cfg: BacktestConfig) -> Dict:
    if equity_df.empty:
        return {"error": "No equity data generated"}

    final_equity = float(equity_df["equity_inr"].iloc[-1])
    total_return = (final_equity / cfg.initial_capital_inr) - 1.0
    running_max = equity_df["equity_inr"].cummax()
    dd = (equity_df["equity_inr"] / running_max) - 1.0
    max_drawdown = float(dd.min()) if not dd.empty else 0.0

    if trades_df.empty:
        return {
            "initial_capital_inr": cfg.initial_capital_inr,
            "final_equity_inr": final_equity,
            "total_return_pct": total_return * 100,
            "max_drawdown_pct": max_drawdown * 100,
            "trade_sequences": 0,
        }

    wins = (trades_df["net_pnl_inr"] > 0).sum()
    losses = (trades_df["net_pnl_inr"] <= 0).sum()
    avg_win = float(trades_df.loc[trades_df["net_pnl_inr"] > 0, "net_pnl_inr"].mean()) if wins else 0.0
    avg_loss = float(trades_df.loc[trades_df["net_pnl_inr"] <= 0, "net_pnl_inr"].mean()) if losses else 0.0
    profit_factor = (
        abs(trades_df.loc[trades_df["net_pnl_inr"] > 0, "net_pnl_inr"].sum() /
            trades_df.loc[trades_df["net_pnl_inr"] <= 0, "net_pnl_inr"].sum())
        if losses and trades_df.loc[trades_df["net_pnl_inr"] <= 0, "net_pnl_inr"].sum() != 0 else np.inf
    )

    return {
        "initial_capital_inr": cfg.initial_capital_inr,
        "final_equity_inr": round(final_equity, 2),
        "total_return_pct": round(total_return * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "trade_sequences": int(len(trades_df)),
        "win_rate_pct": round(float(wins / len(trades_df) * 100), 2),
        "avg_win_inr": round(avg_win, 2),
        "avg_loss_inr": round(avg_loss, 2),
        "profit_factor": round(float(profit_factor), 3) if np.isfinite(profit_factor) else "inf",
        "avg_ladder_steps": round(float(trades_df["ladder_steps_hit"].mean()), 2),
    }


def main():
    cfg = BacktestConfig(
        symbol="BTCUSD",
        resolution="1h",
        start="2024-01-01",
        end="2026-01-01",
        initial_capital_inr=10000,
        leverage=5,
        lot_size=0.001,
        lots=1,
        funding_rate_per_8h=0.0,
    )

    client = DeltaHistoricalClient()
    df = client.fetch_candles(cfg.symbol, cfg.resolution, cfg.start, cfg.end)
    trades_df, equity_df, summary = backtest(df, cfg)

    trades_path = "/mnt/data/delta_trades.csv"
    equity_path = "/mnt/data/delta_equity.csv"
    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)

    print("Summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\\nSaved trades to: {trades_path}")
    print(f"Saved equity curve to: {equity_path}")


if __name__ == "__main__":
    main()
