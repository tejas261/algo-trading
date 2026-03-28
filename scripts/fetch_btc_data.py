"""Fetch BTC/USDT 1h OHLCV data from the Binance public API and save to CSV.

Usage:
    python scripts/fetch_btc_data.py
    python scripts/fetch_btc_data.py --start 2025-01-01 --end 2025-03-28
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "4h"
MAX_LIMIT = 1000  # Binance caps at 1000 per request
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "btc_4h.csv"


def _fetch_klines(
    start_ms: int,
    end_ms: int,
    limit: int = MAX_LIMIT,
) -> list[list]:
    """Fetch a single batch of klines from Binance."""
    params = (
        f"symbol={SYMBOL}&interval={INTERVAL}"
        f"&startTime={start_ms}&endTime={end_ms}&limit={limit}"
    )
    url = f"{BINANCE_KLINES_URL}?{params}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def fetch_all_klines(start_dt: datetime, end_dt: datetime) -> list[list]:
    """Page through Binance klines API to collect all candles in the range."""
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    all_klines: list[list] = []
    cursor_ms = start_ms

    while cursor_ms < end_ms:
        batch = _fetch_klines(cursor_ms, end_ms, limit=MAX_LIMIT)
        if not batch:
            break
        all_klines.extend(batch)
        # Move cursor past the last candle's open time
        last_open_ms = batch[-1][0]
        cursor_ms = last_open_ms + 1
        if len(batch) < MAX_LIMIT:
            break
        # Be polite to the public endpoint
        time.sleep(0.25)

    return all_klines


def klines_to_dataframe(klines: list[list]) -> pd.DataFrame:
    """Convert raw Binance kline arrays to a clean OHLCV DataFrame."""
    # Binance kline format: [open_time, open, high, low, close, volume, ...]
    rows = []
    for k in klines:
        rows.append(
            {
                "timestamp": datetime.fromtimestamp(
                    k[0] / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
        )
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch BTC/USDT 1h OHLCV data from Binance"
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date in YYYY-MM-DD format (default: ~3 months ago)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format (default: now)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    now = datetime.now(tz=timezone.utc)
    end_dt = (
        datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end
        else now
    )
    start_dt = (
        datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.start
        else end_dt - timedelta(days=730)
    )

    print(f"Fetching BTC/USDT {INTERVAL} candles from {start_dt.date()} to {end_dt.date()} ...")
    klines = fetch_all_klines(start_dt, end_dt)

    if not klines:
        print("No data returned from Binance. Check your date range.")
        return

    df = klines_to_dataframe(klines)

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # Summary
    print(f"\nSaved {len(df)} rows to {OUTPUT_PATH}")
    print(f"Date range : {df['timestamp'].iloc[0]}  ->  {df['timestamp'].iloc[-1]}")
    print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f} USDT")
    print(f"Volume     : {df['volume'].sum():,.0f} BTC total")


if __name__ == "__main__":
    main()
