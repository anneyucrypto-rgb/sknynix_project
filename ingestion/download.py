"""Download the most recent month of five-minute SK hynix research data."""

from __future__ import annotations

import argparse
import calendar
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from massive_client import MassiveAPIError, MassiveClient, write_parquet


PROJECT_DIR = Path(__file__).resolve().parents[1]
INGESTION_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
SECRET_FILE = INGESTION_DIR / ".env"
DATASETS = (
    ("SKHY", DATA_DIR / "us" / "SKHY_5min.parquet", True),
    ("SPY", DATA_DIR / "us" / "SPY_5min.parquet", False),
    ("SOXX", DATA_DIR / "us" / "SOXX_5min.parquet", False),
    ("QQQ", DATA_DIR / "us" / "QQQ_5min.parquet", False),
    ("C:USDKRW", DATA_DIR / "fx" / "USDKRW_5min.parquet", False),
)


def subtract_months(value: date, months: int) -> date:
    """Subtract whole calendar months while clamping to the target month's last day."""
    month_index = value.year * 12 + value.month - 1 - months
    year, month_zero_based = divmod(month_index, 12)
    month = month_zero_based + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def load_secret(path: Path = SECRET_FILE) -> str:
    """Load MASSIVE_API_KEY from the process environment or a local .env file."""
    key = os.getenv("MASSIVE_API_KEY", "").strip()
    if key:
        return key
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            if name.strip() == "MASSIVE_API_KEY":
                return value.strip().strip("\"'")
    raise RuntimeError(
        f"MASSIVE_API_KEY is missing. Add it to the environment or {path}."
    )


def download_korean_daily(start: date, end: date, destination: Path) -> int:
    """Download unadjusted daily SK hynix (000660.KS) prices from Yahoo Finance."""
    import yfinance as yf

    raw = yf.download(
        "000660.KS",
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise RuntimeError("the Korean-market provider returned no rows")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    frame = raw.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjusted_close",
            "Volume": "volume",
        }
    )
    frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
    frame.insert(0, "ticker", "000660.KS")
    output_columns = [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    ]
    write_parquet(frame[output_columns], destination)
    return len(frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=date.today(),
        help="Inclusive end date (YYYY-MM-DD); defaults to today.",
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        help="Inclusive start date; defaults to one calendar month before end-date.",
    )
    parser.add_argument(
        "--skip-quotes",
        action="store_true",
        help="Skip the potentially very large tick-level SKHY quote download.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = args.start_date or subtract_months(args.end_date, 1)
    end = args.end_date
    if start > end:
        raise SystemExit("start-date must not be later than end-date")

    print(f"Downloading inclusive window {start} through {end}")
    client = MassiveClient(load_secret())
    failures: list[str] = []

    for ticker, target, include_volume in DATASETS:
        label = ticker.removeprefix("C:")
        try:
            frame = client.get_aggregates(
                ticker,
                start.isoformat(),
                end.isoformat(),
                multiplier=5,
                timespan="minute",
                include_volume=include_volume,
            )
            if frame.empty:
                raise RuntimeError("no aggregate rows returned")
            write_parquet(frame, target)
            content = "price+volume" if include_volume else "price"
            print(f"OK  {label:7} {len(frame):>10,} rows ({content}) -> {target}")
        except Exception as exc:
            failures.append(f"{label}: {exc}")
            print(f"ERR {label:7} {exc}", file=sys.stderr)

    korea_target = DATA_DIR / "korea" / "000660_daily.parquet"
    try:
        count = download_korean_daily(start, end, korea_target)
        print(f"OK  000660  {count:>10,} rows (daily price+volume) -> {korea_target}")
    except Exception as exc:
        failures.append(f"000660: {exc}")
        print(f"ERR 000660  {exc}", file=sys.stderr)

    if not args.skip_quotes:
        quotes_target = DATA_DIR / "quotes" / "SKHY_quotes.parquet"
        try:
            count = client.download_quotes_parquet(
                "SKHY", start.isoformat(), end.isoformat(), quotes_target
            )
            if count == 0:
                raise RuntimeError("no quote rows returned")
            print(f"OK  quotes   {count:>10,} rows -> {quotes_target}")
        except (MassiveAPIError, RuntimeError, OSError) as exc:
            failures.append(f"SKHY quotes: {exc}")
            print(f"ERR quotes   {exc}", file=sys.stderr)

    if failures:
        print("\nCompleted with failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("All requested datasets downloaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
