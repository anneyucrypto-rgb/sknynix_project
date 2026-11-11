"""Download raw one-minute and event-level SK hynix research data."""

from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

import pandas as pd

from massive_client import MassiveAPIError, MassiveClient, write_parquet


PROJECT_DIR: Final = Path(__file__).resolve().parents[1]
INGESTION_DIR: Final = Path(__file__).resolve().parent
RAW_DIR: Final = PROJECT_DIR / "data" / "raw"
SECRET_FILE: Final = INGESTION_DIR / ".env"
NY_TZ: Final = ZoneInfo("America/New_York")
UTC_TZ: Final = ZoneInfo("UTC")

BAR_DATASETS: Final = (
    ("SKHY", RAW_DIR / "us" / "SKHY_1min.parquet", True),
    ("SPY", RAW_DIR / "us" / "SPY_1min.parquet", True),
    ("SOXX", RAW_DIR / "us" / "SOXX_1min.parquet", True),
    ("QQQ", RAW_DIR / "us" / "QQQ_1min.parquet", True),
    ("C:USDKRW", RAW_DIR / "fx" / "USDKRW_1min.parquet", True),
)
EVENT_ANCHORS: Final = (
    ("premarket_0400", time(4, 0)),
    ("regular_0930", time(9, 30)),
)


def subtract_months(value: date, months: int) -> date:
    """Subtract whole calendar months while clamping the day when necessary."""
    month_index = value.year * 12 + value.month - 1 - months
    year, month_zero_based = divmod(month_index, 12)
    month = month_zero_based + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def load_secret(path: Path = SECRET_FILE) -> str:
    """Load MASSIVE_API_KEY from the environment or ingestion/.env."""
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
    raise RuntimeError(f"MASSIVE_API_KEY is missing from the environment and {path}")


def download_korean_daily(start: date, end: date, destination: Path) -> int:
    """Download raw daily 000660.KS OHLCV observations."""
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
    write_parquet(
        frame[
            [
                "ticker",
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjusted_close",
                "volume",
            ]
        ],
        destination,
    )
    return len(frame)


def _event_schedule(skhy_bars: pd.DataFrame) -> list[dict[str, Any]]:
    timestamps = pd.to_datetime(skhy_bars["timestamp"], utc=True)
    trading_dates = sorted(set(timestamps.dt.tz_convert(NY_TZ).dt.date))
    schedule: list[dict[str, Any]] = []
    for trading_date in trading_dates:
        session_end_local = datetime.combine(trading_date, time(20, 0), NY_TZ)
        for anchor_type, anchor_time in EVENT_ANCHORS:
            scheduled_local = datetime.combine(trading_date, anchor_time, NY_TZ)
            schedule.append(
                {
                    "event_date_ny": trading_date,
                    "anchor_type": anchor_type,
                    "scheduled_timestamp_utc": pd.Timestamp(
                        scheduled_local.astimezone(UTC_TZ)
                    ),
                    "interval_end_utc": pd.Timestamp(
                        session_end_local.astimezone(UTC_TZ)
                    ),
                }
            )
    return schedule


def _is_entitlement_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "403" in message or "not entitled" in message or "upgrade your plan" in message


def _trade_row(event: dict[str, Any], trade: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_date_ny": event["event_date_ny"],
        "anchor_type": event["anchor_type"],
        "scheduled_timestamp_utc": event["scheduled_timestamp_utc"],
        "t_first_trade": pd.to_datetime(trade.get("sip_timestamp"), unit="ns", utc=True),
        "participant_timestamp": trade.get("participant_timestamp"),
        "sequence_number": trade.get("sequence_number"),
        "trade_price": trade.get("price"),
        "trade_size": trade.get("size"),
        "trade_exchange": trade.get("exchange"),
        "trade_conditions": json.dumps(trade.get("conditions") or [], separators=(",", ":")),
        "tape": trade.get("tape"),
    }


def _quote_row(event: dict[str, Any], quote: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_date_ny": event["event_date_ny"],
        "anchor_type": event["anchor_type"],
        "scheduled_timestamp_utc": event["scheduled_timestamp_utc"],
        "t_first_quote": pd.to_datetime(quote.get("sip_timestamp"), unit="ns", utc=True),
        "participant_timestamp": quote.get("participant_timestamp"),
        "sequence_number": quote.get("sequence_number"),
        "bid_price": quote.get("bid_price"),
        "bid_size": quote.get("bid_size"),
        "bid_exchange": quote.get("bid_exchange"),
        "ask_price": quote.get("ask_price"),
        "ask_size": quote.get("ask_size"),
        "ask_exchange": quote.get("ask_exchange"),
        "quote_conditions": json.dumps(quote.get("conditions") or [], separators=(",", ":")),
        "tape": quote.get("tape"),
    }


def _empty_event_frame(kind: str) -> pd.DataFrame:
    common: dict[str, pd.Series] = {
        "event_date_ny": pd.Series(dtype="datetime64[ns]"),
        "anchor_type": pd.Series(dtype="string"),
        "scheduled_timestamp_utc": pd.Series(dtype="datetime64[ns, UTC]"),
    }
    if kind == "trade":
        common.update(
            {
                "t_first_trade": pd.Series(dtype="datetime64[ns, UTC]"),
                "participant_timestamp": pd.Series(dtype="Int64"),
                "sequence_number": pd.Series(dtype="Int64"),
                "trade_price": pd.Series(dtype="Float64"),
                "trade_size": pd.Series(dtype="Float64"),
                "trade_exchange": pd.Series(dtype="Int64"),
                "trade_conditions": pd.Series(dtype="string"),
                "tape": pd.Series(dtype="Int64"),
            }
        )
    else:
        common.update(
            {
                "t_first_quote": pd.Series(dtype="datetime64[ns, UTC]"),
                "participant_timestamp": pd.Series(dtype="Int64"),
                "sequence_number": pd.Series(dtype="Int64"),
                "bid_price": pd.Series(dtype="Float64"),
                "bid_size": pd.Series(dtype="Float64"),
                "bid_exchange": pd.Series(dtype="Int64"),
                "ask_price": pd.Series(dtype="Float64"),
                "ask_size": pd.Series(dtype="Float64"),
                "ask_exchange": pd.Series(dtype="Int64"),
                "quote_conditions": pd.Series(dtype="string"),
                "tape": pd.Series(dtype="Int64"),
            }
        )
    return pd.DataFrame(common)


def _download_first_events(
    client: MassiveClient,
    schedule: list[dict[str, Any]],
    *,
    kind: str,
    destination: Path,
) -> tuple[int, str | None]:
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(schedule, start=1):
        start_ns = int(event["scheduled_timestamp_utc"].value)
        end_ns = int(event["interval_end_utc"].value)
        try:
            if kind == "trade":
                record = client.get_first_trade("SKHY", start_ns, end_ns)
                if record is not None:
                    rows.append(_trade_row(event, record))
            else:
                record = client.get_first_valid_quote("SKHY", start_ns, end_ns)
                if record is not None:
                    rows.append(_quote_row(event, record))
        except MassiveAPIError as exc:
            if _is_entitlement_error(exc):
                write_parquet(_empty_event_frame(kind), destination)
                return 0, str(exc)
            raise
        if index % 25 == 0:
            print(f"    {kind}: checked {index:,}/{len(schedule):,} events")

    frame = pd.DataFrame.from_records(rows) if rows else _empty_event_frame(kind)
    if not frame.empty:
        frame["event_date_ny"] = pd.to_datetime(frame["event_date_ny"])
        for column in ("scheduled_timestamp_utc", f"t_first_{kind}"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
    write_parquet(frame, destination)
    return len(frame), None


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
    parser.add_argument("--skip-trades", action="store_true")
    parser.add_argument("--skip-quotes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = args.start_date or subtract_months(args.end_date, 1)
    end = args.end_date
    if start > end:
        raise SystemExit("start-date must not be later than end-date")

    print(f"Downloading raw inclusive window {start} through {end}")
    client = MassiveClient(load_secret())
    failures: list[str] = []
    skhy_bars: pd.DataFrame | None = None

    for ticker, target, include_volume in BAR_DATASETS:
        label = ticker.removeprefix("C:")
        try:
            frame = client.get_aggregates(
                ticker,
                start.isoformat(),
                end.isoformat(),
                multiplier=1,
                timespan="minute",
                include_volume=include_volume,
            )
            if frame.empty:
                raise RuntimeError("no one-minute aggregate rows returned")
            write_parquet(frame, target)
            if ticker == "SKHY":
                skhy_bars = frame
            print(f"OK  {label:7} {len(frame):>10,} one-minute rows -> {target}")
        except Exception as exc:
            failures.append(f"{label}: {exc}")
            print(f"ERR {label:7} {exc}", file=sys.stderr)

    korea_target = RAW_DIR / "korea" / "000660_daily.parquet"
    try:
        count = download_korean_daily(start, end, korea_target)
        print(f"OK  000660  {count:>10,} daily rows -> {korea_target}")
    except Exception as exc:
        failures.append(f"000660: {exc}")
        print(f"ERR 000660  {exc}", file=sys.stderr)

    if skhy_bars is not None:
        schedule = _event_schedule(skhy_bars)
        if not args.skip_trades:
            target = RAW_DIR / "us" / "SKHY_first_trades.parquet"
            count, warning = _download_first_events(
                client, schedule, kind="trade", destination=target
            )
            print(f"OK  trades   {count:>10,} event rows -> {target}")
            if warning:
                print(f"WARN trades   {warning}", file=sys.stderr)
        if not args.skip_quotes:
            target = RAW_DIR / "us" / "SKHY_first_quotes.parquet"
            count, warning = _download_first_events(
                client, schedule, kind="quote", destination=target
            )
            print(f"OK  quotes   {count:>10,} event rows -> {target}")
            if warning:
                print(f"WARN quotes   {warning}", file=sys.stderr)

    if failures:
        print("\nCompleted with failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Raw data download completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
