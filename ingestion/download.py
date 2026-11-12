"""Download raw data for configured Korean/U.S. price-discovery pairs."""

from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from massive_client import MassiveAPIError, MassiveClient, write_parquet


PROJECT_DIR: Final = Path(__file__).resolve().parents[1]
INGESTION_DIR: Final = Path(__file__).resolve().parent
RAW_DIR: Final = PROJECT_DIR / "data" / "raw"
SECRET_FILE: Final = INGESTION_DIR / ".env"
NY_TZ: Final = ZoneInfo("America/New_York")
UTC_TZ: Final = ZoneInfo("UTC")
BAR_MINUTES: Final = 5

CONTROL_BAR_DATASETS: Final = (
    ("SPY", RAW_DIR / "us" / "SPY_5min.parquet", True),
    ("SOXX", RAW_DIR / "us" / "SOXX_5min.parquet", True),
    ("QQQ", RAW_DIR / "us" / "QQQ_5min.parquet", True),
    ("C:USDKRW", RAW_DIR / "fx" / "USDKRW_5min.parquet", True),
)
COMMON_DAILY_CONTROLS: Final = (
    ("^KS11", RAW_DIR / "macro" / "KOSPI_daily.parquet"),
    ("^VIX", RAW_DIR / "macro" / "VIX_daily.parquet"),
)
FRED_YIELD_SERIES: Final = ("DGS2", "DGS10")
EVENT_ANCHORS: Final = (
    ("premarket_0400", time(4, 0)),
    ("regular_0930", time(9, 30)),
)


@dataclass(frozen=True)
class ResearchPair:
    company: str
    korea_ticker: str
    us_ticker: str
    us_exchange: str
    industry: str

    @property
    def korea_file_stem(self) -> str:
        return self.korea_ticker.split(".", 1)[0]


RESEARCH_PAIRS: Final = (
    ResearchPair("SK hynix", "000660.KS", "SKHY", "Nasdaq", "Semiconductors"),
    ResearchPair("LG Display", "034220.KS", "LPL", "NYSE", "Displays"),
    ResearchPair("SK Telecom", "017670.KS", "SKM", "NYSE", "Telecommunications"),
    ResearchPair("KT Corporation", "030200.KS", "KT", "NYSE", "Telecommunications"),
    ResearchPair("POSCO Holdings", "005490.KS", "PKX", "NYSE", "Steel, materials"),
    ResearchPair("KB Financial", "105560.KS", "KB", "NYSE", "Banking"),
    ResearchPair("Shinhan Financial", "055550.KS", "SHG", "NYSE", "Banking"),
    ResearchPair("Woori Financial", "316140.KS", "WF", "NYSE", "Banking"),
    ResearchPair("Korea Electric Power", "015760.KS", "KEP", "NYSE", "Utilities"),
)
PAIR_BY_US_TICKER: Final = {pair.us_ticker: pair for pair in RESEARCH_PAIRS}


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


def download_korean_daily(
    ticker: str, start: date, end: date, destination: Path
) -> int:
    """Download raw daily Korean OHLCV observations."""
    import yfinance as yf

    raw = yf.download(
        ticker,
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
    frame.insert(0, "ticker", ticker)
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


def download_us_treasury_yields(
    start: date, end: date, destination: Path
) -> int:
    """Download official daily U.S. 2Y/10Y Treasury yields from FRED."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(
        FRED_YIELD_SERIES
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text)).rename(
        columns={
            "observation_date": "date",
            "DGS2": "us_2y_yield",
            "DGS10": "us_10y_yield",
        }
    )
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("us_2y_yield", "us_10y_yield"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        (frame["date"].dt.date >= start) & (frame["date"].dt.date <= end)
    ].dropna(how="all", subset=["us_2y_yield", "us_10y_yield"])
    if frame.empty:
        raise RuntimeError("FRED returned no U.S. Treasury yield observations")
    write_parquet(frame, destination)
    return len(frame)


def _event_schedule(us_bars: pd.DataFrame) -> list[dict[str, Any]]:
    timestamps = pd.to_datetime(us_bars["timestamp"], utc=True)
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


def _trade_row(
    ticker: str, event: dict[str, Any], trade: dict[str, Any]
) -> dict[str, Any]:
    return {
        "ticker": ticker,
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


def _quote_row(
    ticker: str, event: dict[str, Any], quote: dict[str, Any]
) -> dict[str, Any]:
    return {
        "ticker": ticker,
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
        "ticker": pd.Series(dtype="string"),
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
    ticker: str,
    kind: str,
    destination: Path,
) -> tuple[int, str | None]:
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(schedule, start=1):
        start_ns = int(event["scheduled_timestamp_utc"].value)
        end_ns = int(event["interval_end_utc"].value)
        try:
            if kind == "trade":
                record = client.get_first_trade(ticker, start_ns, end_ns)
                if record is not None:
                    rows.append(_trade_row(ticker, event, record))
            else:
                record = client.get_first_valid_quote(ticker, start_ns, end_ns)
                if record is not None:
                    rows.append(_quote_row(ticker, event, record))
        except MassiveAPIError as exc:
            if _is_entitlement_error(exc):
                write_parquet(_empty_event_frame(kind), destination)
                return 0, str(exc)
            raise
        if index % 25 == 0:
            print(
                f"    {ticker} {kind}: checked {index:,}/{len(schedule):,} events"
            )

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
    parser.add_argument(
        "--pairs",
        nargs="+",
        type=str.upper,
        choices=[*PAIR_BY_US_TICKER, "ALL"],
        default=["SKHY"],
        metavar="US_TICKER",
        help=(
            "U.S. pair ticker(s) to download. Default: SKHY. Use ALL for all "
            "configured Korean/U.S. research pairs."
        ),
    )
    parser.add_argument(
        "--skip-controls",
        action="store_true",
        help=(
            "Skip common SPY, SOXX, QQQ, USD/KRW, KOSPI, VIX, and U.S. "
            "Treasury-control downloads."
        ),
    )
    parser.add_argument("--skip-trades", action="store_true")
    parser.add_argument("--skip-quotes", action="store_true")
    parser.add_argument(
        "--full-ticks",
        action="store_true",
        help=(
            "Also stream all selected U.S. trades/quotes for post-anchor "
            "microstructure windows."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = args.start_date or subtract_months(args.end_date, 1)
    end = args.end_date
    if start > end:
        raise SystemExit("start-date must not be later than end-date")

    selected_pairs = (
        list(RESEARCH_PAIRS)
        if "ALL" in args.pairs
        else [PAIR_BY_US_TICKER[ticker] for ticker in dict.fromkeys(args.pairs)]
    )
    selected_tickers = {pair.us_ticker for pair in selected_pairs}

    print(f"Downloading raw inclusive window {start} through {end}")
    print("Research pairs: " + ", ".join(sorted(selected_tickers)))
    print(
        "Common controls: "
        + (
            "skipped"
            if args.skip_controls
            else "SPY, SOXX, QQQ, USD/KRW, KOSPI, VIX, U.S. 2Y/10Y"
        )
    )
    client = MassiveClient(load_secret())
    failures: list[str] = []
    pair_bars: dict[str, pd.DataFrame] = {}

    pair_bar_datasets = [
        (
            pair.us_ticker,
            RAW_DIR / "us" / f"{pair.us_ticker}_5min.parquet",
            True,
        )
        for pair in selected_pairs
    ]

    bar_datasets = [
        *pair_bar_datasets,
        *([] if args.skip_controls else CONTROL_BAR_DATASETS),
    ]
    for ticker, target, include_volume in bar_datasets:
        label = ticker.removeprefix("C:")
        try:
            frame = client.get_aggregates(
                ticker,
                start.isoformat(),
                end.isoformat(),
                multiplier=BAR_MINUTES,
                timespan="minute",
                include_volume=include_volume,
            )
            if frame.empty:
                raise RuntimeError("no five-minute aggregate rows returned")
            write_parquet(frame, target)
            if ticker in selected_tickers:
                pair_bars[ticker] = frame
            print(f"OK  {label:7} {len(frame):>10,} five-minute rows -> {target}")
        except Exception as exc:
            failures.append(f"{label}: {exc}")
            print(f"ERR {label:7} {exc}", file=sys.stderr)

    if not args.skip_controls:
        macro_start = start - timedelta(days=10)
        for ticker, target in COMMON_DAILY_CONTROLS:
            try:
                count = download_korean_daily(ticker, macro_start, end, target)
                print(
                    f"OK  {ticker:7} {count:>10,} daily control rows -> {target}"
                )
            except Exception as exc:
                failures.append(f"{ticker}: {exc}")
                print(f"ERR {ticker:7} {exc}", file=sys.stderr)

        treasury_target = RAW_DIR / "macro" / "US_TREASURY_daily.parquet"
        try:
            count = download_us_treasury_yields(
                macro_start, end, treasury_target
            )
            print(
                f"OK  FRED    {count:>10,} U.S. yield rows -> {treasury_target}"
            )
        except Exception as exc:
            failures.append(f"FRED yields: {exc}")
            print(f"ERR FRED yields  {exc}", file=sys.stderr)

    for pair in selected_pairs:
        korea_target = (
            RAW_DIR / "korea" / f"{pair.korea_file_stem}_daily.parquet"
        )
        try:
            count = download_korean_daily(
                pair.korea_ticker, start, end, korea_target
            )
            print(
                f"OK  {pair.korea_file_stem:7} {count:>10,} daily rows -> "
                f"{korea_target}"
            )
        except Exception as exc:
            failures.append(f"{pair.korea_ticker}: {exc}")
            print(f"ERR {pair.korea_ticker}  {exc}", file=sys.stderr)

    for pair in selected_pairs:
        us_ticker = pair.us_ticker
        us_bars = pair_bars.get(us_ticker)
        if us_bars is None:
            continue
        schedule = _event_schedule(us_bars)
        if not args.skip_trades:
            target = RAW_DIR / "us" / f"{us_ticker}_first_trades.parquet"
            count, warning = _download_first_events(
                client,
                schedule,
                ticker=us_ticker,
                kind="trade",
                destination=target,
            )
            print(
                f"OK  {us_ticker:7} {count:>10,} first-trade rows -> {target}"
            )
            if warning:
                print(f"WARN {us_ticker} trades   {warning}", file=sys.stderr)
        if not args.skip_quotes:
            target = RAW_DIR / "us" / f"{us_ticker}_first_quotes.parquet"
            count, warning = _download_first_events(
                client,
                schedule,
                ticker=us_ticker,
                kind="quote",
                destination=target,
            )
            print(
                f"OK  {us_ticker:7} {count:>10,} first-quote rows -> {target}"
            )
            if warning:
                print(f"WARN {us_ticker} quotes   {warning}", file=sys.stderr)

        if args.full_ticks and not args.skip_trades:
            target = RAW_DIR / "us" / f"{us_ticker}_trades.parquet"
            try:
                count = client.download_trades_parquet(
                    us_ticker, start.isoformat(), end.isoformat(), target
                )
                print(
                    f"OK  {us_ticker:7} {count:>10,} full trade rows -> {target}"
                )
            except MassiveAPIError as exc:
                if _is_entitlement_error(exc):
                    print(
                        f"WARN {us_ticker} full trades   {exc}", file=sys.stderr
                    )
                else:
                    failures.append(f"{us_ticker} full trades: {exc}")
                    print(f"ERR {us_ticker} full trades   {exc}", file=sys.stderr)

        if args.full_ticks and not args.skip_quotes:
            target = RAW_DIR / "us" / f"{us_ticker}_quotes.parquet"
            try:
                count = client.download_quotes_parquet(
                    us_ticker, start.isoformat(), end.isoformat(), target
                )
                print(
                    f"OK  {us_ticker:7} {count:>10,} full quote rows -> {target}"
                )
            except MassiveAPIError as exc:
                if _is_entitlement_error(exc):
                    print(
                        f"WARN {us_ticker} full quotes   {exc}", file=sys.stderr
                    )
                else:
                    failures.append(f"{us_ticker} full quotes: {exc}")
                    print(f"ERR {us_ticker} full quotes   {exc}", file=sys.stderr)

    if failures:
        print("\nCompleted with failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Raw data download completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
