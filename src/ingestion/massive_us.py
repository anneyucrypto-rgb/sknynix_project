"""Download SKHY, SPY, and SOXX bars plus raw SKHY trades and quotes."""

from __future__ import annotations

import json
import os
import time
import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from src.config import IngestionSettings, read_secret


BASE_URL = "https://api.massive.com"
AGGREGATE_COLUMNS = [
    "ticker",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "transactions",
]
PRICE_COLUMNS = [
    "ticker",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "vwap",
]
QUOTE_COLUMNS = [
    "ticker",
    "sip_timestamp",
    "sip_datetime",
    "participant_timestamp",
    "participant_datetime",
    "trf_timestamp",
    "sequence_number",
    "bid_price",
    "bid_size",
    "bid_exchange",
    "ask_price",
    "ask_size",
    "ask_exchange",
    "conditions",
    "indicators",
    "tape",
]
TRADE_COLUMNS = [
    "ticker",
    "sip_timestamp",
    "sip_datetime",
    "participant_timestamp",
    "participant_datetime",
    "trf_timestamp",
    "sequence_number",
    "price",
    "size",
    "exchange",
    "conditions",
    "tape",
]

# Massive can omit nullable fields on one page and populate them on the next.
# Without an explicit Arrow schema, an all-null page is inferred as ``null`` and
# a later numeric/string page cannot be appended to the same Parquet writer.
QUOTE_SCHEMA = pa.schema(
    [
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("sip_timestamp", pa.int64()),
        pa.field("sip_datetime", pa.timestamp("ns", tz="UTC")),
        pa.field("participant_timestamp", pa.int64()),
        pa.field("participant_datetime", pa.timestamp("ns", tz="UTC")),
        pa.field("trf_timestamp", pa.int64()),
        pa.field("sequence_number", pa.int64()),
        pa.field("bid_price", pa.float64()),
        pa.field("bid_size", pa.float64()),
        pa.field("bid_exchange", pa.int64()),
        pa.field("ask_price", pa.float64()),
        pa.field("ask_size", pa.float64()),
        pa.field("ask_exchange", pa.int64()),
        pa.field("conditions", pa.string()),
        pa.field("indicators", pa.string()),
        pa.field("tape", pa.int64()),
    ]
)
TRADE_SCHEMA = pa.schema(
    [
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("sip_timestamp", pa.int64()),
        pa.field("sip_datetime", pa.timestamp("ns", tz="UTC")),
        pa.field("participant_timestamp", pa.int64()),
        pa.field("participant_datetime", pa.timestamp("ns", tz="UTC")),
        pa.field("trf_timestamp", pa.int64()),
        pa.field("sequence_number", pa.int64()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.float64()),
        pa.field("exchange", pa.int64()),
        pa.field("conditions", pa.string()),
        pa.field("tape", pa.int64()),
    ]
)


@dataclass(frozen=True, order=True)
class DateRange:
    """Inclusive calendar-date interval requested from Massive."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("DateRange start must not follow end")

    def label(self) -> str:
        return f"{self.start.isoformat()}..{self.end.isoformat()}"


def print_status(message: str) -> None:
    """Print an immediately visible ingestion status message."""

    print(f"[INGEST] {message}", flush=True)


def print_task(index: int, total: int, label: str, state: str) -> None:
    """Print dataset-level progress in a consistent format."""

    print_status(f"[{index}/{total}] {state:<10} {label}")


def print_page(label: str, page_number: int, rows: int) -> None:
    """Print throttled progress for large paginated responses."""

    if page_number == 1 or page_number % 10 == 0:
        print_status(
            f"{label}: received {page_number:,} pages / {rows:,} rows"
        )


class MassiveAPIError(RuntimeError):
    """Raised when Massive returns an unsuccessful response."""


class MassiveClient:
    """REST client with pagination, retry handling, and Parquet conversion."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 6,
    ) -> None:
        if not api_key:
            raise ValueError("A Massive API key is required")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "User-Agent": "skhynix-project/1.0",
            }
        )

    def _request_json(
        self, url: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if not url.startswith("http"):
            url = f"{self.base_url}/{url.lstrip('/')}"

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise MassiveAPIError(f"Massive request failed: {exc}") from exc
                delay = min(2**attempt, 30)
                print_status(
                    f"request failed ({exc}); retry {attempt + 1}/"
                    f"{self.max_retries} in {delay}s"
                )
                time.sleep(delay)
                continue

            if response.status_code == 200:
                payload = response.json()
                if payload.get("status") in {"ERROR", "NOT_AUTHORIZED"}:
                    message = payload.get("error") or payload.get("message") or payload["status"]
                    raise MassiveAPIError(f"Massive API error: {message}")
                return payload

            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt < self.max_retries:
                    wait = response.headers.get("Retry-After")
                    delay = float(wait) if wait else min(2**attempt, 30)
                    print_status(
                        f"Massive HTTP {response.status_code}; retry "
                        f"{attempt + 1}/{self.max_retries} in {delay:g}s"
                    )
                    time.sleep(delay)
                    continue

            try:
                body = response.json()
                detail = body.get("error") or body.get("message")
            except (ValueError, AttributeError):
                detail = response.text[:300]
            raise MassiveAPIError(
                f"Massive returned HTTP {response.status_code}: {detail or 'request rejected'}"
            )

        raise MassiveAPIError("Massive request failed after retries")

    def iter_pages(
        self, path: str, params: dict[str, Any] | None = None
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield result pages while following Massive's opaque next_url cursor."""
        url: str | None = path
        next_params = params
        while url:
            payload = self._request_json(url, next_params)
            yield payload.get("results") or []
            url = payload.get("next_url")
            next_params = None

    def get_first_trade(
        self, ticker: str, start_timestamp_ns: int, end_timestamp_ns: int
    ) -> dict[str, Any] | None:
        """Return the earliest reported trade within a nanosecond UTC interval."""
        params = {
            "timestamp.gte": start_timestamp_ns,
            "timestamp.lt": end_timestamp_ns,
            "sort": "timestamp",
            "order": "asc",
            "limit": 50_000,
        }
        for page in self.iter_pages(f"/v3/trades/{ticker}", params):
            if page:
                return page[0]
        return None

    def get_first_valid_quote(
        self, ticker: str, start_timestamp_ns: int, end_timestamp_ns: int
    ) -> dict[str, Any] | None:
        """Return the earliest two-sided, non-crossed NBBO quote in an interval."""
        params = {
            "timestamp.gte": start_timestamp_ns,
            "timestamp.lt": end_timestamp_ns,
            "sort": "timestamp",
            "order": "asc",
            "limit": 50_000,
        }
        for page in self.iter_pages(f"/v3/quotes/{ticker}", params):
            for quote in page:
                bid = quote.get("bid_price")
                ask = quote.get("ask_price")
                if (
                    bid is not None
                    and ask is not None
                    and bid > 0
                    and ask > 0
                    and ask >= bid
                ):
                    return quote
        return None

    def get_aggregates(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        *,
        multiplier: int = 5,
        timespan: str = "minute",
        adjusted: bool = True,
        include_volume: bool = True,
        progress: Callable[[int, int], None] | None = None,
    ) -> pd.DataFrame:
        """Download normalized bars, optionally excluding volume-related fields."""
        output_columns = AGGREGATE_COLUMNS if include_volume else PRICE_COLUMNS
        path = (
            f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/"
            f"{start_date}/{end_date}"
        )
        records: list[dict[str, Any]] = []
        for page_number, page in enumerate(
            self.iter_pages(
                path,
                {
                    "adjusted": str(adjusted).lower(),
                    "sort": "asc",
                    "limit": 50_000,
                },
            ),
            start=1,
        ):
            records.extend(page)
            if progress is not None:
                progress(page_number, len(records))

        if not records:
            return pd.DataFrame(columns=output_columns)

        frame = pd.DataFrame.from_records(records).rename(
            columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "vw": "vwap",
                "n": "transactions",
            }
        )
        frame.insert(0, "ticker", ticker)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        for column in AGGREGATE_COLUMNS:
            if column not in frame:
                frame[column] = pd.NA
        return frame[output_columns].sort_values("timestamp").reset_index(drop=True)

    def download_quotes_parquet(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        destination: Path,
        progress: Callable[[int, int], None] | None = None,
    ) -> int:
        """Stream raw NBBO quote pages into Parquet without retaining all ticks in RAM."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        writer: pq.ParquetWriter | None = None
        total = 0
        params = {
            "timestamp.gte": start_date,
            "timestamp.lt": f"{end_date}T23:59:59.999999999Z",
            "sort": "timestamp",
            "order": "asc",
            "limit": 50_000,
        }

        try:
            for page_number, page in enumerate(
                self.iter_pages(f"/v3/quotes/{ticker}", params), start=1
            ):
                if not page:
                    continue
                frame = _normalize_quotes(page, ticker)
                table = pa.Table.from_pandas(
                    frame, schema=QUOTE_SCHEMA, preserve_index=False
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary,
                        table.schema,
                        compression="zstd",
                        use_dictionary=True,
                    )
                writer.write_table(table)
                total += len(frame)
                if progress is not None:
                    progress(page_number, total)
        except Exception:
            if writer is not None:
                writer.close()
                writer = None
            temporary.unlink(missing_ok=True)
            raise
        finally:
            if writer is not None:
                writer.close()

        if total == 0:
            pq.write_table(
                pa.Table.from_pylist([], schema=QUOTE_SCHEMA),
                temporary,
                compression="zstd",
            )
        os.replace(temporary, destination)
        return total

    def download_trades_parquet(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        destination: Path,
        progress: Callable[[int, int], None] | None = None,
    ) -> int:
        """Stream raw trade pages into Parquet without retaining all ticks in RAM."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        writer: pq.ParquetWriter | None = None
        total = 0
        params = {
            "timestamp.gte": start_date,
            "timestamp.lt": f"{end_date}T23:59:59.999999999Z",
            "sort": "timestamp",
            "order": "asc",
            "limit": 50_000,
        }

        try:
            for page_number, page in enumerate(
                self.iter_pages(f"/v3/trades/{ticker}", params), start=1
            ):
                if not page:
                    continue
                frame = _normalize_trades(page, ticker)
                table = pa.Table.from_pandas(
                    frame, schema=TRADE_SCHEMA, preserve_index=False
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary,
                        table.schema,
                        compression="zstd",
                        use_dictionary=True,
                    )
                writer.write_table(table)
                total += len(frame)
                if progress is not None:
                    progress(page_number, total)
        except Exception:
            if writer is not None:
                writer.close()
                writer = None
            temporary.unlink(missing_ok=True)
            raise
        finally:
            if writer is not None:
                writer.close()

        if total == 0:
            pq.write_table(
                pa.Table.from_pylist([], schema=TRADE_SCHEMA),
                temporary,
                compression="zstd",
            )
        os.replace(temporary, destination)
        return total


def write_parquet(frame: pd.DataFrame, destination: Path) -> None:
    """Atomically write a DataFrame as compressed Parquet."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, destination)


def parquet_date_bounds(path: Path, date_column: str) -> tuple[date, date] | None:
    """Return observed UTC date bounds without reading unrelated columns."""

    if not path.is_file():
        return None
    parquet: pq.ParquetFile | None = None
    try:
        parquet = pq.ParquetFile(path)
        column_index = parquet.schema_arrow.get_field_index(date_column)
        if column_index < 0:
            return None
        minima: list[Any] = []
        maxima: list[Any] = []
        for row_group in range(parquet.metadata.num_row_groups):
            statistics = parquet.metadata.row_group(row_group).column(
                column_index
            ).statistics
            if statistics is not None and statistics.has_min_max:
                minima.append(statistics.min)
                maxima.append(statistics.max)
        if minima and maxima:
            minimum = pd.to_datetime(min(minima), errors="coerce", utc=True)
            maximum = pd.to_datetime(max(maxima), errors="coerce", utc=True)
            if not pd.isna(minimum) and not pd.isna(maximum):
                return minimum.date(), maximum.date()
        values = pd.read_parquet(path, columns=[date_column])[date_column]
    except Exception:
        return None
    finally:
        if parquet is not None:
            parquet.close()
    timestamps = pd.to_datetime(values, errors="coerce", utc=True).dropna()
    if timestamps.empty:
        return None
    return timestamps.min().date(), timestamps.max().date()


def missing_edge_ranges(
    path: Path,
    date_column: str,
    start: date,
    end: date,
) -> tuple[DateRange, ...]:
    """Find uncovered leading/trailing intervals in an immutable snapshot.

    Existing snapshots are produced atomically from complete requests, so a
    refresh only needs to extend their observed boundaries. Interior dates are
    deliberately not inferred as missing because weekends and exchange
    holidays legitimately contain no U.S. observations.
    """

    if start > end:
        raise ValueError("start must not be later than end")
    bounds = parquet_date_bounds(path, date_column)
    if bounds is None:
        return (DateRange(start, end),)
    observed_start, observed_end = bounds
    gaps: list[DateRange] = []
    leading_end = min(end, observed_start - timedelta(days=1))
    if start <= leading_end:
        gaps.append(DateRange(start, leading_end))
    trailing_start = max(start, observed_end + timedelta(days=1))
    if trailing_start <= end:
        gaps.append(DateRange(trailing_start, end))
    return tuple(gaps)


def merge_bar_snapshot(existing_path: Path, additions: list[pd.DataFrame]) -> int:
    """Merge new bars into a target snapshot and return its total row count."""

    return merge_tabular_snapshot(
        existing_path, additions, keys=("ticker", "timestamp")
    )


def merge_tabular_snapshot(
    existing_path: Path,
    additions: list[pd.DataFrame],
    *,
    keys: tuple[str, ...],
) -> int:
    """Atomically merge small/medium Parquet snapshots by natural key."""

    frames: list[pd.DataFrame] = []
    if existing_path.is_file():
        frames.append(pd.read_parquet(existing_path))
    frames.extend(frame for frame in additions if not frame.empty)
    if not frames:
        return 0
    combined = pd.concat(frames, ignore_index=True)
    missing = set(keys).difference(combined.columns)
    if missing:
        raise ValueError(f"Cannot merge snapshot; missing key columns: {sorted(missing)}")
    combined = combined.drop_duplicates(list(keys), keep="last").sort_values(list(keys))
    write_parquet(combined, existing_path)
    return len(combined)


def merge_streamed_snapshots(
    destination: Path,
    additions: list[tuple[DateRange, Path]],
    schema: pa.Schema,
) -> int:
    """Atomically combine non-overlapping tick snapshots without pandas loads."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".merge.part")
    temporary.unlink(missing_ok=True)
    existing_bounds = parquet_date_bounds(destination, "sip_datetime")
    ordered: list[Path] = []
    if existing_bounds is None:
        ordered = [path for _, path in sorted(additions)]
    else:
        observed_start, observed_end = existing_bounds
        ordered.extend(path for gap, path in sorted(additions) if gap.end < observed_start)
        ordered.append(destination)
        ordered.extend(path for gap, path in sorted(additions) if gap.start > observed_end)

    writer: pq.ParquetWriter | None = None
    total = 0
    try:
        writer = pq.ParquetWriter(
            temporary, schema, compression="zstd", use_dictionary=True
        )
        for source in ordered:
            parquet = pq.ParquetFile(source)
            try:
                for batch in parquet.iter_batches(batch_size=250_000):
                    table = pa.Table.from_batches([batch]).cast(schema, safe=False)
                    writer.write_table(table)
                    total += table.num_rows
            finally:
                parquet.close()
        writer.close()
        writer = None
        os.replace(temporary, destination)
    except Exception:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    finally:
        for _, path in additions:
            path.unlink(missing_ok=True)
    return total


def _normalize_quotes(records: list[dict[str, Any]], ticker: str) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records)
    frame.insert(0, "ticker", ticker)

    for raw, converted in (
        ("sip_timestamp", "sip_datetime"),
        ("participant_timestamp", "participant_datetime"),
    ):
        if raw not in frame:
            frame[raw] = pd.NA
        frame[converted] = pd.to_datetime(frame[raw], unit="ns", utc=True, errors="coerce")

    for column in ("conditions", "indicators"):
        if column not in frame:
            frame[column] = None
        frame[column] = frame[column].map(
            lambda value: json.dumps(value, separators=(",", ":"))
            if isinstance(value, (list, dict))
            else None
        )

    integer_columns = [
        "sip_timestamp",
        "participant_timestamp",
        "trf_timestamp",
        "sequence_number",
        "bid_exchange",
        "ask_exchange",
        "tape",
    ]
    float_columns = ["bid_price", "bid_size", "ask_price", "ask_size"]
    for column in integer_columns:
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in float_columns:
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")

    for column in QUOTE_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[QUOTE_COLUMNS]


def _normalize_trades(records: list[dict[str, Any]], ticker: str) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records)
    frame.insert(0, "ticker", ticker)
    for raw, converted in (
        ("sip_timestamp", "sip_datetime"),
        ("participant_timestamp", "participant_datetime"),
    ):
        if raw not in frame:
            frame[raw] = pd.NA
        frame[converted] = pd.to_datetime(
            frame[raw], unit="ns", utc=True, errors="coerce"
        )
    if "conditions" not in frame:
        frame["conditions"] = None
    frame["conditions"] = frame["conditions"].map(
        lambda value: json.dumps(value, separators=(",", ":"))
        if isinstance(value, (list, dict))
        else None
    )
    for column in (
        "sip_timestamp",
        "participant_timestamp",
        "trf_timestamp",
        "sequence_number",
        "exchange",
        "tape",
    ):
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in ("price", "size"):
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
    for column in TRADE_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[TRADE_COLUMNS]


def download_configured_us(
    start: date,
    end: date,
    *,
    include_trades: bool = False,
    include_quotes: bool = False,
    force: bool = False,
) -> dict[str, int | str]:
    """Download the U.S. raw inputs used by H1–H3."""

    if start > end:
        raise ValueError("start must not be later than end")
    settings = IngestionSettings()
    massive = settings.massive
    client = MassiveClient(
        read_secret(massive.api_key_env),
        base_url=massive.base_url,
        timeout=massive.timeout_seconds,
        max_retries=massive.max_retries,
    )
    results: dict[str, int | str] = {}
    total_tasks = len((massive.ticker, *massive.control_tickers)) * len(
        massive.bar_minutes
    ) + int(include_trades) + int(include_quotes)
    task_index = 0
    print_status(
        f"U.S. ingestion started for {start} through {end} "
        f"({total_tasks} datasets)"
    )
    for ticker in (massive.ticker, *massive.control_tickers):
        for minutes in massive.bar_minutes:
            task_index += 1
            label = f"{ticker} {minutes}-minute bars"
            destination = settings.paths.raw_part(
                "us", "market", f"{ticker}_{minutes}min"
            )
            gaps = (
                (DateRange(start, end),)
                if force
                else missing_edge_ranges(destination, "timestamp", start, end)
            )
            if not gaps:
                results[str(destination)] = "existing"
                print_task(task_index, total_tasks, label, "SKIPPED")
                continue
            print_task(
                task_index, total_tasks, label,
                "DOWNLOADING " + ", ".join(gap.label() for gap in gaps),
            )
            additions: list[pd.DataFrame] = []
            for gap_index, gap in enumerate(gaps, start=1):
                gap_label = f"{label} gap {gap_index}/{len(gaps)}"
                frame = client.get_aggregates(
                    ticker,
                    gap.start.isoformat(),
                    gap.end.isoformat(),
                    multiplier=minutes,
                    timespan="minute",
                    include_volume=True,
                    progress=lambda page, rows, current_label=gap_label: print_page(
                        current_label, page, rows
                    ),
                )
                additions.append(frame)
            downloaded = sum(len(frame) for frame in additions)
            if downloaded == 0 and (force or not destination.is_file()):
                raise RuntimeError(f"Massive returned no {minutes}-minute bars for {ticker}")
            if force:
                replacement = pd.concat(additions, ignore_index=True)
                replacement = replacement.drop_duplicates(
                    ["ticker", "timestamp"], keep="last"
                ).sort_values(["ticker", "timestamp"])
                write_parquet(replacement, destination)
                total_rows = len(replacement)
            elif downloaded:
                total_rows = merge_bar_snapshot(destination, additions)
            else:
                parquet = pq.ParquetFile(destination)
                try:
                    total_rows = parquet.metadata.num_rows
                finally:
                    parquet.close()
            results[str(destination)] = downloaded
            print_task(
                task_index, total_tasks, label,
                f"DONE +{downloaded:,} / {total_rows:,} total rows",
            )

    tick_jobs = (
        (include_trades, "trades", "sip_datetime", client.download_trades_parquet, TRADE_SCHEMA),
        (include_quotes, "quotes", "sip_datetime", client.download_quotes_parquet, QUOTE_SCHEMA),
    )
    for enabled, kind, date_column, downloader, schema in tick_jobs:
        if not enabled:
            continue
        task_index += 1
        label = f"{massive.ticker} full {kind}"
        destination = settings.paths.raw_part(
            "us", "market", f"{massive.ticker}_{kind}"
        )
        gaps = (
            (DateRange(start, end),)
            if force
            else missing_edge_ranges(destination, date_column, start, end)
        )
        if not gaps:
            results[str(destination)] = "existing"
            print_task(task_index, total_tasks, label, "SKIPPED")
            continue
        print_task(
            task_index, total_tasks, label,
            "DOWNLOADING " + ", ".join(gap.label() for gap in gaps),
        )
        if force:
            downloaded = downloader(
                massive.ticker,
                start.isoformat(),
                end.isoformat(),
                destination,
                progress=lambda page, rows, current_label=label: print_page(
                    current_label, page, rows
                ),
            )
            total_rows = downloaded
        else:
            additions: list[tuple[DateRange, Path]] = []
            downloaded = 0
            for gap_index, gap in enumerate(gaps, start=1):
                segment = destination.with_name(
                    f".{destination.stem}.gap-{gap_index}.parquet"
                )
                gap_label = f"{label} gap {gap_index}/{len(gaps)}"
                rows = downloader(
                    massive.ticker,
                    gap.start.isoformat(),
                    gap.end.isoformat(),
                    segment,
                    progress=lambda page, count, current_label=gap_label: print_page(
                        current_label, page, count
                    ),
                )
                downloaded += rows
                additions.append((gap, segment))
            total_rows = merge_streamed_snapshots(destination, additions, schema)
        results[str(destination)] = downloaded
        print_task(
            task_index,
            total_tasks,
            label,
            f"DONE +{downloaded:,} / {total_rows:,} total rows",
        )
    print_status("U.S. ingestion complete")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--trades", action="store_true", help="Download full SKHY trades")
    parser.add_argument("--quotes", action="store_true", help="Download full SKHY quotes")
    parser.add_argument(
        "--force",
        "--refresh-all",
        dest="force",
        action="store_true",
        help=(
            "Re-download every selected dataset for the requested date range "
            "and replace its existing snapshot"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = download_configured_us(
        args.start_date,
        args.end_date,
        include_trades=args.trades,
        include_quotes=args.quotes,
        force=args.force,
    )
    for path, result in results.items():
        label = f"{result:,} rows" if isinstance(result, int) else result
        print(f"{label:>16} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
