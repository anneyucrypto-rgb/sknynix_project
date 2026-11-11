"""Project-specific Massive REST client for bars and historical quotes."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests


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
                time.sleep(min(2**attempt, 30))
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
                    time.sleep(float(wait) if wait else min(2**attempt, 30))
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
    ) -> pd.DataFrame:
        """Download normalized bars, optionally excluding volume-related fields."""
        output_columns = AGGREGATE_COLUMNS if include_volume else PRICE_COLUMNS
        path = (
            f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/"
            f"{start_date}/{end_date}"
        )
        records: list[dict[str, Any]] = []
        for page in self.iter_pages(
            path,
            {
                "adjusted": str(adjusted).lower(),
                "sort": "asc",
                "limit": 50_000,
            },
        ):
            records.extend(page)

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
            for page in self.iter_pages(f"/v3/quotes/{ticker}", params):
                if not page:
                    continue
                frame = _normalize_quotes(page, ticker)
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary,
                        table.schema,
                        compression="zstd",
                        use_dictionary=True,
                    )
                writer.write_table(table)
                total += len(frame)
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
            empty = _empty_quote_frame()
            pq.write_table(
                pa.Table.from_pandas(empty, preserve_index=False),
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
            for page in self.iter_pages(f"/v3/trades/{ticker}", params):
                if not page:
                    continue
                frame = _normalize_trades(page, ticker)
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary,
                        table.schema,
                        compression="zstd",
                        use_dictionary=True,
                    )
                writer.write_table(table)
                total += len(frame)
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
                pa.Table.from_pandas(_empty_trade_frame(), preserve_index=False),
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


def _empty_quote_frame() -> pd.DataFrame:
    frame = pd.DataFrame({column: pd.Series(dtype="object") for column in QUOTE_COLUMNS})
    for column in ("sip_datetime", "participant_datetime"):
        frame[column] = pd.Series(dtype="datetime64[ns, UTC]")
    return frame


def _empty_trade_frame() -> pd.DataFrame:
    frame = pd.DataFrame({column: pd.Series(dtype="object") for column in TRADE_COLUMNS})
    for column in ("sip_datetime", "participant_datetime"):
        frame[column] = pd.Series(dtype="datetime64[ns, UTC]")
    return frame
