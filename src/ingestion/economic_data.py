"""Download Massive U.S. macro data and Korean market-implied proxies.

Massive does not provide Korean CPI, Bank of Korea policy rates, or Korean
government-bond yields. The Korean output from this module is therefore a
market-proxy dataset and must never be interpreted as those direct measures.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from src.config import IngestionSettings, read_secret
from src.ingestion.massive_us import (
    DateRange,
    MassiveClient,
    merge_tabular_snapshot,
    missing_edge_ranges,
    print_page,
    print_status,
    print_task,
    write_parquet,
)


PROXY_WARNING = (
    "Market-implied proxy only; not Korean CPI, a BOK policy rate, or a "
    "Korean government-bond yield."
)


def _client(settings: IngestionSettings) -> MassiveClient:
    massive = settings.massive
    return MassiveClient(
        read_secret(massive.api_key_env),
        base_url=massive.base_url,
        timeout=massive.timeout_seconds,
        max_retries=massive.max_retries,
    )


def fetch_massive_economy(
    client: MassiveClient,
    endpoint: str,
    start: date,
    end: date,
    *,
    fields: tuple[str, ...] = (),
    label: str | None = None,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Fetch and normalize one paginated Massive Economy endpoint."""

    params: dict[str, object] = {
        "date.gte": start.isoformat(),
        "date.lte": end.isoformat(),
        "limit": 50_000,
        "sort": "date.asc",
    }
    if len(fields) > 1:
        params["fields"] = ",".join(fields)
    records: list[dict[str, object]] = []
    progress_label = label or endpoint
    for page_number, page in enumerate(client.iter_pages(endpoint, params), start=1):
        records.extend(page)
        print_page(progress_label, page_number, len(records))
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        if allow_empty:
            return frame
        raise RuntimeError(f"Massive returned no observations from {endpoint}")
    if "date" not in frame:
        raise RuntimeError(f"Massive response from {endpoint} has no date field")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["source"] = "Massive Economy API"
    frame["source_endpoint"] = endpoint
    frame["retrieved_at_utc"] = datetime.now(timezone.utc)
    return frame.sort_values("date").drop_duplicates("date", keep="last")


def download_us_economic(
    start: date, end: date, *, force: bool = False
) -> dict[str, int | str]:
    """Download U.S. yields, inflation, and funding/policy conditions."""

    if start > end:
        raise ValueError("start must not be later than end")
    settings = IngestionSettings()
    client = _client(settings)
    results: dict[str, int | str] = {}
    jobs = settings.economic.massive_us_endpoints
    total = len(jobs)
    print_status(
        f"U.S. economic ingestion started for {start} through {end} "
        f"({total} datasets)"
    )
    for index, (endpoint, fields, relative_dataset) in enumerate(jobs, start=1):
        label = relative_dataset.rsplit("/", 1)[-1]
        destination = settings.paths.raw_part(
            "us", "economic", *relative_dataset.split("/")
        )
        gaps = (
            (DateRange(start, end),)
            if force
            else missing_edge_ranges(destination, "date", start, end)
        )
        if not gaps:
            results[str(destination)] = "existing"
            print_task(index, total, label, "SKIPPED")
            continue
        print_task(
            index, total, label,
            "DOWNLOADING " + ", ".join(gap.label() for gap in gaps),
        )
        additions: list[pd.DataFrame] = []
        for gap_index, gap in enumerate(gaps, start=1):
            frame = fetch_massive_economy(
                client, endpoint, gap.start, gap.end, fields=fields,
                label=f"{label} gap {gap_index}/{len(gaps)}",
                allow_empty=destination.is_file() and not force,
            )
            if not frame.empty:
                additions.append(frame)
        downloaded = sum(len(frame) for frame in additions)
        if force:
            if not additions:
                raise RuntimeError(f"Massive returned no observations from {endpoint}")
            replacement = pd.concat(additions, ignore_index=True)
            replacement = replacement.drop_duplicates("date", keep="last").sort_values("date")
            write_parquet(replacement, destination)
            total_rows = len(replacement)
        elif additions:
            total_rows = merge_tabular_snapshot(
                destination, additions, keys=("date",)
            )
        else:
            total_rows = len(pd.read_parquet(destination, columns=["date"]))
        results[str(destination)] = downloaded
        print_task(
            index, total, label,
            f"DONE +{downloaded:,} / {total_rows:,} total rows",
        )
    print_status("U.S. economic ingestion complete")
    return results


def _proxy_missing_ranges(
    existing: pd.DataFrame,
    ticker: str,
    start: date,
    end: date,
) -> tuple[DateRange, ...]:
    if existing.empty or "ticker" not in existing or "timestamp" not in existing:
        return (DateRange(start, end),)
    values = pd.to_datetime(
        existing.loc[existing["ticker"].eq(ticker), "timestamp"],
        errors="coerce", utc=True,
    ).dropna()
    if values.empty:
        return (DateRange(start, end),)
    observed_start, observed_end = values.min().date(), values.max().date()
    gaps: list[DateRange] = []
    leading_end = min(end, observed_start - timedelta(days=1))
    if start <= leading_end:
        gaps.append(DateRange(start, leading_end))
    trailing_start = max(start, observed_end + timedelta(days=1))
    if trailing_start <= end:
        gaps.append(DateRange(trailing_start, end))
    return tuple(gaps)


def download_korean_market_proxies(
    start: date, end: date, *, force: bool = False
) -> dict[str, int | str]:
    """Write EWY, FLKR, and USD/KRW daily market-implied controls."""

    if start > end:
        raise ValueError("start must not be later than end")
    settings = IngestionSettings()
    destination = settings.paths.raw_part(
        "korea", "economic", "proxies", "KR_MARKET_PROXIES_daily"
    )
    print_status(
        f"Korean market-proxy ingestion started for {start} through {end}"
    )
    existing = (
        pd.DataFrame()
        if force or not destination.is_file()
        else pd.read_parquet(destination)
    )
    client = _client(settings)
    additions: list[pd.DataFrame] = []
    proxies = settings.massive.korean_proxy_tickers
    for index, (ticker, role) in enumerate(proxies, start=1):
        gaps = (
            (DateRange(start, end),)
            if force
            else _proxy_missing_ranges(existing, ticker, start, end)
        )
        label = f"{ticker} ({role})"
        if not gaps:
            print_task(index, len(proxies), label, "SKIPPED")
            continue
        print_task(
            index, len(proxies), label,
            "DOWNLOADING " + ", ".join(gap.label() for gap in gaps),
        )
        for gap_index, gap in enumerate(gaps, start=1):
            frame = client.get_aggregates(
                ticker, gap.start.isoformat(), gap.end.isoformat(),
                multiplier=1, timespan="day",
                include_volume=not ticker.startswith("C:"),
                progress=lambda page, rows, current_label=f"{label} gap {gap_index}/{len(gaps)}": print_page(current_label, page, rows),
            )
            if frame.empty:
                if force or not destination.is_file():
                    raise RuntimeError(f"Massive returned no daily proxy bars for {ticker}")
                continue
            frame["proxy_role"] = role
            frame["is_direct_korean_macro_measure"] = False
            frame["interpretation_warning"] = PROXY_WARNING
            frame["source"] = "Massive market aggregates"
            frame["retrieved_at_utc"] = datetime.now(timezone.utc)
            additions.append(frame)
    downloaded = sum(len(frame) for frame in additions)
    if force:
        if not additions:
            raise RuntimeError("Massive returned no Korean market-proxy rows")
        frame = pd.concat(additions, ignore_index=True)
        frame = frame.drop_duplicates(["ticker", "timestamp"], keep="last").sort_values(["ticker", "timestamp"])
        write_parquet(frame, destination)
        total_rows = len(frame)
    elif additions:
        total_rows = merge_tabular_snapshot(
            destination, additions, keys=("ticker", "timestamp")
        )
    elif destination.is_file():
        total_rows = len(existing)
    else:
        raise RuntimeError("No Korean market-proxy data are available")
    print_status(
        f"Korean market-proxy ingestion complete: +{downloaded:,} / "
        f"{total_rows:,} total rows -> "
        f"{destination}"
    )
    return {str(destination): downloaded if downloaded else "existing"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, default=date.today())
    parser.add_argument(
        "--source",
        choices=("all", "us", "korean-proxies"),
        default="all",
    )
    parser.add_argument(
        "--force", "--refresh-all", dest="force", action="store_true",
        help="Replace every selected snapshot for the requested date range",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start_date > args.end_date:
        raise SystemExit("start-date must not be later than end-date")
    results: dict[str, int | str] = {}
    if args.source in {"all", "us"}:
        results.update(download_us_economic(
            args.start_date, args.end_date, force=args.force
        ))
    if args.source in {"all", "korean-proxies"}:
        results.update(
            download_korean_market_proxies(
                args.start_date, args.end_date, force=args.force
            )
        )
    for path, rows in results.items():
        label = f"{rows:,} rows" if isinstance(rows, int) else rows
        print(f"{label:>16} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
