"""Download SK hynix and Korean benchmark daily OHLCV snapshots."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import IngestionSettings
from src.ingestion.massive_us import (
    DateRange,
    merge_tabular_snapshot,
    missing_edge_ranges,
    print_status,
    print_task,
    write_parquet,
)


def fetch_daily(
    symbol: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Download and normalize one Korean daily price series."""

    if start > end:
        raise ValueError("start must not be later than end")
    raw = yf.download(
        symbol,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise RuntimeError(f"Yahoo Finance returned no daily rows for {symbol}")
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
    frame.columns = [str(column).lower().replace(" ", "_") for column in frame]
    frame.insert(0, "ticker", symbol)
    columns = [
        "ticker", "date", "open", "high", "low", "close",
        "adjusted_close", "volume",
    ]
    return frame[columns]


def download_daily(
    symbol: str,
    start: date,
    end: date,
    destination: Path,
    *,
    force: bool = False,
) -> tuple[int, int] | None:
    """Download only uncovered daily intervals and merge by ticker/date."""

    gaps = (
        (DateRange(start, end),)
        if force
        else missing_edge_ranges(destination, "date", start, end)
    )
    if not gaps:
        return None
    additions: list[pd.DataFrame] = []
    for gap in gaps:
        try:
            additions.append(fetch_daily(symbol, gap.start, gap.end))
        except RuntimeError:
            if force or not destination.is_file():
                raise
    downloaded = sum(len(frame) for frame in additions)
    if force:
        if not additions:
            raise RuntimeError(f"Yahoo Finance returned no daily rows for {symbol}")
        replacement = pd.concat(additions, ignore_index=True)
        replacement = replacement.drop_duplicates(
            ["ticker", "date"], keep="last"
        ).sort_values(["ticker", "date"])
        write_parquet(replacement, destination)
        return downloaded, len(replacement)
    total = merge_tabular_snapshot(
        destination, additions, keys=("ticker", "date")
    ) if additions else len(pd.read_parquet(destination, columns=["date"]))
    return downloaded, total


def download_configured_daily(
    start: date, end: date, *, force: bool = False
) -> dict[str, int | str]:
    settings = IngestionSettings()
    results: dict[str, int | str] = {}
    datasets = settings.korean.daily_symbols
    total = len(datasets)
    print_status(
        f"Korean daily ingestion started for {start} through {end} "
        f"({total} datasets)"
    )
    for index, (symbol, dataset) in enumerate(datasets, start=1):
        parent = "000660_daily" if dataset == "000660_daily" else f"sector_daily/{dataset}"
        destination = settings.paths.raw_part("korea", "market", *parent.split("/"))
        label = f"{symbol} -> {dataset}"
        gaps = (DateRange(start, end),) if force else missing_edge_ranges(
            destination, "date", start, end
        )
        if not gaps:
            results[str(destination)] = "existing"
            print_task(index, total, label, "SKIPPED")
            continue
        print_task(
            index, total, label,
            "DOWNLOADING " + ", ".join(gap.label() for gap in gaps),
        )
        outcome = download_daily(symbol, start, end, destination, force=force)
        if outcome is None:
            results[str(destination)] = "existing"
            print_task(index, total, label, "SKIPPED")
            continue
        downloaded, total_rows = outcome
        results[str(destination)] = downloaded
        print_task(
            index, total, label,
            f"DONE +{downloaded:,} / {total_rows:,} total rows",
        )
    print_status("Korean daily ingestion complete")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, default=date.today())
    parser.add_argument(
        "--force", "--refresh-all", dest="force", action="store_true",
        help="Replace every selected snapshot for the requested date range",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path, rows in download_configured_daily(
        args.start_date, args.end_date, force=args.force
    ).items():
        label = f"{rows:,} rows" if isinstance(rows, int) else rows
        print(f"{label:>16} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
