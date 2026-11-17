"""Download USD/KRW one-minute and five-minute raw bars from Massive."""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from src.config import IngestionSettings, read_secret
from src.ingestion.massive_us import (
    DateRange,
    MassiveClient,
    merge_bar_snapshot,
    missing_edge_ranges,
    print_status,
    print_page,
    print_task,
    write_parquet,
)


def download_usdkrw(
    start: date,
    end: date,
    *,
    force: bool = False,
) -> dict[str, int | str]:
    if start > end:
        raise ValueError("start must not be later than end")
    settings = IngestionSettings()
    massive = settings.massive
    fx = settings.fx
    client = MassiveClient(
        read_secret(massive.api_key_env),
        base_url=massive.base_url,
        timeout=massive.timeout_seconds,
        max_retries=massive.max_retries,
    )
    results: dict[str, int | str] = {}
    total = len(fx.bar_minutes)
    print_status(
        f"USD/KRW ingestion started for {start} through {end} "
        f"({total} datasets)"
    )
    for index, minutes in enumerate(fx.bar_minutes, start=1):
        label = f"USD/KRW {minutes}-minute bars"
        destination = settings.paths.raw_part(
            "fx", f"{fx.dataset_stem}_{minutes}min"
        )
        gaps = (
            (DateRange(start, end),)
            if force
            else missing_edge_ranges(destination, "timestamp", start, end)
        )
        if not gaps:
            results[str(destination)] = "existing"
            print_task(index, total, label, "SKIPPED")
            continue
        print_task(
            index, total, label,
            "DOWNLOADING " + ", ".join(gap.label() for gap in gaps),
        )
        additions = []
        for gap_index, gap in enumerate(gaps, start=1):
            gap_label = f"{label} gap {gap_index}/{len(gaps)}"
            frame = client.get_aggregates(
                fx.massive_ticker,
                gap.start.isoformat(),
                gap.end.isoformat(),
                multiplier=minutes,
                timespan="minute",
                include_volume=False,
                progress=lambda page, rows, current_label=gap_label: print_page(
                    current_label, page, rows
                ),
            )
            additions.append(frame)
        downloaded = sum(len(frame) for frame in additions)
        if downloaded == 0 and (force or not destination.is_file()):
            raise RuntimeError(f"Massive returned no {minutes}-minute USD/KRW bars")
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
            total_rows = len(pd.read_parquet(destination, columns=["timestamp"]))
        results[str(destination)] = downloaded
        print_task(
            index, total, label,
            f"DONE +{downloaded:,} / {total_rows:,} total rows",
        )
    print_status("USD/KRW ingestion complete")
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
    for path, result in download_usdkrw(
        args.start_date, args.end_date, force=args.force
    ).items():
        label = f"{result:,} rows" if isinstance(result, int) else result
        print(f"{label:>16} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
