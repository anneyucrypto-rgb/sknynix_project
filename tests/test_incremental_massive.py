from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingestion.economic_data import _proxy_missing_ranges
from src.ingestion.massive_us import (
    DateRange,
    TRADE_SCHEMA,
    merge_bar_snapshot,
    merge_tabular_snapshot,
    merge_streamed_snapshots,
    missing_edge_ranges,
    write_parquet,
)


def _bars(timestamps: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["SKHY"] * len(timestamps),
        "timestamp": pd.to_datetime(timestamps, utc=True),
        "close": closes,
    })


def test_missing_ranges_only_extend_snapshot_edges():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "bars.parquet"
        write_parquet(
            _bars(["2026-01-02 14:30Z", "2026-01-05 14:30Z"], [10, 11]),
            path,
        )
        assert missing_edge_ranges(
            path, "timestamp", date(2026, 1, 1), date(2026, 1, 7)
        ) == (
            DateRange(date(2026, 1, 1), date(2026, 1, 1)),
            DateRange(date(2026, 1, 6), date(2026, 1, 7)),
        )


def test_bar_merge_preserves_existing_and_deduplicates_keys():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "bars.parquet"
        write_parquet(_bars(["2026-01-02 14:30Z"], [10]), path)
        total = merge_bar_snapshot(
            path,
            [_bars(["2026-01-02 14:30Z", "2026-01-05 14:30Z"], [12, 13])],
        )
        result = pd.read_parquet(path)
        assert total == 2
        assert result["close"].tolist() == [12, 13]


def test_tick_merge_streams_leading_and_trailing_segments_in_order():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        destination = root / "trades.parquet"
        leading = root / "leading.parquet"
        trailing = root / "trailing.parquet"

        def write_tick(path: Path, timestamp: str, sequence: int) -> None:
            row = {
                "ticker": "SKHY",
                "sip_timestamp": pd.Timestamp(timestamp).value,
                "sip_datetime": pd.Timestamp(timestamp),
                "sequence_number": sequence,
                "price": 10.0,
                "size": 1.0,
            }
            pq.write_table(pa.Table.from_pylist([row], schema=TRADE_SCHEMA), path)

        write_tick(destination, "2026-01-02 14:30Z", 2)
        write_tick(leading, "2026-01-01 14:30Z", 1)
        write_tick(trailing, "2026-01-03 14:30Z", 3)

        total = merge_streamed_snapshots(
            destination,
            [
                (DateRange(date(2026, 1, 1), date(2026, 1, 1)), leading),
                (DateRange(date(2026, 1, 3), date(2026, 1, 3)), trailing),
            ],
            TRADE_SCHEMA,
        )
        result = pd.read_parquet(destination)
        assert total == 3
        assert result["sequence_number"].tolist() == [1, 2, 3]
        assert not leading.exists()
        assert not trailing.exists()


def test_generic_snapshot_merge_uses_dataset_natural_key():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "daily.parquet"
        existing = pd.DataFrame({
            "ticker": ["000660.KS"],
            "date": pd.to_datetime(["2026-01-02"]),
            "close": [100.0],
        })
        addition = pd.DataFrame({
            "ticker": ["000660.KS", "000660.KS"],
            "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "close": [101.0, 102.0],
        })
        write_parquet(existing, path)
        total = merge_tabular_snapshot(
            path, [addition], keys=("ticker", "date")
        )
        result = pd.read_parquet(path)
        assert total == 2
        assert result["close"].tolist() == [101.0, 102.0]


def test_proxy_gap_detection_is_ticker_specific():
    existing = pd.DataFrame({
        "ticker": ["EWY", "EWY", "FLKR"],
        "timestamp": pd.to_datetime([
            "2026-01-02T00:00:00Z", "2026-01-05T00:00:00Z",
            "2026-01-03T00:00:00Z",
        ]),
    })
    assert _proxy_missing_ranges(
        existing, "EWY", date(2026, 1, 2), date(2026, 1, 6)
    ) == (DateRange(date(2026, 1, 6), date(2026, 1, 6)),)
    assert _proxy_missing_ranges(
        existing, "C:USDKRW", date(2026, 1, 2), date(2026, 1, 6)
    ) == (DateRange(date(2026, 1, 2), date(2026, 1, 6)),)
