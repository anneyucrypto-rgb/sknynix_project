from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.config import PathSettings
from src.engine.dataset_builder import DatasetBuilder, _normalize_contract_timestamps
from src.engine.duckdb_engine import DuckDBEngine


def test_quote_bucketing_casts_nanosecond_timestamptz():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        quote_dir = root / "data" / "raw" / "us" / "market" / "SKHY_quotes"
        quote_dir.mkdir(parents=True)
        quotes = pd.DataFrame({
            "sip_datetime": pd.to_datetime([
                "2026-01-05T12:01:00.000000001Z",
                "2026-01-05T12:04:00.000000002Z",
            ]),
            "bid_price": [10.0, 10.1],
            "ask_price": [10.2, 10.3],
            "bid_size": [100.0, 110.0],
            "ask_size": [120.0, 130.0],
        })
        quotes.to_parquet(quote_dir / "part-00000.parquet", index=False)

        engine = DuckDBEngine(":memory:")
        builder = DatasetBuilder(PathSettings(project_root=root), engine)
        result = builder._quote_buckets()
        builder.close()

        assert len(result) == 1
        assert result.loc[0, "bid"] == 10.1
        assert result.loc[0, "quote_updates"] == 2
        assert str(result["bucket"].dtype).endswith("UTC]")


def test_cached_contract_timestamps_are_normalized_to_nanoseconds():
    frame = pd.DataFrame({
        "us_close_ts_utc": pd.Series(
            pd.to_datetime(["2026-01-05T21:00:00Z"]),
            dtype="datetime64[us, UTC]",
        )
    })
    result = _normalize_contract_timestamps(frame, "daily")
    assert str(result["us_close_ts_utc"].dtype) == "datetime64[ns, UTC]"
