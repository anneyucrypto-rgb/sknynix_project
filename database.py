"""Build a persistent DuckDB database with UTC-normalized event-time data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import duckdb


PROJECT_DIR: Final = Path(__file__).resolve().parent
DEFAULT_DATA_DIR: Final = PROJECT_DIR / "data"
DEFAULT_DATABASE_PATH: Final = DEFAULT_DATA_DIR / "skhynix.duckdb"

REQUIRED_FILES: Final = {
    "skhy": Path("us/SKHY_5min.parquet"),
    "spy": Path("us/SPY_5min.parquet"),
    "soxx": Path("us/SOXX_5min.parquet"),
    "qqq": Path("us/QQQ_5min.parquet"),
    "usdkrw": Path("fx/USDKRW_5min.parquet"),
    "korea": Path("korea/000660_daily.parquet"),
}
OPTIONAL_QUOTES_FILE: Final = Path("quotes/SKHY_quotes.parquet")


def _source_paths(data_dir: Path) -> dict[str, Path]:
    paths = {name: data_dir / relative for name, relative in REQUIRED_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        joined = "\n- ".join(missing)
        raise FileNotFoundError(f"Required Parquet files are missing:\n- {joined}")
    return paths


def _load_us_price_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    parquet_path: Path,
    *,
    include_volume: bool,
) -> None:
    extra_columns = (
        ", volume::DOUBLE AS volume, transactions::BIGINT AS transactions"
        if include_volume
        else ""
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT
            ticker::VARCHAR AS ticker,
            timestamp::TIMESTAMPTZ AS timestamp_utc,
            timezone('America/New_York', timestamp::TIMESTAMPTZ) AS timestamp_ny,
            open::DOUBLE AS open,
            high::DOUBLE AS high,
            low::DOUBLE AS low,
            close::DOUBLE AS close,
            vwap::DOUBLE AS vwap
            {extra_columns}
        FROM read_parquet(?)
        ORDER BY timestamp_utc
        """,
        [str(parquet_path)],
    )


def _load_korean_daily_table(
    connection: duckdb.DuckDBPyConnection, parquet_path: Path
) -> None:
    connection.execute(
        """
        CREATE OR REPLACE TABLE korea_daily AS
        WITH source AS (
            SELECT
                ticker::VARCHAR AS ticker,
                date::DATE AS trading_date,
                date::DATE + TIME '15:30:00' AS close_timestamp_kst,
                timezone(
                    'Asia/Seoul',
                    date::DATE + TIME '15:30:00'
                ) AS availability_timestamp_utc,
                open::DOUBLE AS open,
                high::DOUBLE AS high,
                low::DOUBLE AS low,
                close::DOUBLE AS close,
                adjusted_close::DOUBLE AS adjusted_close,
                volume::BIGINT AS volume
            FROM read_parquet(?)
        ),
        returns AS (
            SELECT
                *,
                lag(adjusted_close) OVER (ORDER BY trading_date) AS prior_adjusted_close
            FROM source
        )
        SELECT
            *,
            CASE
                WHEN adjusted_close > 0 AND prior_adjusted_close > 0
                THEN ln(adjusted_close / prior_adjusted_close)
                ELSE NULL
            END AS log_return
        FROM returns
        ORDER BY availability_timestamp_utc
        """,
        [str(parquet_path)],
    )


def _load_quotes_table(
    connection: duckdb.DuckDBPyConnection, parquet_path: Path
) -> bool:
    if not parquet_path.is_file() or parquet_path.stat().st_size == 0:
        connection.execute(
            """
            CREATE OR REPLACE TABLE skhy_quotes (
                ticker VARCHAR,
                quote_timestamp_utc TIMESTAMPTZ,
                participant_timestamp BIGINT,
                sequence_number BIGINT,
                bid_price DOUBLE,
                bid_size DOUBLE,
                bid_exchange BIGINT,
                ask_price DOUBLE,
                ask_size DOUBLE,
                ask_exchange BIGINT,
                conditions VARCHAR,
                indicators VARCHAR,
                tape BIGINT
            )
            """
        )
        return False

    connection.execute(
        """
        CREATE OR REPLACE TABLE skhy_quotes AS
        SELECT
            ticker::VARCHAR AS ticker,
            sip_datetime::TIMESTAMPTZ AS quote_timestamp_utc,
            participant_timestamp::BIGINT AS participant_timestamp,
            sequence_number::BIGINT AS sequence_number,
            bid_price::DOUBLE AS bid_price,
            bid_size::DOUBLE AS bid_size,
            bid_exchange::BIGINT AS bid_exchange,
            ask_price::DOUBLE AS ask_price,
            ask_size::DOUBLE AS ask_size,
            ask_exchange::BIGINT AS ask_exchange,
            conditions::VARCHAR AS conditions,
            indicators::VARCHAR AS indicators,
            tape::BIGINT AS tape
        FROM read_parquet(?)
        ORDER BY quote_timestamp_utc
        """,
        [str(parquet_path)],
    )
    return True


def _create_event_anchors(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE OR REPLACE TABLE us_event_anchors AS
        WITH localized AS (
            SELECT
                *,
                timestamp_ny::DATE AS event_date_ny,
                timestamp_ny::TIME AS event_time_ny
            FROM skhy_5min
        ),
        candidates AS (
            SELECT
                'premarket_0400'::VARCHAR AS anchor_type,
                *
            FROM localized
            WHERE event_time_ny >= TIME '04:00:00'
              AND event_time_ny <= TIME '20:00:00'

            UNION ALL

            SELECT
                'regular_0930'::VARCHAR AS anchor_type,
                *
            FROM localized
            WHERE event_time_ny >= TIME '09:30:00'
              AND event_time_ny <= TIME '20:00:00'
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY anchor_type, event_date_ny
                    ORDER BY timestamp_utc
                ) AS anchor_rank
            FROM candidates
        )
        SELECT
            event_date_ny,
            anchor_type,
            timestamp_utc AS anchor_timestamp_utc,
            timestamp_ny AS anchor_timestamp_ny,
            event_time_ny AS actual_anchor_time_ny,
            CASE
                WHEN anchor_type = 'premarket_0400'
                THEN event_time_ny = TIME '04:00:00'
                WHEN anchor_type = 'regular_0930'
                THEN event_time_ny = TIME '09:30:00'
                ELSE FALSE
            END AS exact_scheduled_anchor,
            ticker AS skhy_ticker,
            open AS skhy_open,
            high AS skhy_high,
            low AS skhy_low,
            close AS skhy_close,
            vwap AS skhy_vwap,
            volume AS skhy_volume,
            transactions AS skhy_transactions
        FROM ranked
        WHERE anchor_rank = 1
        ORDER BY event_date_ny, anchor_type
        """
    )


def _create_korean_asof_matches(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE OR REPLACE TABLE event_korea_asof AS
        SELECT
            u.event_date_ny,
            u.anchor_type,
            u.anchor_timestamp_utc,
            u.anchor_timestamp_ny,
            u.actual_anchor_time_ny,
            u.exact_scheduled_anchor,
            u.skhy_ticker,
            u.skhy_open,
            u.skhy_high,
            u.skhy_low,
            u.skhy_close,
            u.skhy_vwap,
            u.skhy_volume,
            u.skhy_transactions,
            k.ticker AS korea_ticker,
            k.trading_date AS korea_trading_date,
            k.close_timestamp_kst AS korea_close_timestamp_kst,
            k.availability_timestamp_utc AS korea_availability_timestamp_utc,
            k.open AS korea_open,
            k.high AS korea_high,
            k.low AS korea_low,
            k.close AS korea_close,
            k.adjusted_close AS korea_adjusted_close,
            k.volume AS korea_volume,
            k.log_return AS korea_log_return,
            CASE
                WHEN k.trading_date IS NULL THEN NULL
                ELSE date_diff('day', k.trading_date, u.event_date_ny)
            END AS korea_stale_days,
            coalesce(k.trading_date = u.event_date_ny, FALSE) AS korea_new_close,
            coalesce(k.availability_timestamp_utc <= u.anchor_timestamp_utc, FALSE)
                AS availability_check_passed
        FROM us_event_anchors AS u
        ASOF LEFT JOIN korea_daily AS k
            ON u.anchor_timestamp_utc >= k.availability_timestamp_utc
        ORDER BY u.event_date_ny, u.anchor_type
        """
    )


def _create_indexes(connection: duckdb.DuckDBPyConnection) -> None:
    index_statements = (
        "CREATE INDEX IF NOT EXISTS idx_skhy_timestamp ON skhy_5min(timestamp_utc)",
        "CREATE INDEX IF NOT EXISTS idx_spy_timestamp ON spy_5min(timestamp_utc)",
        "CREATE INDEX IF NOT EXISTS idx_soxx_timestamp ON soxx_5min(timestamp_utc)",
        "CREATE INDEX IF NOT EXISTS idx_qqq_timestamp ON qqq_5min(timestamp_utc)",
        "CREATE INDEX IF NOT EXISTS idx_usdkrw_timestamp ON usdkrw_5min(timestamp_utc)",
        "CREATE INDEX IF NOT EXISTS idx_korea_available ON korea_daily(availability_timestamp_utc)",
        "CREATE INDEX IF NOT EXISTS idx_event_anchor ON us_event_anchors(anchor_timestamp_utc)",
    )
    for statement in index_statements:
        connection.execute(statement)


def _write_metadata(
    connection: duckdb.DuckDBPyConnection,
    data_dir: Path,
    database_path: Path,
    quotes_loaded: bool,
) -> None:
    connection.execute(
        """
        CREATE OR REPLACE TABLE database_metadata (
            built_at_utc TIMESTAMPTZ,
            data_directory VARCHAR,
            database_path VARCHAR,
            timezone_policy VARCHAR,
            korean_close_policy VARCHAR,
            quotes_loaded BOOLEAN
        )
        """
    )
    connection.execute(
        """
        INSERT INTO database_metadata
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            datetime.now(timezone.utc),
            str(data_dir),
            str(database_path),
            "All market instants stored as TIMESTAMPTZ and displayed in UTC",
            "Daily 000660.KS data available at 15:30 Asia/Seoul",
            quotes_loaded,
        ],
    )


def build_database(
    database_path: Path = DEFAULT_DATABASE_PATH,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict[str, int]:
    """Materialize Parquet data and event-time joins into a DuckDB file."""
    data_dir = data_dir.resolve()
    database_path = database_path.resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    paths = _source_paths(data_dir)

    connection = duckdb.connect(str(database_path))
    try:
        connection.execute("SET TimeZone='UTC'")
        connection.execute("BEGIN TRANSACTION")

        _load_us_price_table(
            connection, "skhy_5min", paths["skhy"], include_volume=True
        )
        _load_us_price_table(
            connection, "spy_5min", paths["spy"], include_volume=False
        )
        _load_us_price_table(
            connection, "soxx_5min", paths["soxx"], include_volume=False
        )
        _load_us_price_table(
            connection, "qqq_5min", paths["qqq"], include_volume=False
        )
        _load_us_price_table(
            connection, "usdkrw_5min", paths["usdkrw"], include_volume=False
        )
        _load_korean_daily_table(connection, paths["korea"])
        quotes_loaded = _load_quotes_table(
            connection, data_dir / OPTIONAL_QUOTES_FILE
        )

        _create_event_anchors(connection)
        _create_korean_asof_matches(connection)
        _create_indexes(connection)
        _write_metadata(
            connection, data_dir, database_path, quotes_loaded=quotes_loaded
        )
        connection.execute("COMMIT")
        connection.execute("CHECKPOINT")

        tables = (
            "skhy_5min",
            "spy_5min",
            "soxx_5min",
            "qqq_5min",
            "usdkrw_5min",
            "korea_daily",
            "skhy_quotes",
            "us_event_anchors",
            "event_korea_asof",
        )
        return {
            table: connection.execute(
                f"SELECT count(*) FROM {table}"
            ).fetchone()[0]
            for table in tables
        }
    except Exception:
        try:
            connection.execute("ROLLBACK")
        except duckdb.Error:
            pass
        raise
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help=f"Persistent DuckDB path (default: {DEFAULT_DATABASE_PATH})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Parquet data directory (default: {DEFAULT_DATA_DIR})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counts = build_database(args.database, args.data_dir)
    print(f"DuckDB saved to: {args.database.resolve()}")
    for table, count in counts.items():
        print(f"{table:24} {count:>10,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
