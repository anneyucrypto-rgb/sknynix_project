# Research data

This directory contains the Parquet and DuckDB inputs for the SKHY–SK hynix
cointegration, error-correction, and directional price-discovery study.

## Storage convention

Every logical Parquet dataset has its own directory. Its current snapshot is
stored as `part-00000.parquet`:

```text
data/raw/us/market/SKHY_1min/part-00000.parquet
```

Ingestion writes to a temporary `.part` file and atomically replaces the
destination after a complete write. Research runs should record source-file
hashes so a later refresh cannot silently change an experiment.

| Directory | Role |
| --- | --- |
| `raw/` | Source observations with minimal normalization |
| `core/` | Standardized types, timestamps, identifiers, and schemas |
| `features/` | Reusable daily, intraday, parity, liquidity, and anchor features |
| `features/experiments/run_id=.../` | Versioned model-ready datasets |
| `research.duckdb` | DuckDB catalog and SQL query engine |

Directories named `legacy_*` and `features/experiments/run_id=legacy/` belong
to the earlier multi-company Granger design. They are retained for provenance,
not as H1–H3 inputs.

## Current raw-data coverage

Inventory date: 2026-11-17. Coverage is taken from Parquet metadata.

| Dataset | Rows | Coverage |
| --- | ---: | --- |
| SKHY trades | 35,437,266 | 2026-07-13 through 2026-11-11 |
| SKHY quotes | 22,387,191 | 2026-07-13 through 2026-11-11 |
| SKHY one-minute bars | 80,317 | 2026-07-13 through 2026-11-17 |
| SKHY five-minute bars | 17,236 | 2026-07-13 through 2026-11-17 |
| SPY one-minute bars | 1,449,720 | 2020-01-02 through 2026-11-17 |
| SOXX one-minute bars | 786,053 | 2020-01-02 through 2026-11-17 |
| 000660 daily prices | 1,684 | 2020-01-02 through 2026-11-17 |
| USD/KRW one-minute bars | 672,832 | 2024-11-17 through 2026-11-16 |
| U.S. Treasury yields | 1,718 | 2020-01-02 through 2026-11-13 |
| U.S. inflation | 82 | 2020-01 through 2026-10 |
| U.S. funding conditions | 2,513 | 2020-01-01 through 2026-11-17 |
| Korean market proxies | 4,079 | 2020-01-02 through 2026-11-16 |
| Korean semiconductor daily proxy | 1,662 | 2020-01-02 through 2026-11-17 |

The common H1 window is constrained by SKHY and currently begins on
2026-07-13. File presence alone does not establish an adequate sample for a
long-run ARDL–UECM bounds test; see [MISSING_DATA.md](MISSING_DATA.md).

## Ingestion

Run all commands from the repository root:

```powershell
python -m src.ingestion.massive_us --start-date 2020-01-01 --end-date 2026-11-17
python -m src.ingestion.massive_us --start-date 2026-07-01 --end-date 2026-11-17 --trades --quotes
python -m src.ingestion.korean_daily --start-date 2020-01-01 --end-date 2026-11-17
python -m src.ingestion.fx_data --start-date 2020-01-01 --end-date 2026-11-17
python -m src.ingestion.economic_data --start-date 2020-01-01 --end-date 2026-11-17
```

Do not request unnecessary multi-year tick history: trade and quote files can
be extremely large. Every ingestion module inspects its target Parquet
snapshot and requests only uncovered leading or trailing date intervals.
Market and FX bars are deduplicated by ticker and timestamp, Korean daily data
by ticker and date, and U.S. economic observations by date. Combined Korean
proxy coverage is checked separately for each ticker. New tick segments are
streamed into the existing snapshot without loading the full trade or quote
history into pandas. Covered intervals are skipped. `--force` and its clearer
alias `--refresh-all` deliberately replace selected snapshots instead of
applying incremental behavior.

All ingestion commands print:

- dataset-level `[current/total]` status;
- `DOWNLOADING`, `SKIPPED`, and `DONE` states;
- cumulative pages and rows for paginated Massive responses;
- retry delays for rate limits, server errors, and network failures; and
- final row counts and output paths.

| Module | Source | Output scope |
| --- | --- | --- |
| `src.ingestion.massive_us` | Massive aggregates/trades/quotes | SKHY, SPY, and SOXX bars; optional SKHY ticks |
| `src.ingestion.korean_daily` | Yahoo Finance | 000660, KOSPI, KOSDAQ, and Korean semiconductor ETF proxy |
| `src.ingestion.fx_data` | Massive forex aggregates | USD/KRW one- and five-minute bars |
| `src.ingestion.economic_data` | Massive Economy and market aggregates | U.S. macro data and labeled Korean market proxies |

## Authentication and configuration

Only `MASSIVE_API_KEY` is required for Massive-backed modules. It can be set in
the environment or in the project-root `.env` file:

```dotenv
MASSIVE_API_KEY="your_key"  # comments outside quotes are supported
```

The `.env` file must remain local. If it has already been tracked by Git,
remove it from the index/history and rotate the exposed credential before
sharing the repository. Configuration is defined by Python dataclasses in
`src/config.py`.

## Korean control limitation

Massive does not provide Korean CPI, Bank of Korea policy rates, Korean
government-bond yields, or Korean-exchange listings. The project therefore
stores these market-implied controls in
`raw/korea/economic/proxies/KR_MARKET_PROXIES_daily/`:

| Ticker | Role |
| --- | --- |
| `EWY` | Korean country-equity risk proxy |
| `FLKR` | Alternative Korean country-equity proxy for robustness |
| `C:USDKRW` | Currency and policy-expectations proxy |

Every row is marked `is_direct_korean_macro_measure = false` and carries an
interpretation warning. These series must not be described as Korean CPI,
policy-rate, or government-yield observations.
