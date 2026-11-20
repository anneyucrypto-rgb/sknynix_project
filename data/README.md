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

The retained overnight run materializes these linked artifacts:

| Artifact | Purpose |
| --- | --- |
| `features/daily/part-00000.parquet` | Reusable paired daily data used to estimate the retained H1 equilibrium |
| `features/overnight_h2/part-00000.parquet` | Standalone H2 input including point-in-time FX |
| `features/overnight/part-00000.parquet` | Standalone Chronos-2 target and covariates with execution fields |
| `features/experiments/run_id=overnight-best/overnight.parquet` | One date per close-to-open target with previous-close ECT inputs, Korean return, two SOXX gauges, and executable prices |
| `features/experiments/run_id=overnight-best/h1_model.parquet` | Retained bivariate H1 estimates |
| `features/experiments/run_id=overnight-best/h2_model.parquet` | Retained Korean-return/ECT/SOXX-gauge H2 estimates |
| `features/experiments/run_id=overnight-best/overnight_backtest.parquet` | Retained H2 expanding-window predictions and executable fields |
| `features/experiments/run_id=overnight-best/premarket_to_open_strategy.parquet` | Net 09:25–09:30 strategy path |

Build the standalone feature tables with `python -m src.features.h1_h2` and
`python -m src.features.chronos_2`. Model estimates, forecasts, strategy paths,
and machine-readable summaries are Parquet tables under `data/features/`.
Final Markdown reports go in the repository's `reports/` directory.

## Current raw-data coverage

Inventory date: 2026-11-19. Coverage is taken from Parquet metadata.

| Dataset | Rows | Coverage |
| --- | ---: | --- |
| SKHY trades | 35,437,266 | 2026-07-13 through 2026-11-11 |
| SKHY quotes | 22,387,191 | 2026-07-13 through 2026-11-11 |
| SKHY one-minute bars | 80,317 | 2026-07-13 through 2026-11-17 |
| SKHY five-minute bars | 17,236 | 2026-07-13 through 2026-11-17 |
| SKHY scored news events | 144 | 2026-07-14 through 2026-11-17 |
| SKHY one-minute sentiment buckets | 143 | 2026-07-14 through 2026-11-17 |
| SKHY five-minute sentiment buckets | 142 | 2026-07-14 through 2026-11-17 |
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
python -m src.ingestion.massive_news --start-date 2026-07-13 --end-date 2026-11-18
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
| `src.ingestion.massive_news` | Massive ticker news insights | SKHY article-level LLM sentiment and sparse one-/five-minute buckets |
| `src.ingestion.korean_daily` | Yahoo Finance | 000660, KOSPI, KOSDAQ, and Korean semiconductor ETF proxy |
| `src.ingestion.fx_data` | Massive forex aggregates | USD/KRW one- and five-minute bars |
| `src.ingestion.economic_data` | Massive Economy and market aggregates | U.S. macro data and labeled Korean market proxies |

Ticker-news sentiment is mapped as `positive = 1`, `neutral = 0`, and
`negative = -1`. The primary normalized bucket score is the bounded net
article balance `(positive_count - negative_count) / article_count`; it is
zero for neutral-only or empty inputs. The bucket also retains a
Jeffreys-smoothed positive/negative log-odds score and its bounded hyperbolic-
tangent transform, along with the sentiment sum, label counts, article count,
and article IDs. Minutes without news are absent and should be filled with zero
only when joining the sparse signal to a complete price grid. Article records retain both `published_utc` and
`received_at_utc` because Massive does not expose the historical time at which
an article first became available through its hourly-updated endpoint.

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
