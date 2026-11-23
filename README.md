# SKHY–Korean Stock Cointegration Research Platform

This project builds reproducible datasets for testing long-run equilibrium,
error correction, and directional price discovery between SKHY and SK hynix's
Korean ordinary shares (000660).

## Research datasets

| Dataset | Grain | Main consumer |
| --- | --- | --- |
| `daily` | Paired U.S./Korean trading date | H1, H2, H3 |
| `us_5min` | Completed SKHY five-minute bar | reusable market-quality features |
| `h1` | Eligible daily observation | daily ARDL--UECM equilibrium |
| `h2` | Eligible daily observation | ablation-selected controlled regression |
| `h3` | Directional target observation | directional ECM |

All timestamps are stored in UTC. Session dates retain their market meaning,
and as-of joins only use observations available at the modeled information
cutoff. Korean ETF and currency series are explicitly labelled as market
proxies; they are not represented as direct Bank of Korea macro data.

## Build features

Install the dependencies in `requirements.txt`, then run the feature builders
from the project root.

```powershell
python -m src.features.h1_h2
python -m src.features.chronos_2
```

The H1/H2 builder writes the paired daily and overnight H2 input tables.
The Chronos-2 builder writes a dedicated experiment table whose target is the
SKHY previous-close-to-open return and whose known daily covariates are the
FX-adjusted Korean ordinary-share return, VIX return, FLKR return, and USD/KRW
return. Shared transformations and validation live in
[`src/features/utils.py`](src/features/utils.py).

Reusable outputs are written under `data/features/<dataset>/`. Experiment
datasets are isolated under
`data/features/experiments/run_id=<spec-fingerprint>/`; pass an explicit
`run_id` to `build()` to associate them with a model-registry run.
Build status is printed as each dependency starts, completes, or is restored
from cache. The cache key incorporates the feature version, experiment
specification, and immutable raw-snapshot content.

## Code map

Package `__init__.py` markers are omitted because they contain no executable
workflow. Every other Python module is described below.

### Feature, model, and backtest code

| Module | Function |
| --- | --- |
| [`src/features/h1_h2.py`](src/features/h1_h2.py) | Constructs the point-in-time daily panel: SKHY close-to-open target, FX-adjusted Korean level and return, and every daily control candidate. It does not estimate H1 or H2. |
| [`src/features/chronos_2.py`](src/features/chronos_2.py) | Selects the Chronos-2 target and four daily covariates from the H1/H2 panel and writes the dedicated Chronos experiment input. |
| [`src/features/utils.py`](src/features/utils.py) | Provides shared Parquet readers, timestamp normalization, validation, atomic feature writes, and legacy execution-feature calculations. |
| [`src/hypotheses/h1.py`](src/hypotheses/h1.py) | Estimates the daily ARDL--UECM, performs bounds and bootstrap inference, constructs the ECT, and evaluates rolling stability. |
| [`src/hypotheses/overnight_ablation.py`](src/hypotheses/overnight_ablation.py) | Imports H1, reconstructs the ECT, evaluates H2 control ablations, applies validity filters, performs nested p/q and control selection, and writes model/audit artifacts. |
| [`backtest/backtest.py`](backtest/backtest.py) | Reproduces the retained H1/H2 expanding-window forecasts from a materialized ablation run. |
| [`backtest/chronos_2.py`](backtest/chronos_2.py) | Runs causal expanding-window Chronos-2 LoRA fine-tuning and one-step quantile forecasts. |
| [`backtest/forecast_comparison.py`](backtest/forecast_comparison.py) | Builds the fixed VIX-only H1/H2 benchmark and compares it with Chronos-2 on identical forecast dates. |
| [`backtest/strategy.py`](backtest/strategy.py) | Applies the shared 09:25–09:30 execution, calibration, cost, sizing, stop, and drawdown rules to one or more forecast files. |

### Data ingestion and configuration

| Module | Function |
| --- | --- |
| [`src/config.py`](src/config.py) | Defines project paths, source settings, output-path safeguards, and local environment-secret loading. |
| [`src/ingestion/massive_us.py`](src/ingestion/massive_us.py) | Incrementally downloads U.S. bars, SKHY trades, and SKHY quotes from Massive and merges them into deduplicated Parquet snapshots. |
| [`src/ingestion/massive_news.py`](src/ingestion/massive_news.py) | Downloads SKHY news sentiment and creates timestamped minute sentiment buckets. |
| [`src/ingestion/korean_daily.py`](src/ingestion/korean_daily.py) | Downloads daily SK hynix ordinary-share and Korean benchmark OHLCV data. |
| [`src/ingestion/fx_data.py`](src/ingestion/fx_data.py) | Incrementally downloads USD/KRW one- and five-minute bars. |
| [`src/ingestion/economic_data.py`](src/ingestion/economic_data.py) | Downloads U.S. Treasury, inflation, and funding data plus explicitly labelled Korean market proxies. |

### Reproducibility engine and application entry point

| Module | Function |
| --- | --- |
| [`src/engine/dataset_builder.py`](src/engine/dataset_builder.py) | Orchestrates point-in-time dataset builds, dependency resolution, cache keys, validation, and artifact publication. |
| [`src/engine/feature_registry.py`](src/engine/feature_registry.py) | Declares canonical dataset grains, dependencies, builders, postprocessors, and feature versions. |
| [`src/engine/control_registry.py`](src/engine/control_registry.py) | Declares stable control metadata, joining rules, coverage requirements, and proxy labels. |
| [`src/engine/dependency_resolver.py`](src/engine/dependency_resolver.py) | Topologically orders feature dependencies and identifies downstream invalidation. |
| [`src/engine/cache_manager.py`](src/engine/cache_manager.py) | Hashes source snapshots and maintains content-addressed feature-cache metadata. |
| [`src/engine/duckdb_engine.py`](src/engine/duckdb_engine.py) | Wraps DuckDB connections, transactions, identifier validation, and query helpers. |
| [`src/engine/sql_builder.py`](src/engine/sql_builder.py) | Loads, validates, renders, and hashes strict SQL templates. |
| [`src/engine/experiment.py`](src/engine/experiment.py) | Defines validated, serializable, and hashable experiment specifications. |
| [`src/engine/model_registry.py`](src/engine/model_registry.py) | Stores experiment metadata, coefficients, and diagnostics in DuckDB. |
| [`app.py`](app.py) | Reserved application entry point; it currently contains no runtime workflow. |

### Automated tests

| Test module | Coverage |
| --- | --- |
| [`tests/test_overnight_ablation.py`](tests/test_overnight_ablation.py) | H1/H2 estimation, eligibility filters, nested selection, and proxy-redundancy analysis. |
| [`tests/test_forecast_comparison.py`](tests/test_forecast_comparison.py) | Chronos variable contract and common-date model comparison. |
| [`tests/test_overnight_strategy.py`](tests/test_overnight_strategy.py) | Calibration, execution rules, costs, stops, and portfolio safeguards. |
| [`tests/test_price_parity.py`](tests/test_price_parity.py) | FX-adjusted parity and price-feature calculations. |
| [`tests/test_massive_news.py`](tests/test_massive_news.py) | News ingestion, sentiment normalization, and bucketing. |
| [`tests/test_incremental_massive.py`](tests/test_incremental_massive.py) | Incremental source coverage, range detection, and snapshot merging. |
| [`tests/test_engine_specs.py`](tests/test_engine_specs.py) | Experiment, registry, dependency, and builder contracts. |
| [`tests/test_duckdb_quote_buckets.py`](tests/test_duckdb_quote_buckets.py) | DuckDB quote aggregation and time-bucket behavior. |

H1 estimates the long-run level relation between the prior SKHY close and the
latest FX-adjusted Korean ordinary-share close using a daily ARDL--UECM. It
derives the equilibrium slope from `-rho_k/rho_a`, tests the level terms with
ARDL bounds and bootstrap inference, and reports rolling stability. The full
specification is in [`design/h1.md`](design/h1.md). H2 includes its pre-response
equilibrium error. At every forecast origin, H1 and the ECT are re-estimated
from prior rows, BIC selects p/q within each control specification, and models
must have \(-2<\lambda<0\), significant lambda, and positive significant Korean
return transmission. Candidate groups are SOXX, normalized SKHY sentiment, Korean
semiconductor/KOSDAQ/KOSPI returns, Korean market proxies, U.S. inflation,
funding conditions, VIX returns, and Treasury changes. Eligible models are
ranked by inner expanding-window RMSE, and the simplest model within one
standard error of the best is selected. No intraday anchor-time feature enters.

## Retained overnight-return model

`src.hypotheses.overnight_ablation` forecasts SKHY's previous-close-to-09:30
open log return. H1 supplies the equilibrium benchmark; H2 produces the return
forecast. The study compares each control group alone, the all-control model,
and leave-one-group-out models. Forecast diagnostics use expanding-window
estimation with earlier dates only but do not determine the Korean-return ranking.

The executable `premarket_to_open` strategy is narrower than the H2 target. It
enters at the 09:25 premarket price and liquidates at the 09:30 opening auction
proxy. A prior-only calibration maps the full-gap H2 forecast to the expected
remaining five-minute return. Cost hurdles, volatility-scaled sizing,
stop-loss/take-profit levels, and portfolio drawdown limits are applied before
performance is reported.

Run:

```text
python -m src.hypotheses.overnight_ablation --run-id overnight-daily-ablation --max-p 3 --max-q 3 --min-train 40
python -m backtest.backtest --run-id overnight-daily-ablation --h1-spec daily_ardl_uecm --h2-spec ablation_selected --min-train 40
```

The forecasting design and retained H1 and H2 estimates are documented in
[`design/OVERNIGHT_FORECAST_DESIGN.md`](design/OVERNIGHT_FORECAST_DESIGN.md).
The implementation, nested-selection workflow, artifacts, and commands are
documented in [`design/H1_H2_CODE_GUIDE.md`](design/H1_H2_CODE_GUIDE.md).
The execution rules are documented in
[`design/BACKTEST_DESIGN.md`](design/BACKTEST_DESIGN.md).
The current common-date Chronos-2/H1-H2 benchmark is in
[`reports/CHRONOS_H2_COMPARISON.md`](reports/CHRONOS_H2_COMPARISON.md), and its
shared strategy evaluation is in
[`reports/forecast_strategies_summary.md`](reports/forecast_strategies_summary.md).
Parquet model, forecast, and strategy artifacts are stored under
`data/features/experiments/`. The `reports/` directory holds Markdown
results. See [run_commands.txt](run_commands.txt) for the Chronos-2 forecast
and multi-model comparison commands.

See [data/README.md](data/README.md) for the storage layout and
[data/MISSING_DATA.md](data/MISSING_DATA.md) for current source coverage.
