# Missing, limited, and unbuilt datasets

Inventory date: 2026-11-19

Previously missing downloadable controls have now been populated. The retained
overnight feature dataset is implemented and materialized.
Remaining gaps are raw-data coverage, limited sample size, point-in-time
release metadata, unavailable direct Korean macro data, and the unbuilt
standardized core layer.

## Remaining raw-data gaps

| Gap | Current evidence | Required action |
| --- | --- | --- |
| SKHY tick coverage after 2026-11-11 | 35,437,266 trades and 22,387,191 quotes end on 2026-11-11; minute bars continue into 2026-11-17 UTC | Run the incremental `massive_us` command with `--trades --quotes`; it will request only the uncovered tail |
| Long-run SKHY history | One- and five-minute bars begin on 2026-07-13 | Confirm Massive's earliest SKHY availability and assess whether the short common sample supports H1 |
| Overlapping FX history | USD/KRW covers 2024-11-17 through 2026-11-16 UTC, while 000660 and controls begin in 2020 | Enforce the actual SKHY–000660–FX intersection in coverage diagnostics |
| U.S. inflation release timing | The monthly inflation file has an observation `date` but no `release_timestamp` | Add a verified release timestamp before using CPI/PCE in intraday point-in-time models |
| Optional Korean underlying intraday data | Only daily 000660 is part of the baseline design | Acquire only if a future intraday Korean specification is activated |

SKHY ticker-news sentiment is present through 2026-11-17. Massive updates the
endpoint hourly and does not provide a historical ingestion timestamp, so
`published_utc` must not be interpreted as proof that the LLM insight was
tradable at that exact time. New downloads preserve `received_at_utc` for
forward use.

The most important issue is statistical rather than a missing filename: the
current H1 common window begins on 2026-07-13. A short sample may be unsuitable
for a claim of stable long-run cointegration even when every required column is
present.

Incremental tick refresh command:

```powershell
python -m src.ingestion.massive_us --start-date 2026-06-01 --end-date 2026-11-17 --trades --quotes
```

Use `--refresh-all` only when every selected snapshot should be replaced.

## Direct Korean macro data unavailable from Massive

| Direct dataset | Status |
| --- | --- |
| Korean 3Y and 10Y government yields | Not supplied by Massive |
| Korean CPI | Not supplied by Massive |
| Bank of Korea base rate | Not supplied by Massive |

The populated replacement is
`raw/korea/economic/proxies/KR_MARKET_PROXIES_daily/part-00000.parquet`, which
contains `EWY`, `FLKR`, and `C:USDKRW`. It is a market-expectations control,
not a reconstruction of the unavailable macro series. Direct observations may
be added later from a licensed source without overwriting the proxy dataset.

## Raw controls now present

| Dataset | Path | Status |
| --- | --- | --- |
| U.S. 2Y/10Y Treasury yields | `raw/us/economic/treasury/US_TREASURY_daily/part-00000.parquet` | Present |
| U.S. CPI/PCE inflation | `raw/us/economic/inflation/US_INFLATION_monthly/part-00000.parquet` | Values present; release timestamp missing |
| U.S. funding and policy conditions | `raw/us/economic/policy/US_FUNDING_CONDITIONS_daily/part-00000.parquet` | Present |
| VIX daily risk control | `raw/us/economic/risk/VIX_daily/part-00000.parquet` | Present, legacy schema |
| Korean market-implied macro proxies | `raw/korea/economic/proxies/KR_MARKET_PROXIES_daily/part-00000.parquet` | Present |
| KOSPI daily control | `raw/korea/market/sector_daily/KOSPI_daily/part-00000.parquet` | Present |
| KOSDAQ daily control | `raw/korea/market/sector_daily/KOSDAQ_daily/part-00000.parquet` | Present |
| Korean semiconductor daily proxy | `raw/korea/market/sector_daily/KOREA_SEMICONDUCTOR_daily/part-00000.parquet` | Present |

## Standardized datasets not yet materialized

The directories exist, but no standardized Parquet output has been built.
The current feature builder reads the typed raw snapshots directly, so this
does not block a baseline feature build; the core layer remains required for
the complete target architecture.

| Unbuilt dataset | Expected path | Needed by |
| --- | --- | --- |
| Standardized U.S. trades | `core/us_trades/` | Optional future microstructure analysis |
| Standardized U.S. quotes | `core/us_quotes/` | Optional future microstructure analysis |
| Standardized U.S. minute prices | `core/us_prices_1min/` | H1 aggregation and H2/H3 returns |
| Standardized Korean daily prices | `core/kr_daily/` | H1–H3 |
| Standardized FX observations | `core/fx_quotes/` | Parity and FX-adjusted Korean prices |
| Standardized economic observations | `core/economic_observations/` | Conditional H2/H3 specifications |

## Materialized retained-model datasets

| Dataset | Path | Current status |
| --- | --- | --- |
| H1/H2 daily input | `features/daily/part-00000.parquet` | Present: 90 close-to-open targets with FX-adjusted Korean returns and daily control candidates |
| Retained overnight model data | `features/experiments/run_id=overnight-daily-ablation/overnight.parquet` | Present: unified daily model panel without anchor variables |
| H1 estimates | `features/experiments/run_id=overnight-daily-ablation/h1_model.parquet` | Present: long-run daily level equilibrium used to construct ECT |
| Ablation-selected H2 | `features/experiments/run_id=overnight-daily-ablation/h2_model.parquet` | Present: final-origin nested VIX specification and negative significant ECT lambda |
| Control ablation | `features/experiments/run_id=overnight-daily-ablation/control_ablation.parquet` | Present: full-sample coefficient comparison for 21 H2 specifications |
| Nested selection audit | `features/experiments/run_id=overnight-daily-ablation/nested_selection_audit.parquet` | Present: origin-by-origin eligibility, inner RMSE, one-SE set, and selected model |
| Korean proxy redundancy | `features/experiments/run_id=overnight-daily-ablation/proxy_redundancy_analysis.parquet` | Present: FLKR + USD/KRW retained; EWY excluded as duplicative |
| Expanding-window forecasts | `features/experiments/run_id=overnight-daily-ablation/overnight_backtest.parquet` | Present: 46 fully nested forecasts after 40 initial training observations |
| Premarket strategy path | `features/experiments/run_id=overnight-best/premarket_to_open_strategy.parquet` | Present: 09:25 entry, minute-level stop/take-profit checks, and 09:30 auction exit proxy |

The common p/q-order comparison has 86 complete dates. Earlier experimental model
artifacts have been removed; raw vendor snapshots remain available for future
research without entering the retained H1/H2 specification.

## Validation required before estimation

1. Determine the common SKHY–000660–USD/KRW daily window and justify its sample
   size for H1.
2. Confirm exchange calendars, daylight-saving transitions, and Korea-to-U.S.
   session mapping.
3. Verify the SOXX previous-close, 07:00, and 09:25 endpoints after every data
   refresh.
4. Continue genuinely forward validation as new SKHY sessions arrive; the H1
   equilibrium and selected specification were chosen on the historical sample.
5. Report missingness and observation counts after every refresh before
   comparing forecast accuracy with the retained 48-forecast sample.
