# Missing, limited, and unbuilt datasets

Inventory date: 2026-11-17

Previously missing downloadable controls have now been populated. The engine
and all six baseline feature datasets are implemented and materialized.
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
| Standardized U.S. trades | `core/us_trades/` | H2 liquidity and execution checks |
| Standardized U.S. quotes | `core/us_quotes/` | H2 dynamic anchor selection |
| Standardized U.S. minute prices | `core/us_prices_1min/` | H1 aggregation and H2/H3 returns |
| Standardized Korean daily prices | `core/kr_daily/` | H1–H3 |
| Standardized FX observations | `core/fx_quotes/` | Parity and FX-adjusted Korean prices |
| Standardized economic observations | `core/economic_observations/` | Conditional H2/H3 specifications |

## Materialized baseline feature datasets

The forced `baseline-default` build completed successfully. These counts are
the current materialized inventory, not claims of econometric adequacy.

| Dataset | Path | Current status |
| --- | --- | --- |
| Daily parity and controls | `features/daily/part-00000.parquet` | Present: 90 rows |
| U.S. five-minute liquidity | `features/us_5min/part-00000.parquet` | Present: 17,236 rows |
| Dynamic and fixed anchors | `features/anchors/part-00000.parquet` | Present: 180 rows; 174 valid |
| H1 model data | `features/experiments/run_id=baseline-default/h1.parquet` | Present: 90 rows; 88 sample-eligible |
| H2 anchor × horizon data | `features/experiments/run_id=baseline-default/h2.parquet` | Present: 1,392 rows; 1,387 valid outcomes |
| Normalized directional H3 data | `features/experiments/run_id=baseline-default/h3.parquet` | Present: 1,482 rows; 1,477 valid outcomes |

Anchor coverage consists of 87 valid dynamic premarket anchors and 87 valid
09:30 anchors. Three dates have missing premarket quotes and three corresponding
fixed-opening observations are stale or invalid. H3 contains 1,392
Korea-to-U.S. observations and 90 U.S.-to-Korea observations.

The baseline feature command in `run_commands.txt` uses `force=True`, so it
rebuilds these outputs even when their content-addressed caches are valid.

`features/experiments/run_id=legacy/` contains earlier Granger and arbitrage
panels. They are preserved but are not substitutes for the current H1–H3
datasets.

## Validation required before estimation

1. Determine the common SKHY–000660–USD/KRW daily window and justify its sample
   size for H1.
2. Confirm exchange calendars, daylight-saving transitions, and Korea-to-U.S.
   session mapping.
3. Require valid quotes/trades, spread and depth thresholds, and the
   7:00–9:20 a.m. ET search window for every dynamic anchor.
4. Add verified release timestamps for event-driven U.S. macro controls and
   prevent look-ahead joins. Until then, CPI/PCE must remain excluded from
   intraday experiments.
5. Report missingness and observation counts for every requested horizon.
6. Keep Korean proxy coefficients labeled as market-proxy effects, not direct
   macroeconomic effects.
