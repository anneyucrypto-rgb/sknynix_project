# Code sequence and responsibilities

## Recommended execution sequence

| Step | Command | Input | Output | Purpose |
|---|---|---|---|---|
| 1 | `python ingestion/download.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --pairs SKHY --full-ticks --skip-controls` | Massive API and Yahoo Finance | Raw SKHY bars, daily data, first-event records, and full quotes/trades | Download the research pair. `--full-ticks` is required for dynamic premarket-anchor selection. |
| 2 | `python ingestion/features.py --controls --start-date YYYY-MM-DD --end-date YYYY-MM-DD` | Massive API, Yahoo Finance, FRED, and raw SKHY data | All control files plus `data/processed/granger_panel.parquet` and `data/processed/arbitrage_panel_5min.parquet` | Refresh U.S. market/FX controls, U.S. macro, Korean daily controls, Korean five-minute benchmarks and signals; then build both processed panels. |
| 3 | `python models/feature_analysis.py` | Granger panel | `design/macro_multicollinearity.md` | Diagnose feature- and group-level macro multicollinearity before interpreting the macro-control block. |
| 4 | `python models/granger.py` | Granger panel | `models/results/granger_ablation_results.parquet`, `design/model.md`, `design/model_summary.md` | Estimate the mixed-frequency event-time ARDL/Granger models and control-group ablations. |
| 5 | `python ingestion/test.py` | Raw and processed market data | Console audit report | Check data freshness and coverage, then list curated and directly discoverable Korean indices that can serve as SKHY macro signals. |
| 6 | Backtest/application commands as implemented | Model outputs and processed panels | Strategy/application outputs | Consume validated research results; these modules are currently placeholders. |

Run steps 2--4 again whenever the raw-data window, anchor thresholds, horizons,
or feature definitions change.

## Anchor and horizon implementation

`ingestion/features.py` creates exactly two conceptual anchor records per U.S.
trading day:

- `premarket_dynamic`: earliest eligible five-minute decision timestamp from
  07:00 through 09:20 ET. Selection uses only the latest observable quote and
  completed-bar volume. With the defaults, quoted spread must be at most 100
  bps, combined displayed depth at least 100 size units, quote age at most 60
  seconds, and volume during the previous five minutes positive. If full quote
  ticks are unavailable or no candidate qualifies, the row is retained with
  `anchor_available = False` and a reason; no time or price is invented.
- `regular_0930`: fixed 09:30 ET research reference. The exact bar open is a
  statistical reference price, while first-trade and first-quote fields remain
  separate execution diagnostics rather than assumed fills.

Both anchors use horizons `{5, 10, 15, 30, 45, 60, 100, 120}` minutes. This is
16 conceptual anchor--horizon observations per trading day before missing-data
and execution-availability filtering.

## File-by-file functionality

### Root files

| File | Definition and functionality |
|---|---|
| `app.py` | Application entry-point placeholder. It does not currently orchestrate the research pipeline. |
| `config.py` | Project-configuration placeholder for future shared settings. |

### Ingestion and feature engineering

| File | Definition and functionality |
|---|---|
| `ingestion/download.py` | Main raw-data downloader. Defines supported Korean/U.S. research pairs; downloads five-minute bars, Korean daily data, KOSPI, VIX, USD/KRW, and Treasury yields; records first trades/quotes; and optionally downloads full tick streams. Its event schedule starts dynamic-premarket observation at 07:00 and retains the fixed 09:30 event. |
| `ingestion/massive_client.py` | Project-specific Massive REST wrapper. Handles authenticated requests, pagination/retries, aggregate-bar downloads, first-event lookups, and full historical trade/quote streaming to Parquet. |
| `ingestion/features.py` | Control downloader and central feature builder. With `--controls`, downloads U.S. market/FX bars, VIX, Treasury yields, Korean daily controls, five-minute KOSPI/KOSPI 200, labeled KOSDAQ 150/KRX 300 tracking proxies, KOSDAQ, and semiconductor signals. It then validates raw data; selects the dynamic premarket anchor; records unavailable anchors; creates model fields and eight forward-return horizons; validates identities; and writes both processed panels atomically. |
| `ingestion/read_data.py` | Convenience script that loads and displays the two processed Parquet panels for inspection. |
| `ingestion/test.py` | Data-audit utility that reports raw/processed coverage and freshness and lists curated and directly discoverable Korean indices for SKHY macro-signal research. It is not a complete unit-test suite. |
| `ingestion/korea.py` | Placeholder for additional Korean-market ingestion logic. Current Korean daily downloads are handled by `download.py`. |

### Statistical models

| File | Definition and functionality |
|---|---|
| `models/feature_analysis.py` | Evaluates macro-control multicollinearity separately by anchor. Reports feature VIF, PCA-based conceptual-group VIF, condition numbers, Pearson correlations, and an iterative VIF reduction that retains at least one feature per group. Writes one combined Markdown report. |
| `models/granger.py` | Estimates mixed-frequency event-time ARDL models separately for each anchor and horizon. Selects AR and Korean lag orders by BIC or AIC, uses HAC inference, performs a joint Wald test on Korean-return terms, and compares baseline, macro, sector, liquidity, and microstructure control blocks on common samples. Unavailable dynamic anchors produce explicit unavailable result cells instead of fabricated estimates. |
| `models/arbitrage.py` | Placeholder for arbitrage models and signal generation. The feature panel it will consume is created by `features.py`. |

### Backtesting

| File | Definition and functionality |
|---|---|
| `backtest/backtest.py` | Placeholder for strategy backtesting. Execution assumptions should use executable bid/ask or trade data and must not treat research reference bars as guaranteed fills. |

## Key generated artifacts

| Artifact | Producer | Meaning |
|---|---|---|
| `data/processed/granger_panel.parquet` | `ingestion/features.py` | One wide row per trading day and anchor, with availability state, predictors, eight forward responses, and lagged responses. |
| `data/processed/arbitrage_panel_5min.parquet` | `ingestion/features.py` | Five-minute parity, premium, residual, session, and event-timestamp panel. |
| `models/results/granger_ablation_results.parquet` | `models/granger.py` | Machine-readable ARDL estimates, HAC tests, lag orders, availability states, and ablation comparisons. |
| `design/model.md` | `models/granger.py` | Current model and event-anchor design. |
| `design/model_summary.md` | `models/granger.py` | Human-readable results summary generated from the latest estimates. |
| `design/macro_multicollinearity.md` | `models/feature_analysis.py` | Combined macro-feature and macro-group multicollinearity report. |

## Reproducibility rules

- Freeze liquidity thresholds before estimating returns; do not choose the
  dynamic anchor by realized profitability.
- Require full quote ticks for `premarket_dynamic`. A first quote alone cannot
  prove that later candidates were evaluated consistently.
- Use only completed volume windows and quotes timestamped at or before the
  candidate decision time.
- Keep the fixed 09:30 research reference distinct from an executable fill.
- Rebuild the processed panels before rerunning either analysis whenever an
  anchor or horizon definition changes.
