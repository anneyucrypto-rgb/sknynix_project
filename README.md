# SKHY–Korean Stock Cointegration Research Platform

This project builds reproducible datasets for testing long-run equilibrium,
error correction, and directional price discovery between SKHY and SK hynix's
Korean ordinary shares (000660).

## Research datasets

| Dataset | Grain | Main consumer |
| --- | --- | --- |
| `daily` | Paired U.S. trading session | H1, H3 |
| `us_5min` | Completed SKHY five-minute bar | reusable market-quality features |
| `h1` | Eligible daily observation | ARDL–UECM |
| `h2` | One U.S. session observed at 09:25 ET | retained close-to-open SOXX-gauge forecast |
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
The Chronos-2 builder writes a standalone overnight table from the original
market inputs. Shared transformations and validation live in
[`src/features/utils.py`](src/features/utils.py).

Reusable outputs are written under `data/features/<dataset>/`. Experiment
datasets are isolated under
`data/features/experiments/run_id=<spec-fingerprint>/`; pass an explicit
`run_id` to `build()` to associate them with a model-registry run.
Build status is printed as each dependency starts, completes, or is restored
from cache. The cache key incorporates the feature version, experiment
specification, and immutable raw-snapshot content.

The retained H1 is a bivariate daily ARDL–UECM. It constructs the equilibrium
error from the previous SKHY regular-session close and the latest FX-adjusted
Korean close. The only retained H2 combines that ECT with the Korean daytime
return, the SOXX previous-close-to-07:00 return, and the SOXX
07:00-to-09:25 return. Superseded experimental specifications are not kept.

## Retained overnight-return model

`src.hypotheses.overnight_ablation` forecasts SKHY's previous-close-to-09:30
open log return. H1 supplies the equilibrium benchmark; H2 produces the return
forecast. BIC selected `p=0, q=0` for the retained equation. Forecast accuracy
is evaluated by expanding-window estimation using earlier dates only.

The executable `premarket_to_open` strategy is narrower than the H2 target. It
enters at the 09:25 premarket price and liquidates at the 09:30 opening auction
proxy. A prior-only calibration maps the full-gap H2 forecast to the expected
remaining five-minute return. Cost hurdles, volatility-scaled sizing,
stop-loss/take-profit levels, and portfolio drawdown limits are applied before
performance is reported.

Run:

```text
python -m src.hypotheses.overnight_ablation --run-id overnight-best --max-p 3 --max-q 0 --min-train 40
python -m backtest.backtest --run-id overnight-best --h1-spec bivariate --h2-spec korean_ect_soxx_gauges --min-train 40
python -m backtest.strategy --run-id overnight-best --calibration-observations 10 --safety-buffer-bps 10 --transaction-cost-bps-per-side 1 --slippage-bps-per-side 3 --impact-coefficient-bps 10 --risk-aversion 0.0025 --max-weight 1 --stop-loss-sigma 0.75 --take-profit-sigma 1.5 --drawdown-pause-pct 1 --pause-sessions 5 --hard-drawdown-stop-pct 5 --initial-fund 10000 --minimum-fund 9000
```

The forecasting design and retained H1 and H2 estimates are documented in
[`design/OVERNIGHT_FORECAST_DESIGN.md`](design/OVERNIGHT_FORECAST_DESIGN.md).
The execution rules are documented in
[`design/BACKTEST_DESIGN.md`](design/BACKTEST_DESIGN.md).
Trading performance is reported separately in
[`reports/TRADING_RESULTS.md`](reports/TRADING_RESULTS.md).
Parquet model, forecast, and strategy artifacts are stored under
`data/features/experiments/`. The `reports/` directory holds Markdown
results. See [run_commands.txt](run_commands.txt) for the Chronos-2 forecast
and multi-model comparison commands.

See [data/README.md](data/README.md) for the storage layout and
[data/MISSING_DATA.md](data/MISSING_DATA.md) for current source coverage.
