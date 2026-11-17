# **SKHY–Korean Stock Cointegration Research Platform**

Consolidated Implementation Design · Version 4.0 · November 2026

**Technology:** Python · DuckDB · Parquet · SQL · Statsmodels · Streamlit
**Research methodology:** ARDL Bounds Testing · Unrestricted Error-Correction Models · Conditional ECM · Directional Price Adjustment
**Primary output:** Reproducible Markdown research reports with statistical tests, diagnostics, and conclusions.

## **1. Research objectives and design principles**

The system investigates the relationship between SK hynix's Korean ordinary shares (`000660`) and its Nasdaq-listed American Depositary Shares (`SKHY`), controlling for U.S. and Korean market conditions.

The SEC prospectus specifies that one SKHY ADS represents **0.1** Korean ordinary shares, establishing the conversion relationship required for the research.  The platform will initially implement three hypotheses:

| Hypothesis                          | Research question                                            | Main econometric method                |
| ----------------------------------- | ------------------------------------------------------------ | -------------------------------------- |
| **H1: Long-run cointegration**      | Do FX-adjusted Korean shares and SKHY maintain a statistically supported long-run price equilibrium? | Daily ARDL–UECM bounds test            |
| **H2: Error correction**            | Does SKHY adjust toward equilibrium after a liquidity-qualified premarket or regular-market anchor, and at what speed? | Daily ECM and intraday conditional ECM |
| **H3: Directional price discovery** | Does equilibrium adjustment occur predominantly in SKHY or in the subsequent Korean trading session? | Two direction-specific ECMs            |

No Granger causality tests are included.

### **Fixed design requirements**

| Component                       | Requirement                                                 |
| ------------------------------- | ----------------------------------------------------------- |
| Premarket start                 | 7:00 a.m. U.S. Eastern                                      |
| Dynamic anchor search           | 7:00–9:20 a.m. ET                                           |
| Premarket execution eligibility | Subject to Schwab's 7:00–9:25 a.m. window                   |
| Fixed comparison anchor         | 9:30 a.m. ET                                                |
| Default horizons                | 5, 10, 15, 30, 45, 60, 100, 120 minutes                     |
| Horizon flexibility             | Arbitrary supported minute intervals through SQL parameters |
| Korean underlying data          | Daily, with optional intraday support in future             |
| U.S. ADR data                   | Raw trades, quotes, one-minute and five-minute bars         |
| Data storage                    | Parquet                                                     |
| Query engine                    | DuckDB                                                      |
| Configuration                   | Python dataclasses, no YAML                                 |
| Dashboard                       | Root-level `app.py` using Streamlit                         |
| Reports                         | Markdown, no CSV research summaries                         |

Schwab's published standard premarket execution hours are 7:00–9:25 a.m. ET. The anchor search ends at 9:20 to retain a five-minute buffer. Individual security eligibility and execution conditions must still be verified. 

## **2. Core architectural principle**

use a **feature-first, SQL-driven research platform**, where hypotheses are independent consumers of reusable datasets. This architecture supports three types of change without restructuring the application.

| Research change                    | Required modification                                        |
| ---------------------------------- | ------------------------------------------------------------ |
| Change response horizons           | Modify `ExperimentSpec.horizons`                             |
| Add an existing macro control      | Add its registered identifier to `ExperimentSpec.controls`   |
| Introduce a completely new control | Add a source definition, SQL transformation, and registry entry |
| Change the anchor threshold        | Modify the anchor specification and regenerate dependent features |
| Test a new ECM specification       | Add a model specification or estimator                       |
| Introduce H4/H5                    | Register a hypothesis class and its required feature dependencies |
| Extend Korean data to five minutes | Introduce a new Korean intraday feature source without replacing the daily models |

The architecture must also distinguish between **recomputing a feature**, **refitting a model**, and **rerendering a report**. These operations should have separate caching and versioning rules.

## **3. Final project structure**

```
skhynix_project/
│
├── app.py                         # Streamlit application
│
├── data/
│   ├── raw/                       # Immutable vendor snapshots
│   │   ├── us/
│   │   │   ├── market/
│   │   │   │   ├── SKHY_trades/
│   │   │   │   ├── SKHY_quotes/
│   │   │   │   ├── SKHY_1min/
│   │   │   │   ├── SPY_1min/
│   │   │   │   └── SOXX_1min/
│   │   │   └── economic/
│   │   │       ├── treasury/
│   │   │       ├── inflation/
│   │   │       ├── policy/
│   │   │       └── risk/
│   │   ├── korea/
│   │   │   ├── market/
│   │   │   │   ├── 000660_daily/
│   │   │   │   └── sector_daily/
│   │   │   └── economic/
│   │   │       ├── yields/
│   │   │       ├── inflation/
│   │   │       └── policy/
│   │   └── fx/
│   │       └── USDKRW_1min/
│   │
│   ├── core/                      # Standardized, typed data
│   │   ├── us_trades/
│   │   ├── us_quotes/
│   │   ├── us_prices_1min/
│   │   ├── kr_daily/
│   │   ├── fx_quotes/
│   │   └── economic_observations/
│   │
│   ├── features/                  # Reusable derived datasets
│   │   ├── daily/
│   │   ├── us_5min/
│   │   ├── anchors/
│   │   └── experiments/
│   │       └── run_id=.../
│   │           └── model_data.parquet
│   │
│   └── research.duckdb
│
├── sql/
│   ├── core/
│   │   ├── us_prices.sql
│   │   ├── kr_prices.sql
│   │   ├── fx_prices.sql
│   │   └── economic_observations.sql
│   │
│   ├── features/
│   │   ├── daily_prices.sql
│   │   ├── parity.sql
│   │   ├── liquidity_5min.sql
│   │   ├── anchor_candidates.sql
│   │   ├── intraday_returns.sql
│   │   └── fx_volatility.sql
│   │
│   ├── controls/
│   │   ├── us_sector_return.sql
│   │   ├── kr_sector_return.sql
│   │   ├── us_yields.sql
│   │   ├── kr_yields.sql
│   │   ├── economic_surprises.sql
│   │   └── market_risk.sql
│   │
│   ├── datasets/
│   │   ├── h1_daily.sql
│   │   ├── h2_anchor_horizons.sql
│   │   ├── h3_korea_to_us.sql
│   │   └── h3_us_to_korea.sql
│   │
│   └── diagnostics/
│       ├── missingness.sql
│       ├── coverage.sql
│       └── anchor_quality.sql
│
├── src/
│   ├── config.py                  # Typed Python settings
│   │
│   ├── ingestion/
│   │   ├── massive_us.py
│   │   ├── korean_daily.py
│   │   ├── fx_data.py
│   │   └── economic_data.py
│   │
│   ├── engine/
│   │   ├── duckdb_engine.py
│   │   ├── sql_builder.py
│   │   ├── feature_registry.py
│   │   ├── control_registry.py
│   │   ├── dependency_resolver.py
│   │   ├── dataset_builder.py
│   │   ├── experiment.py
│   │   ├── model_registry.py
│   │   └── cache_manager.py
│   │
│   ├── features/
│   │   ├── anchor_selector.py
│   │   ├── price_parity.py
│   │   ├── contracts.py
│   │   └── validators.py
│   │
│   ├── hypotheses/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── h1_cointegration.py
│   │   ├── h2_error_correction.py
│   │   └── h3_price_discovery.py
│   │
│   ├── econometrics/
│   │   ├── ardl.py
│   │   ├── bounds.py
│   │   ├── ecm.py
│   │   ├── unit_root.py
│   │   ├── residual_diagnostics.py
│   │   ├── stability.py
│   │   └── robust_inference.py
│   │
│   ├── reporting/
│   │   ├── markdown_writer.py
│   │   ├── report_models.py
│   │   ├── comparison.py
│   │   └── templates/
│   │       ├── h1.md.j2
│   │       ├── h2.md.j2
│   │       ├── h3.md.j2
│   │       └── summary.md.j2
│   │
│   └── pipelines/
│       ├── ingest.py
│       ├── build_core.py
│       ├── run_experiment.py
│       └── run_all.py
│
├── reports/
│   └── run_id=.../
│       ├── H1_cointegration.md
│       ├── H2_error_correction.md
│       ├── H3_price_discovery.md
│       └── Research_Summary.md
│
├── tests/
│   ├── test_sql_features.py
│   ├── test_timestamp_alignment.py
│   ├── test_anchor_selection.py
│   ├── test_ardl_bounds.py
│   ├── test_ecm.py
│   ├── test_diagnostics.py
│   └── test_reproducibility.py
│
├── requirements.txt
└── README.md
```

U.S. economic series reside under `data/raw/us/economic/`, and Korean economic series under `data/raw/korea/economic/`. The `app.py` file is the single Streamlit entry point.

## **4. Data architecture and dataset contracts**

Use three storage layers:

| Layer      | Storage                              | Main rule                                                    |
| ---------- | ------------------------------------ | ------------------------------------------------------------ |
| `raw`      | Parquet                              | Preserve original vendor observations and provenance         |
| `core`     | Parquet                              | Normalize timestamps, currencies, identifiers, units, and availability |
| `features` | DuckDB views or materialized Parquet | Generate reusable research features from SQL                 |

All timestamps should be UTC-aware. Maintain a separate U.S. trading-session date using `America/New_York` and Korean session date using `Asia/Seoul`.

### **4.1 Six principal research datasets**

| Dataset   | Row granularity                         | Primary content                                              | Consumer          |
| --------- | --------------------------------------- | ------------------------------------------------------------ | ----------------- |
| `daily`   | One paired U.S. trading session         | ADR close, last known Korean close, FX, market controls      | H1 and H3         |
| `us_5min` | One completed SKHY five-minute interval | OHLCV, quotes, spreads, volume, FX, known Korean reference   | Anchor engine, H2 |
| `anchors` | One date × anchor type                  | Selected time, price, spread, liquidity score, validity      | H2 and H3         |
| `h1`      | One eligible daily observation          | Price levels, differences, lags, selected controls           | ARDL–UECM         |
| `h2`      | One date × anchor × horizon             | Initial ECT, liquidity, future ADR return                    | Conditional ECM   |
| `h3`      | One directional target observation      | Pre-response ECT, completed source-market return, target-market return | Directional ECM   |

The first three are reusable foundation datasets. The last three should normally be **SQL-generated views**, optionally materialized in versioned Parquet for a particular experiment.

### `daily` **— canonical daily research view**

Primary fields:

```
date_us                  DATE
us_close_ts_utc          TIMESTAMPTZ
adr_close_usd            DOUBLE
kr_session_date          DATE
kr_close_ts_utc          TIMESTAMPTZ
kr_available_at_utc      TIMESTAMPTZ
kr_close_krw             DOUBLE
fx_asof_ts_utc           TIMESTAMPTZ
usdkrw_asof              DOUBLE
log_adr                  DOUBLE
log_kr_usd               DOUBLE
parity_spread            DOUBLE
adr_daily_return         DOUBLE
kr_daily_return_known    DOUBLE
us_sector_return         DOUBLE
kr_sector_return_known   DOUBLE
data_quality_flags       VARCHAR
```

The Korean close is the last **available** observation before the U.S. price timestamp, not necessarily the same calendar date.

### `us_5min` **— reusable five-minute market features**

Primary fields:

```
date_us                DATE
bar_start_utc          TIMESTAMPTZ
bar_end_utc            TIMESTAMPTZ
session_type           VARCHAR
open, high, low, close DOUBLE
volume                 BIGINT
trade_count            BIGINT
dollar_volume          DOUBLE
bid, ask, mid          DOUBLE
bid_size, ask_size     BIGINT
spread_bps             DOUBLE
quote_updates          BIGINT
quote_coverage         DOUBLE
quote_age_seconds      DOUBLE
fx_mid_asof            DOUBLE
kr_close_known         DOUBLE
parity_spread          DOUBLE
valid_price            BOOLEAN
valid_quote            BOOLEAN
```

Bars are keyed by `(symbol, bar_end_utc)` if the platform later includes more securities.

Preserve the underlying one-minute observations. A future seven-minute or 37-minute response horizon cannot be measured precisely from five-minute bars alone.

### `anchors` **— selection-event dataset**

Primary fields:

```
date_us                 DATE
anchor_type             VARCHAR
anchor_ts_utc           TIMESTAMPTZ
anchor_valid            BOOLEAN
selection_status        VARCHAR
anchor_mid_usd          DOUBLE
anchor_trade_usd        DOUBLE
liquidity_score         DOUBLE
spread_bps              DOUBLE
volume_5min             BIGINT
trade_count_5min        BIGINT
quote_updates_5min      BIGINT
kr_close_known          DOUBLE
usdkrw_anchor           DOUBLE
parity_spread_anchor    DOUBLE
selection_version       VARCHAR
```

Each day contains a dynamic-anchor record and a fixed-opening record. Invalid anchors remain in the data with null observation fields and explicit status codes.

### `h1` **— daily cointegration dataset**

Primary fields:

```
date_us                 DATE
log_adr                 DOUBLE
log_kr_usd              DOUBLE
d_log_adr               DOUBLE
d_log_kr_usd            DOUBLE
log_adr_l1              DOUBLE
log_kr_usd_l1           DOUBLE
d_log_adr_l1            DOUBLE
d_log_kr_usd_l1         DOUBLE
[selected controls]     DOUBLE
sample_eligible         BOOLEAN
```

The SQL engine generates lag columns according to the selected ARDL specification. Unused lags are not generated unnecessarily.

### `h2` **— anchor × horizon observations**

Primary fields:

```
date_us                 DATE
anchor_type             VARCHAR
anchor_ts_utc           TIMESTAMPTZ
horizon_min             INTEGER
endpoint_ts_utc         TIMESTAMPTZ
ect_anchor              DOUBLE
liquidity_score         DOUBLE
ect_x_liquidity         DOUBLE
spread_bps              DOUBLE
kr_return_known         DOUBLE
[selected controls]     DOUBLE
adr_return_h            DOUBLE
ect_change_h            DOUBLE
crosses_open            BOOLEAN
valid_outcome           BOOLEAN
h1_model_id             VARCHAR
```

One day can generate up to 16 default horizon observations if both anchors and all eight endpoints are valid.

### `h3` **— directional observations**

H3 should use two physical SQL-generated datasets because the two directions have different granularities:

- `h3_korea_to_us`: one U.S. date × anchor × horizon.
- `h3_us_to_korea`: one completed U.S. session × next Korean trading session.

Both should expose a normalized interface:

```
direction                VARCHAR
source_session_date      DATE
target_session_date      DATE
information_cutoff_utc   TIMESTAMPTZ
anchor_type              VARCHAR / NULL
horizon_min              INTEGER / NULL
ect_pre_response         DOUBLE
source_market_return     DOUBLE
target_market_return     DOUBLE
[selected controls]      DOUBLE
valid_outcome            BOOLEAN
h1_model_id              VARCHAR
```

The direction-specific SQL must enforce that all predictors were actually observable before the target-market response began.

## **5. Price parity and point-in-time alignment**

The price-parity calculation should be implemented once, in `src/features/price_parity.py` and the corresponding `sql/features/parity.sql`.

For SKHY, the theoretical ADS price in USD is:

\[ P^{A,*}_t=\frac{P^K_t}{10FX_t} \]

Define:

\[ \begin{aligned} a_t&=\ln P^{SKHY}_t\\ k_t&=\ln P^{KR}_t-\ln FX_t-\ln10 \end{aligned} \]

The theoretical log premium is:

\[ s_t=a_t-k_t \]

The empirically estimated equilibrium error is:

\[ \boxed{ ECT_t=a_t-\hat c-\hat\theta k_t } \]

For H2/H3, \(\hat c\) and \(\hat\theta\) must come from an earlier or otherwise appropriately separated H1 estimation period.

**Do not use an ECT estimated on the full future sample for historical out-of-sample evaluation.**

### **5.1 A reusable point-in-time SQL join**

```
-- sql/features/parity.sql
SELECT
    a.ts_utc,
    a.date_us,
    a.mid_usd AS adr_mid,

    k.close_krw,
    k.available_at_utc AS kr_available_at,

    f.usdkrw_mid,
    f.ts_utc AS fx_ts,

    LN(a.mid_usd) AS log_adr,

    LN(
        k.close_krw / (10.0 * f.usdkrw_mid)
    ) AS log_kr_usd,

    LN(a.mid_usd)
      - LN(
          k.close_krw / (10.0 * f.usdkrw_mid)
        ) AS parity_spread

FROM core_us_prices_1min a

ASOF LEFT JOIN core_kr_daily k
    ON a.ts_utc >= k.available_at_utc

ASOF LEFT JOIN core_fx_quotes f
    ON a.ts_utc >= f.ts_utc

WHERE a.mid_usd > 0
  AND k.close_krw > 0
  AND f.usdkrw_mid > 0;
```

DuckDB supports as-of joins that attach the latest preceding observation to each row, making them appropriate for asynchronous market data. 

DuckDB



This example assumes that `core_kr_daily` contains only SK hynix and `core_fx_quotes` contains only USD/KRW. A multi-instrument database must also join on instrument or series identifiers.

After the join, the engine must validate source age. A quote published days earlier must not be accepted simply because it is the most recent available quote.

The query also assumes `available_at_utc` accurately represents the Korean closing price's actual availability. The source's date alone is insufficient.

### **5.2 Economic-control availability**

Every economic-control observation must distinguish:

| Field              | Meaning                               |
| ------------------ | ------------------------------------- |
| `observation_date` | Economic period represented           |
| `release_ts_utc`   | Public announcement timestamp         |
| `available_at_utc` | Earliest time the system could use it |
| `vintage_ts_utc`   | Publication/revision vintage          |
| `value`            | Value available in that vintage       |
| `series_id`        | Stable economic-series identifier     |
| `source_version`   | Downloaded data snapshot              |

For example, a U.S. CPI figure released at 8:30 a.m. ET may be included in a 9:00 a.m. experiment, but it must not appear in features for a 7:45 a.m. anchor on the same day.

This rule applies equally to Korean macroeconomic releases, U.S. market indices, FX, and other controls.

## **6. Dynamic premarket anchor engine**

The anchor engine should be independent of H1, H2, and H3. Future hypotheses may reuse its output without modifications.

### **6.1 Selection algorithm**

At each U.S. trading date \(d\), use completed five-minute SKHY bars ending between 7:05 and 9:20 a.m. ET.

Calculate a market-quality score from:

\[ \mathbf L_{d,t}= \left[ Volume,\ TradeCount,\ QuoteUpdates,\ Spread,\ QuoteCoverage,\ QuoteAge \right] \]

Optional features include displayed bid/ask sizes, dollar volume, and price-impact proxies.

A bar qualifies only when minimum trade and quote conditions are met. To avoid isolated liquidity spikes, require two consecutive qualifying bars.

\[ \boxed{ t_d^*= \min\left\{ t\le09{:}20: Q_{d,t}=1,\ Q_{d,t-5}=1 \right\} } \]

The first possible confirmed anchor is 7:10 a.m.

### **6.2 Division of responsibility**

| Module                  | Responsibility                                               |
| ----------------------- | ------------------------------------------------------------ |
| `liquidity_5min.sql`    | Compute volume, trade count, spreads, quote coverage         |
| `anchor_candidates.sql` | Produce candidate bars and historical standardized features  |
| `anchor_selector.py`    | Apply causal stopping rule and select earliest qualifying time |
| `feature_registry.py`   | Register `anchors` as a reusable feature                     |
| `anchor_quality.sql`    | Summarize anchor coverage and quality                        |

The historical liquidity thresholds should be calculated from prior trading days only, ideally by time-of-day bucket.

The selection engine should never choose the highest-volume interval retrospectively using later observations.

### **6.3 Missing and illiquid days**

The anchor engine must preserve:

- Valid dynamic anchor.
- No qualifying dynamic anchor.
- Invalid or missing quotes.
- Missing trades.
- Incomplete historical liquidity baseline.
- Fixed-opening anchor unavailable or stale.

These statuses become research features as well as data-quality indicators.

For example, a future H4 model could examine whether no-anchor days predict lower liquidity or more extreme volatility at the regular open.

### **6.4 Anchor configuration**

Use Python rather than YAML:

```
from dataclasses import dataclass


@dataclass(frozen=True)
class AnchorSpec:
    start_et: str = "07:00"
    cutoff_et: str = "09:20"
    fixed_open_et: str = "09:30"

    bar_minutes: int = 5
    consecutive_bars: int = 2

    min_trades: int = 10
    min_volume: int = 1000
    min_quote_updates: int = 10

    max_spread_bps: float = 100.0
    min_quote_coverage: float = 0.80
    max_quote_age_seconds: float = 30.0

    history_days: int = 20
    min_history_days: int = 10
    volume_percentile: float = 0.60

    min_liquidity_score: float = 0.0
```

These thresholds are initial research parameters, not validated empirical estimates for SKHY. Their values should be tested against historical data quality, and the chosen version must be saved with every experiment.

## **7. SQL-driven feature engine**

The engine must support two types of change:

1. **Parameter changes:** Change horizons, anchors, control selection, lags, or sample periods without editing SQL files.
2. **Feature-definition changes:** Add a new SQL transformation or input source, register it, and rebuild only affected dependencies.

### **7.1 Feature registration**

```
# src/engine/feature_registry.py

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    sql_path: str
    grain: str
    dependencies: tuple[str, ...]
    version: str = "v1"


FEATURES = {
    "daily": FeatureSpec(
        name="daily",
        sql_path="sql/features/daily_prices.sql",
        grain="us_session",
        dependencies=(
            "us_daily",
            "kr_daily",
            "fx_quotes",
        ),
    ),

    "us_5min": FeatureSpec(
        name="us_5min",
        sql_path="sql/features/liquidity_5min.sql",
        grain="us_5min",
        dependencies=(
            "us_trades",
            "us_quotes",
        ),
    ),

    "anchors": FeatureSpec(
        name="anchors",
        sql_path="sql/features/anchor_candidates.sql",
        grain="us_session_anchor",
        dependencies=("us_5min",),
    ),
}
```

For the anchor feature, the registry must declare that SQL generates the candidates and `anchor_selector.py` applies the selection rule. A general feature definition can therefore specify both an SQL transform and an optional Python postprocessor.

### **7.2 Control registry**

All macroeconomic and market controls should use a stable, normalized name.

```
# src/engine/control_registry.py

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ControlSpec:
    name: str
    sql_path: str
    frequency: str
    join_method: Literal[
        "asof",
        "previous_session",
        "event_window",
    ]
    max_age_minutes: int | None = None
    min_coverage: float = 0.80


CONTROLS = {
    "us_soxx_return": ControlSpec(
        name="us_soxx_return",
        sql_path="sql/controls/us_sector_return.sql",
        frequency="intraday",
        join_method="asof",
        max_age_minutes=5,
    ),
    "kr_sector_return": ControlSpec(
        name="kr_sector_return",
        sql_path="sql/controls/kr_sector_return.sql",
        frequency="daily",
        join_method="previous_session",
    ),
    "us_2y_yield_change": ControlSpec(
        name="us_2y_yield_change",
        sql_path="sql/controls/us_yields.sql",
        frequency="daily",
        join_method="asof",
    ),
    "us_cpi_surprise": ControlSpec(
        name="us_cpi_surprise",
        sql_path="sql/controls/economic_surprises.sql",
        frequency="event",
        join_method="event_window",
    ),
}
```

The control registry describes availability and alignment rules. The feature engine must also check statistical suitability, including missingness, integration order, and multicollinearity.

Adding a control already in the registry requires only including its identifier in the experiment specification.

### **7.3 Parameterized horizon SQL**

The system should not hardcode eight prediction horizons.

Instead, define:

\[ \mathcal H=\{h_1,h_2,\ldots,h_n\} \]

and generate future endpoints from the requested horizons.

```
-- sql/datasets/h2_anchor_horizons.sql

WITH requested_horizons AS (
    SELECT UNNEST($horizons::INTEGER[]) AS horizon_min
),

requests AS (
    SELECT
        a.date_us,
        a.anchor_type,
        a.anchor_ts_utc,
        a.anchor_mid_usd,
        a.ect_anchor,
        h.horizon_min,

        a.anchor_ts_utc
            + h.horizon_min * INTERVAL '1 minute'
            AS target_ts_utc

    FROM feature_anchors a
    CROSS JOIN requested_horizons h

    WHERE a.anchor_valid
),

endpoints AS (
    SELECT
        r.*,
        p.ts_utc AS endpoint_ts_utc,
        p.mid_usd AS endpoint_mid_usd

    FROM requests r

    ASOF LEFT JOIN core_us_prices_1min p
      ON r.date_us = p.date_us
     AND r.target_ts_utc >= p.ts_utc
)

SELECT
    *,
    CASE
        WHEN endpoint_mid_usd > 0
         AND endpoint_ts_utc >=
             target_ts_utc - INTERVAL '60 seconds'
        THEN LN(endpoint_mid_usd / anchor_mid_usd)
        ELSE NULL
    END AS adr_return_h

FROM endpoints;
```

The example uses the latest valid endpoint price no later than the requested horizon. Its maximum permitted staleness is 60 seconds, subject to calibration.

For a strict exact-horizon study, the system should instead require a quote at the requested endpoint or within a separately documented tolerance.

The production query must also check session eligibility and opening-boundary crossings.

### **Result**

Changing the horizons to:

```
horizons = (5, 10, 20, 40, 90, 180)
```

requires no change to the SQL transformation itself.

As long as the underlying one-minute or tick data cover the requested endpoints, the new feature dataset can be generated immediately.

## **8. Hypothesis engine: independent models, shared features**

All hypotheses should implement the same interface, while retaining their own estimation and statistical-testing logic.

```
# src/hypotheses/base.py

from abc import ABC, abstractmethod


class BaseHypothesis(ABC):

    hypothesis_id: str
    dataset_name: str

    @abstractmethod
    def required_features(self, spec) -> tuple[str, ...]:
        ...

    @abstractmethod
    def estimate(self, dataset, spec):
        ...

    @abstractmethod
    def diagnose(self, fitted_model, spec):
        ...

    @abstractmethod
    def interpret(self, fitted_model, diagnostics):
        ...


HYPOTHESIS_REGISTRY = {}


def register_hypothesis(cls):
    HYPOTHESIS_REGISTRY[cls.hypothesis_id] = cls
    return cls
```

The registry should validate that every requested hypothesis has a known dataset builder, estimator, diagnostic set, and report template.

This allows future models to be introduced by registering a new hypothesis rather than modifying the core processing pipeline.

## **9. H1 — Long-run cointegration**

### **9.1 Econometric specification**

Use the daily FX-adjusted prices:

\[ a_d=\ln P^{SKHY}_d \]

\[ k_d=\ln P^K_d-\ln FX_d-\ln10 \]

The unrestricted error-correction specification is:

\[ \boxed{ \begin{aligned} \Delta a_d={}&c+\rho_a a_{d-1}+\rho_k k_{d-1}\\ &+\sum_{i=1}^{p-1}\phi_i\Delta a_{d-i}\\ &+\sum_{j=0}^{q-1}\gamma_j\Delta k_{d-j}\\ &+\boldsymbol\delta^\top Z_d+\varepsilon_d \end{aligned} } \]

The long-run elasticity is:

\[ \theta=-\frac{\rho_k}{\rho_a} \]

The principal bounds-test null is:

\[ H_0:\rho_a=\rho_k=0 \]

The economically motivated parity restriction is:

\[ H_0:\theta=1 \]

Controls \(Z_d\) should initially be included as stationary short-run or predetermined regressors, not indiscriminately added to the cointegrating vector.

### **9.2 H1 pipeline**

```
src/hypotheses/h1_cointegration.py
class H1Cointegration(BaseHypothesis):

    hypothesis_id = "H1"
    dataset_name = "h1_daily"

    def required_features(self, spec):
        return (
            "log_adr",
            "log_kr_usd",
            *spec.controls,
        )

    def estimate(self, dataset, spec):
        # Integration-order assessment
        # Lag selection
        # ARDL and unrestricted ECM
        # Bounds and long-run restrictions
        return fit_ardl_cointegration(dataset, spec)

    def diagnose(self, fitted_model, spec):
        return diagnose_ardl(fitted_model, spec)

    def interpret(self, fitted_model, diagnostics):
        return interpret_h1(fitted_model, diagnostics)
```

The supporting econometric library should contain the actual implementations. Hypothesis classes should orchestrate reusable functions, not duplicate statistical formulas.

### **9.3 Required H1 results**

| Category                   | Results                                                      |
| -------------------------- | ------------------------------------------------------------ |
| Descriptive statistics     | Mean, SD, skewness, kurtosis, correlations                   |
| Integration                | ADF, PP, KPSS                                                |
| Model selection            | Selected ARDL lag orders, AIC/BIC                            |
| Cointegration              | Bounds F-statistic, lower/upper critical values              |
| Small-sample inference     | Narayan reference where applicable; finite-sample simulation/bootstrap |
| Supporting bounds evidence | Lagged-level coefficient tests, including degenerate-case assessment |
| Long-run equilibrium       | \(\hat c,\hat\theta\), confidence intervals                  |
| Parity restriction         | Wald statistic and p-value                                   |
| Daily adjustment           | Estimated \(\lambda_A\)                                      |
| Diagnostics                | Serial correlation, heteroskedasticity, normality, specification, stability |

The attached paper motivates ARDL bounds testing for \(I(0)\)/\(I(1)\) variables and finite-sample critical values, and uses AIC for lag selection. These elements will be preserved. 

**Small-sample rule:** Use **Narayan's tabulated values** only when the relevant sample size, deterministic case, and number of level regressors are covered. Otherwise, use appropriate finite-sample simulation or bootstrap inference rather than extrapolating a critical value.

The system must also record whether valid cointegration evidence was obtained. If H1 is inconclusive, H2/H3 can still investigate adjustments to theoretical parity, but must not present that analysis as a validated cointegration-based ECM.

## **10. H2 — Error-correction analysis**

H2 consists of two estimation families.

### **10.1 H2-A: Daily ECM**

Estimate the daily restricted ECM associated with the accepted long-run relationship:

\[ \boxed{ \Delta a_d= c+\lambda_AECT_{d-1} +\sum_i\phi_i\Delta a_{d-i} +\sum_j\gamma_j\Delta k_{d-j} +\delta^\top Z_d+\varepsilon_d } \]

The key result is \(\lambda_A\), with expected convergence sign:

\[ \lambda_A<0 \]

Report its confidence interval, p-value, and any meaningful implied adjustment half-life.

### **10.2 H2-B: Dynamic-anchor horizon response**

For each dynamically selected premarket anchor \(t_d^*\) and fixed 9:30 benchmark:

\[ \boxed{ \begin{aligned} R^A_{d,h}={}&\alpha_h+\lambda_hECT_{d,t_d^*}\\ &+\beta_h LS_{d,t_d^*}\\ &+\gamma_h(ECT_{d,t_d^*}\times LS_{d,t_d^*})\\ &+\boldsymbol\delta_h^\top Z_{d,t_d^*}+\varepsilon_{d,h} \end{aligned} } \]

This allows the study to investigate both error-correction behavior and its relationship with liquidity.

The H2 estimator should receive the horizons and controls through an `ExperimentSpec`.

```
H2_SPEC = ExperimentSpec(
    hypothesis="H2",
    estimator="conditional_ecm",
    anchors=("dynamic", "fixed_0930"),
    horizons=(5, 10, 15, 30, 45, 60, 100, 120),
    controls=(
        "us_soxx_return",
        "kr_sector_return",
        "fx_volatility",
    ),
)
```

This is an illustrative specification; the full dataclass and defaults are defined in `src/engine/experiment.py`.

### **H2 diagnostic requirements**

In addition to the paper-derived residual tests, include checks specific to the dynamic anchor:

| Diagnostic                       | Required analysis                                            |
| -------------------------------- | ------------------------------------------------------------ |
| Anchor coverage                  | Percentage of dates with valid premarket anchors             |
| Anchor distribution              | Median, quantiles, and range of selected ET times            |
| Liquidity quality                | Selected volume, spread, trade count, quote coverage         |
| Horizon completeness             | Number of valid observations by horizon                      |
| Time-of-day sensitivity          | Does adjustment depend on how close the anchor is to 9:30?   |
| Premarket versus regular session | Separately estimate outcomes crossing the opening            |
| Inference                        | Heteroskedasticity/serial-dependence robust confidence intervals |
| Multiple testing                 | Adjust across horizon families                               |
| Parameter stability              | Rolling estimates, CUSUM/CUSUMSQ where applicable            |

The full H2 analysis must distinguish two types of coefficients:

- The daily restricted ECM adjustment coefficient measures adjustment per daily observation.
- The intraday horizon coefficient measures cumulative conditional response over \(h\) minutes.

They should not be reported as if they were the same statistical quantity.

## **11. H3 — Directional price-discovery analysis**

H3 uses two separately constructed response datasets.

### **11.1 Korea → U.S.**

\[ \boxed{ R^A_{d,h}= c+\lambda_{KA,h}ECT_{d,t_d^*} +\beta_Kr^K_d +\delta^\top Z^{US}_{d,t_d^*} +\epsilon_{d,h} } \]

The expected correction sign is:

\[ \lambda_{KA,h}<0 \]

The source Korean return must come from an already completed session.

### **11.2 U.S. → Korea**

\[ \boxed{ r^K_{next(d)} = c+\lambda_{AK}ECT_{d,preKR} +\beta_Ar^A_d +\omega^\top Z^{KR}_{next(d),known} +\eta_d } \]

The expected correction sign is:

\[ \lambda_{AK}>0 \]

The information cutoff for this model is immediately before the target Korean session.

The economic controls must be known at that cutoff. For example, the target session's completed Korean sector return cannot be included as a predetermined control.

### **11.3 Directional interpretation**

| Empirical finding                                            | Supported interpretation                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| U.S. adjustment significant, Korean adjustment insignificant | Evidence consistent with stronger Korean reference-price leadership |
| Korean adjustment significant, U.S. adjustment insignificant | Evidence consistent with stronger U.S. ADR information leadership |
| Both adjust significantly                                    | Bidirectional cross-session correction                       |
| Neither adjusts significantly                                | No detected directional correction                           |

H3 should report estimates using comparable daily session horizons as an additional robustness exercise.

However, because the markets are asynchronous, the raw coefficients cannot be used alone to calculate a definitive percentage of information contributed by each market.

No Granger tests or Granger-based price-discovery measures should be included.

## **12. Shared statistical diagnostics**

The original paper uses ADF/PP integration tests, ARDL bounds testing, Breusch–Godfrey LM, Breusch–Pagan–Godfrey, Ramsey RESET, Jarque–Bera, CUSUM, and CUSUMSQ.

Its Table 8 contains the main residual diagnostics, while Figures 2 and 3 present stability plots. 

2644.pdf



The project should implement these centrally.

| Module                    | Tests                                                        | Application                  |
| ------------------------- | ------------------------------------------------------------ | ---------------------------- |
| `unit_root.py`            | ADF, PP, KPSS                                                | Integration-order assessment |
| `bounds.py`               | ARDL F-bounds, t-bounds, bootstrap                           | H1                           |
| `residual_diagnostics.py` | Breusch–Godfrey, Breusch–Pagan, Ramsey RESET, Jarque–Bera, ARCH-LM | H1–H3                        |
| `stability.py`            | CUSUM, CUSUMSQ, rolling coefficients                         | H1–H3 where sample permits   |
| `robust_inference.py`     | HAC, clustered inference, bootstrap confidence intervals     | H1–H3                        |
| `data_quality.py`         | Coverage, missingness, stale observations, timestamp validity | All datasets                 |

Each diagnostic should return a structured result:

```
from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticResult:
    test_name: str
    null_hypothesis: str
    statistic: float | None
    p_value: float | None
    alpha: float
    conclusion: str
    status: str
    sample_size: int
    applicable: bool
    details: str
```

Supported statuses should include:

- `PASS`: no significant violation detected.
- `WARN`: questionable assumption or fragile result.
- `FAIL`: mandatory requirement violated.
- `NOT_TESTED`: test inappropriate or insufficient observations.

A nonsignificant diagnostic result should not be described as proof that the model assumption holds.

For very small samples, some diagnostics may have low power or unreliable critical values. The Markdown output must explicitly disclose this.

## **13. Experiment specification and model registry**

The experiment specification is the principal research configuration interface.

It replaces scattered hardcoded parameters and YAML files.

### **13.1 Python specification**

```
# src/engine/experiment.py

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ExperimentSpec:

    hypothesis: str
    estimator: str

    start_date: str
    end_date: str

    anchors: tuple[str, ...] = ("dynamic", "fixed_0930")
    horizons: tuple[int, ...] = (
        5, 10, 15, 30, 45, 60, 100, 120
    )

    controls: tuple[str, ...] = ()
    max_lags: int = 2

    significance: float = 0.05
    selection_criterion: Literal["aic", "bic"] = "bic"

    inference: str = "robust"
    diagnostics: str = "full"

    training_end_date: str | None = None
    h1_model_id: str | None = None

    materialize_features: bool = True
    generate_markdown: bool = True


@dataclass(frozen=True)
class ResearchConfig:

    adr_ticker: str = "SKHY"
    korea_ticker: str = "000660"
    ordinary_shares_per_ads: float = 0.1

    anchor: AnchorSpec = field(default_factory=AnchorSpec)

    default_horizons: tuple[int, ...] = (
        5, 10, 15, 30, 45, 60, 100, 120
    )
``
```

The specification must be validated before execution. A requested control must exist in the registry, a horizon must be positive, and no evaluation observation may occur before the necessary training history.

### **13.2 Model registry**

Store the following experiment metadata in DuckDB:

| Field                                | Description                              |
| ------------------------------------ | ---------------------------------------- |
| `run_id`                             | Unique experiment identifier             |
| `hypothesis_id`                      | H1, H2, H3, etc.                         |
| `spec_hash`                          | Hash of normalized experiment parameters |
| `source_snapshot_hash`               | Version of input datasets                |
| `sql_hash`                           | Version of compiled SQL                  |
| `feature_version`                    | Feature transformation version           |
| `h1_model_id`                        | Equilibrium model used by H2/H3          |
| `training_start`, `training_end`     | Estimation period                        |
| `evaluation_start`, `evaluation_end` | Out-of-sample period                     |
| `effective_n`                        | Valid observations after filtering       |
| `estimator_version`                  | Estimation code version                  |
| `diagnostic_status`                  | Overall test assessment                  |
| `report_path`                        | Generated Markdown report                |
| `created_at_utc`                     | Execution timestamp                      |

Save estimated coefficients, test statistics, confidence intervals, and model statuses in normalized DuckDB tables.

The Markdown files remain the published research artifacts, while DuckDB holds the structured results used by Streamlit and experiment comparisons.

### **13.3 Dependency invalidation**

The engine should recompute only the affected layers.

| Change                              | Recompute                                                  |
| ----------------------------------- | ---------------------------------------------------------- |
| New horizon                         | Horizon endpoints, affected H2/H3 datasets and estimates   |
| Additional existing control         | Control join, affected model dataset and estimate          |
| New macro data source               | Standardized source, dependent control features and models |
| Different liquidity thresholds      | Anchors and all affected H2/H3 analyses                    |
| Different H1 long-run specification | H1 estimates, dependent ECTs, H2/H3 results                |
| Markdown template change            | Reports only                                               |
| Streamlit layout change             | Dashboard only                                             |

A SQL-definition change must generate a new feature version rather than silently replacing an old research result.

## **14. Reporting system: Markdown as the final research product**

Each experiment should produce self-contained Markdown output containing model estimates, tests, diagnostics, interpretations, and sample limitations.

The reporting layer should not contain econometric estimation logic.

It receives a structured `ModelResult` and renders a report using Jinja2 templates.

### **Required reports**

| Report                   | Content                                                      |
| ------------------------ | ------------------------------------------------------------ |
| `H1_cointegration.md`    | Integration tests, lag selection, ARDL bounds, long-run coefficients, restrictions, diagnostics |
| `H2_error_correction.md` | Daily ECM, intraday horizon coefficients, anchor performance, liquidity interaction, diagnostics |
| `H3_price_discovery.md`  | Korea-to-U.S. and U.S.-to-Korea adjustment, uncertainty, comparative interpretation |
| `Research_Summary.md`    | Integrated hypothesis decisions, key estimates, diagnostic status, limitations |

### **Example of an H1 results section**

Illustrative Markdown report format

The values below are placeholders, not empirical findings.

| **Test**           | **Statistic** | **p-value** | **Decision** |
| ------------------ | ------------- | ----------- | ------------ |
| ADF — ADR level    | Computed      | Computed    | Pending      |
| ARDL bounds        | Computed F    | Computed    | Pending      |
| Parity restriction | Computed Wald | Computed    | Pending      |
| Breusch–Godfrey    | Computed      | Computed    | Pending      |
| Breusch–Pagan      | Computed      | Computed    | Pending      |
| Ramsey RESET       | Computed      | Computed    | Pending      |
| Jarque–Bera        | Computed      | Computed    | Pending      |
| CUSUM / CUSUMSQ    | —             | —           | Pending      |

Every test should be accompanied by its null hypothesis, effective sample size, statistical interpretation, and appropriate warnings.

H2 reports must additionally document the number of valid dynamic anchors, their time distribution, and the number of available endpoint observations for each horizon.

### **Report versioning**

```
reports/
└── run_id=h2_20261116_.../
    ├── H2_error_correction.md
    └── ...
```

A consolidated research run generates all four reports. A standalone H2 experiment generates its H2 report and metadata without overwriting the H1 or H3 results from other runs.

## **15. Streamlit application:** `app.py`

The dashboard should be an interface to the experiment engine, not a replacement for it.

### **Recommended navigation**

## **SKHY Research Platform**

Proposed Streamlit navigation

**Research Overview**

Hypothesis conclusions and statistical diagnostics

**Data Explorer**

Korean prices, SKHY prices, FX, economic features, coverage

**Dynamic Anchor Explorer**

7:00–9:20 liquidity candidates, selected time, 9:30 comparison

**Experiment Builder**

Choose hypotheses, horizons, controls, lags, sample dates

**Model Results**

ARDL, ECM, diagnostic tests, stability figures

**Experiment Comparison**

Compare selected horizons, controls, model versions and outcomes

**Markdown Reports**

Read and export complete research reports

### **15.1 Streamlit experiment controls**

The experiment builder should allow the researcher to select:

- Hypothesis and estimator.
- Sample period and training/evaluation split.
- Dynamic, 9:30, or both anchor types.
- Any supported horizon set.
- Registered U.S. and Korean controls.
- ARDL lag limits and selection criteria.
- Statistical significance and diagnostic options.

Before estimation, Streamlit should run a **feature-availability preview**.

For example:

| Check                  | Result                         |
| ---------------------- | ------------------------------ |
| Requested controls     | All registered                 |
| Historical coverage    | Show available dates           |
| Effective observations | Computed after requested joins |
| Missingness            | Percentage by control          |
| Endpoint coverage      | Percentage by horizon          |
| Model complexity       | Check degrees of freedom       |
| Time leakage           | Must pass                      |

If a model is not estimable, the application should explain which variables or sample restrictions caused the problem.

### **15.2 App implementation principle**

`app.py` should call the shared engine:

```
from src.engine.experiment import ExperimentSpec
from src.pipelines.run_experiment import run_experiment
```

Streamlit can construct an `ExperimentSpec` from its UI controls and invoke the pipeline only when the researcher explicitly runs an experiment.

Expensive tests such as bootstrap ARDL bounds should not rerun on every dashboard refresh.

For larger computations, the app can display stored results and execute explicitly requested research runs through the pipeline. A model registry prevents identical specifications from being fitted unnecessarily.

## **16. End-to-end execution pipeline**

The final orchestrator should be dependency-aware and work for both standalone and combined experiments.

```
# src/pipelines/run_experiment.py

def run_experiment(spec):

    # 1. Validate requested research specification.
    validate_experiment_spec(spec)

    # 2. Find registered hypothesis.
    hypothesis = get_hypothesis(spec.hypothesis)

    # 3. Resolve features and controls.
    required = hypothesis.required_features(spec)

    feature_plan = resolve_dependencies(
        required_features=required,
        controls=spec.controls,
    )

    # 4. Build only missing or outdated features.
    execute_feature_plan(feature_plan)

    # 5. Generate model-ready dataset from SQL.
    dataset = build_model_dataset(
        hypothesis=spec.hypothesis,
        specification=spec,
    )

    # 6. Validate sample and point-in-time correctness.
    validate_dataset_contract(dataset, spec)
    validate_no_lookahead(dataset, spec)

    # 7. Obtain H1 equilibrium coefficients if needed.
    dependencies = resolve_model_dependencies(spec)

    # 8. Estimate econometric model.
    fitted = hypothesis.estimate(
        dataset,
        spec,
    )

    # 9. Run statistical diagnostics.
    diagnostics = hypothesis.diagnose(
        fitted,
        spec,
    )

    # 10. Interpret and validate conclusions.
    interpretation = hypothesis.interpret(
        fitted,
        diagnostics,
    )

    # 11. Save experiment metadata and fitted results.
    run_id = save_experiment(
        spec,
        fitted,
        diagnostics,
        interpretation,
        dependencies,
    )

    # 12. Generate versioned Markdown report.
    generate_markdown_report(run_id)

    return run_id
```

The called helper functions are interfaces to be implemented in their corresponding modules.

The production engine must resolve the H1 dependency **before** constructing the final H2/H3 model dataset whenever an estimated ECT is required.

A valid H1 model reference should include the coefficient version, training cutoff, estimation sample, and cointegration decision. If the H1 relationship is not statistically supported, the user should explicitly select a theory-constrained parity model rather than silently treating an estimated ECT as validated.

### **Running from the command line**

```
# Refresh market and economic data
python -m src.pipelines.ingest

# Normalize core datasets
python -m src.pipelines.build_core

# Run the registered default H1–H3 research
python -m src.pipelines.run_all

# Launch Streamlit
streamlit run app.py
```

For new experiments, the same engine should accept a Python `ExperimentSpec` constructed programmatically or through Streamlit.

## **17. Automated testing and acceptance criteria**

The test suite should validate data integrity, feature generation, econometric calculations, and reporting.

| Test category          | Mandatory verification                                       |
| ---------------------- | ------------------------------------------------------------ |
| Price parity           | Correct 0.1 ordinary-share-per-ADS conversion                |
| Time alignment         | No observation is used before its availability timestamp     |
| Korean daily reference | No invented Korean five-minute observations                  |
| Premarket selection    | Start 7:00; earliest qualified anchor 7:10 under two-bar rule |
| Anchor cutoff          | No dynamic anchor after 9:20 ET                              |
| Fixed opening          | 9:30 observation follows a documented quote/trade convention |
| Arbitrary horizons     | New horizons can be generated from raw minute/tick data      |
| Macro controls         | Correct release-time and historical-vintage alignment        |
| Feature registry       | Registered dependencies resolve without circular relationships |
| Model registry         | SQL, inputs, parameters, and estimator versions are preserved |
| H1 ARDL                | Appropriate integration orders, bounds cases, coefficient restrictions |
| H2 ECM                 | Correct ECT timing, horizon endpoints, interaction terms     |
| H3 ECM                 | Correct next Korean session and pre-session controls         |
| Diagnostics            | Test statistics reproduce validated reference cases          |
| Markdown               | Every report includes specifications, test statistics, decisions, and limitations |
| Reproducibility        | Same inputs, code versions, parameters, and seeds reproduce results within numerical tolerance |

Tests should also verify the behavior of missing data, insufficient sample sizes, incompatible control variables, and unavailable macroeconomic vintages.

## **18. Implementation sequence**

1. **Foundation: data contracts and ingestion**

   Implement raw and core Parquet schemas, U.S./Korean exchange calendars, FX conversion, market data normalization, and the economic-observation registry.

2. **Feature engine: SQL and dependencies**

   Implement DuckDB registration, SQL views, point-in-time joins, feature registry, control registry, and dependency resolution.

3. **Market alignment and anchors**

   Generate `daily`, `us_5min`, and `anchors` datasets. Validate quote quality, 7:00–9:20 selection, and the 9:30 benchmark.

4. **H1 estimation and statistical diagnostics**

   Implement daily ARDL–UECM, bounds inference, long-run restrictions, paper-derived diagnostics, and equilibrium-model registration.

5. **H2 and H3**

   Generate arbitrary horizon responses and directional datasets; estimate conditional and daily ECMs using versioned H1 coefficients.

6. **Experiment management and Streamlit**

   Implement run histories, cached experiments, Markdown report generation, interactive feature selection, and model comparisons.

## **19. Final architectural decisions**

| Decision             | Final specification                                          |
| -------------------- | ------------------------------------------------------------ |
| Project organization | Data-centric and hypothesis-independent                      |
| Feature generation   | SQL-first, Python for transformations not well suited to SQL |
| Source-of-truth data | Immutable raw Parquet snapshots                              |
| Query engine         | DuckDB                                                       |
| Macro controls       | Extensible, vintage-aware registry                           |
| Horizons             | Parameterized, calculated from one-minute or finer observations |
| Premarket anchors    | Liquidity-adaptive from 7:00 to 9:20 ET                      |
| Model configuration  | Python dataclasses                                           |
| H1 methodology       | Daily ARDL–UECM bounds and finite-sample inference           |
| H2 methodology       | Daily ECM and dynamic-anchor intraday adjustment             |
| H3 methodology       | Direction-specific ECM, no Granger tests                     |
| Diagnostics          | Paper-derived tests plus robustness and mixed-frequency validation |
| Research output      | Versioned Markdown reports                                   |
| Dashboard            | `app.py` using Streamlit                                     |
| Reproducibility      | Experiment registry, SQL/data/model versioning               |
| Future expansion     | Add controls, feature definitions, models, and hypotheses through registries |

**The most important design decision is to separate feature definitions from econometric hypotheses.** SQL determines the observable information set, Python research specifications determine which observations and controls to use, and the econometric modules determine what hypothesis is tested.

This allows the current H1–H3 study to remain stable while enabling extensions such as different horizons, U.S. and Korean yield-curve controls, macroeconomic surprise variables, liquidity regimes, volatility interactions, and new H4/H5 hypotheses—without rebuilding the underlying system.