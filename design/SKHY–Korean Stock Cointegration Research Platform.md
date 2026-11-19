# **SKHY–Korean Stock Cointegration Research Platform**

Consolidated Implementation Design · Version 5.0 · November 2026

**Technology:** Python · DuckDB · Parquet · SQL · Statsmodels · Streamlit
**Research methodology:** Full-Sample ARDL Bounds Testing · Fixed Thirty-Minute Horizon ECMs · Directional Price Adjustment
**Primary output:** Reproducible Markdown research reports with statistical tests, diagnostics, and conclusions.

## **1. Research objectives and design principles**

The system investigates the relationship between SK hynix's Korean ordinary shares (`000660`) and its Nasdaq-listed American Depositary Shares (`SKHY`), controlling for U.S. and Korean market conditions.

The SEC prospectus specifies that one SKHY ADS represents **0.1** Korean ordinary shares, establishing the conversion relationship required for the research.  The platform will initially implement three hypotheses:

| Hypothesis                          | Research question                                            | Main econometric method                |
| ----------------------------------- | ------------------------------------------------------------ | -------------------------------------- |
| **H1: Long-run cointegration**      | Do FX-adjusted Korean shares and SKHY maintain a statistically supported long-run price equilibrium? | Daily ARDL–UECM bounds test            |
| **H2: Error correction**            | During which fixed 30-minute window from 7:00 to 10:00 a.m. ET does SKHY adjust most strongly toward equilibrium? | Six full-sample ECMs with BIC lag selection and horizon-level multiplicity control |
| **H3: Directional price discovery** | Does equilibrium adjustment occur predominantly in SKHY or in the subsequent Korean trading session? | Two direction-specific ECMs            |

No Granger causality tests are included.

### **Fixed design requirements**

| Component                       | Requirement                                                 |
| ------------------------------- | ----------------------------------------------------------- |
| H2 fixed windows                | 7:00–7:30, 7:30–8:00, 8:00–8:30, 8:30–9:00, 9:00–9:30, 9:30–10:00 ET |
| H2 response horizon             | Six fixed 30-minute intervals per eligible date             |
| H1/H2 estimation window         | Complete available sample; no rolling-forward re-estimation |
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
│   │   └── experiments/
│   │       └── run_id=.../
│   │           ├── h1.parquet
│   │           └── h2.parquet
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
│   │   ├── price_parity.py
│   │   ├── contracts.py
│   │   └── validators.py
│   │
│   ├── hypotheses/
│   │   └── overnight_ablation.py   # Retained bivariate H1 and SOXX-gauge H2
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

### **4.1 Five principal research datasets**

| Dataset   | Row granularity                         | Primary content                                              | Consumer          |
| --------- | --------------------------------------- | ------------------------------------------------------------ | ----------------- |
| `daily`   | One paired U.S. trading session         | ADR close, last known Korean close, FX, market controls      | H1 and H3         |
| `us_5min` | One completed SKHY five-minute interval | OHLCV, quotes, spreads, volume, FX, known Korean reference   | Reusable market-quality analysis |
| `h1`      | One eligible daily observation          | Price levels, differences, lags, selected controls           | ARDL–UECM         |
| `h2`      | One date × one of six fixed 30-minute windows | Full-sample H1 ECT and synchronized SKHY, SOXX and FX returns | Primary H2 comparison |
| `h3`      | One directional target observation      | Pre-response ECT, completed source-market return, target-market return | Directional ECM   |

The first two are reusable foundation datasets. Model-specific datasets are materialized in versioned Parquet under the retained experiment run directory.

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

### `H1_long_run_coefficients.parquet` **— fixed full-sample equilibrium**

Primary fields:

```
parameter                  VARCHAR
estimate                   DOUBLE
std_error                  DOUBLE
p_value                    DOUBLE
q_value                    DOUBLE
conf_low                   DOUBLE
conf_high                  DOUBLE
```

The artifact contains the intercept and slope from the single H1 model estimated on the complete sample. H2 uses these same values for every date.

### `h2` **— primary fixed-window H2 observations**

Primary fields:

```
date_us                 DATE
anchor_type             VARCHAR  -- regular_0930
anchor_ts_utc           TIMESTAMPTZ
horizon_min             INTEGER  -- 30
endpoint_ts_utc         TIMESTAMPTZ  -- 10:00 ET
adr_return_h            DOUBLE
soxx_return_h           DOUBLE
usdkrw_return_h         DOUBLE
ect_anchor              DOUBLE
kr_return_known         DOUBLE
h1_model_id             VARCHAR
valid_outcome           BOOLEAN
```

The dependent variable is the SKHY log return from 9:30 to 10:00 a.m. ET. `ect_anchor` uses the fixed full-sample H1 intercept and slope.

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

For example, a U.S. CPI figure released at 8:30 a.m. ET may be included in the 9:00–9:30 window, but it must not appear in features for an earlier window on the same day.

This rule applies equally to Korean macroeconomic releases, U.S. market indices, FX, and other controls.

## **7. SQL-driven feature engine**

The engine must support two types of change:

1. **Parameter changes:** Change fixed windows, control selection, lags, or sample periods without editing SQL files.
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

}
```

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

### **7.3 Six fixed 30-minute windows**

The primary H2 feature builder constructs synchronized SKHY, SOXX, and USDKRW returns for six fixed 30-minute windows from 7:00 through 10:00 a.m. ET. Each row requires prices at the exact start and end boundaries. Missing endpoints remain invalid rather than being replaced by a stale observation. The production H2 run sets `horizons=(30,)`; the six pre-specified window identifiers determine the timing comparison.

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

The H1 long-run equilibrium is always bivariate. Rolling lag selection may choose the autoregressive and distributed lag orders, but it may not remove `log_kr_usd` from the level relationship.

### **9.2 Full-sample H1 coefficient artifact**

Estimate H1 once on all complete historical observations. BIC selects the ARDL orders, after which the UECM bounds test and long-run coefficient transformation produce \(\hat c\), \(\hat\theta\), and the daily adjustment coefficient.

Write the fixed long-run coefficients to `h1_model.parquet`. The retained H2
equilibrium error is known before the current U.S. opening:

\[
ECT_d=\log(P^{SKHY}_{d-1,close})
-\hat c-\hat\theta\log(P^{KR,USD}_d).
\]

The same \(\hat c\) and \(\hat\theta\) apply to every H2 date. H1 supplies the
equilibrium benchmark; H2 uses it to forecast the overnight return.

### **9.3 H1 pipeline**

```
src/hypotheses/overnight_ablation.py
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

### **9.4 Required H1 results**

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

## **10. H2 — Retained overnight error-correction forecast**

### **10.1 Target and cutoff**

The only retained H2 target is the full SKHY previous-close-to-09:30-open log
return:

\[
r^{ON}_{d}=\log(P^{SKHY}_{d,09{:}30\ open}/P^{SKHY}_{d-1,close}).
\]

The forecast is frozen at 09:25 ET. H2 uses the completed Korean daytime
return, the previous-close H1 equilibrium error, the SOXX prior-close-to-07:00
return, and the SOXX 07:00-to-09:25 return.

### **10.2 Final H2 equation**

\[
\boxed{
\widehat r^{ON}_d=
0.005001
+0.340554r^{KR,day}_d
-0.226168ECT_d
+0.779940r^{SOXX,close\rightarrow07{:}00}_d
+1.184007r^{SOXX,07{:}00\rightarrow09{:}25}_d
}
\]

Return forecasting uses H2, not H1 directly. H1 estimates \(\hat c\) and
\(\hat\theta\); H2 combines the resulting ECT with current Korean and SOXX
information. BIC selected \(p=0,q=0\), so no lagged return or lagged-gauge
terms enter the retained equation.

### **10.3 Retained results**

| Measure | Result |
| --- | ---: |
| Complete observations | 88 |
| Expanding-window forecasts | 48 |
| RMSE | 0.012906 |
| MAE | 0.010055 |
| Directional accuracy | 95.8% |
| OOS \(R^2\) versus zero | 0.8930 |
| H2 adjustment coefficient | −0.226168 |
| H2 adjustment p-value | \(8.63\times10^{-7}\) |

The negative, highly significant adjustment coefficient supports overnight
movement toward the bivariate H1 equilibrium. Forecast validation re-estimates
H2 on expanding samples containing only observations before each prediction.

### **10.4 Executable strategy**

The trading strategy retains the 09:25-to-open design. It enters at the 09:25
SKHY premarket open, applies protective stop-loss/take-profit rules through
09:27, submits an opposite-side MOO before 09:28 for survivors, and exits at
09:30. It credits only the 09:25–09:30 return. Transaction costs, slippage,
square-root impact, volatility-scaled sizing, and drawdown controls are
included. Full execution details are in `design/BACKTEST_DESIGN.md`.

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

    horizons: tuple[int, ...] = (30,)

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
        30, 40, 50, 60, 70, 80, 90
    )
``
```

The specification must be validated before execution. A requested control must exist in the registry, every H2 horizon must be a positive multiple of ten, and no evaluation observation may occur before the necessary training history.

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
| `H2_error_correction.md` | Six fixed-window ECMs, per-window BIC lag selection, HAC inference, multiplicity adjustment, and selected horizon |
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

H2 reports must additionally document valid observations for every fixed window and the multiplicity-adjusted result across all six windows.

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

The production engine must estimate `h1` and write the fixed long-run coefficient artifact before estimating H2. If H1 is not statistically supported, H2 must be described as an estimated-equilibrium exercise rather than a validated cointegration ECM.

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
| Fixed H2 alignment    | SKHY, SOXX and USDKRW use identical boundaries within each of the six windows |
| Full-sample H1        | One reproducible coefficient artifact is used by every H2 row |
| Secondary horizons    | Requested cumulative endpoints are generated independently from raw minute data |
| Macro controls         | Correct release-time and historical-vintage alignment        |
| Feature registry       | Registered dependencies resolve without circular relationships |
| Model registry         | SQL, inputs, parameters, and estimator versions are preserved |
| H1 ARDL                | Appropriate integration orders, bounds cases, coefficient restrictions |
| H2 ECM                 | Fixed full-sample ECT, BIC-selected p/q orders, and HAC inference |
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

3. **Market alignment and fixed windows**

   Generate `daily` and `us_5min`, then validate exact one-minute endpoints for all six fixed 30-minute windows.

4. **H1 estimation and statistical diagnostics**

   Implement daily ARDL–UECM, bounds inference, long-run restrictions, paper-derived diagnostics, and equilibrium-model registration.

5. **H2 and H3**

   Estimate H1 once on the complete sample, materialize its fixed long-run coefficients, and estimate six fixed-window H2 equations with BIC-selected p/q orders and horizon-level multiplicity control.

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
| H2 timing windows    | Six fixed 30-minute windows from 7:00 to 10:00 ET            |
| Model configuration  | Python dataclasses                                           |
| H1 methodology       | Daily ARDL–UECM bounds and finite-sample inference           |
| H2 methodology       | Full-sample H1 with six fixed 30-minute adjustment equations |
| H3 methodology       | Direction-specific ECM, no Granger tests                     |
| Diagnostics          | Paper-derived tests plus robustness and mixed-frequency validation |
| Research output      | Versioned Markdown reports                                   |
| Dashboard            | `app.py` using Streamlit                                     |
| Reproducibility      | Experiment registry, SQL/data/model versioning               |
| Future expansion     | Add controls, feature definitions, models, and hypotheses through registries |

**The most important design decision is to separate feature definitions from econometric hypotheses.** SQL determines the observable information set, Python research specifications determine which observations and controls to use, and the econometric modules determine what hypothesis is tested.

This allows the current H1–H3 study to remain stable while enabling extensions such as different horizons, U.S. and Korean yield-curve controls, macroeconomic surprise variables, liquidity regimes, volatility interactions, and new H4/H5 hypotheses—without rebuilding the underlying system.
