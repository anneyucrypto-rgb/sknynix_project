# SK Hynix Cross-Border Price-Discovery Design

## 1. Research objective

The primary question is whether the most recently available Korean SK hynix closing-price return improves prediction of future U.S.-listed SKHY returns.

The Korean series is daily, while SKHY and the U.S. controls are observed in five-minute bars. The design therefore uses a mixed-frequency, event-time autoregressive distributed-lag model (ARDL), not a standard five-minute VARX.

For every matched Korean/U.S. trading day, estimate separate models for future U.S. horizons of 5, 10, 30, 60, 90, and 120 minutes. The Korean daily observation appears once per event day; it must not be repeated across every five-minute row and treated as independent data.

The analysis has four goals:

1. Estimate the immediate Korean-to-U.S. price-discovery effect.
2. Test whether the Korean-return block improves the unrestricted equation.
3. Evaluate whether the result is robust to microstructure, market, and macroeconomic controls.
4. Measure whether the Korean variables improve genuine out-of-sample forecasts.

## 2. Current project and raw data

```text
sknynix_project/
├── data/
│   ├── us/
│   │   ├── SKHY_5min.parquet
│   │   ├── SPY_5min.parquet
│   │   ├── SOXX_5min.parquet
│   │   └── QQQ_5min.parquet
│   ├── korea/
│   │   └── 000660_daily.parquet
│   ├── fx/
│   │   └── USDKRW_5min.parquet
│   ├── quotes/
│   │   └── SKHY_quotes.parquet       # optional; plan entitlement required
│   └── skhynix.duckdb                 # persistent materialized database
├── ingestion/
│   ├── massive_client.py
│   └── download.py
├── models/
│   ├── granger.py
│   ├── varx.py
│   └── arbitrage.py
├── backtest/
│   └── backtest.py
├── design/
│   └── data.md
├── database.py                        # UTC normalization and event-time joins
└── requirements.txt
```

Raw datasets remain separate during ingestion because they have different frequencies, trading calendars, time zones, and information-availability times.

| Dataset | Source | Frequency | Role |
|---|---|---:|---|
| SKHY | Massive Stocks | 5 minutes | U.S. response variable, volume, and microstructure features |
| SPY | Massive Stocks | 5 minutes | Broad-market control |
| SOXX | Massive Stocks | 5 minutes | Semiconductor-sector control |
| QQQ | Massive Stocks | 5 minutes | Technology-market robustness control |
| 000660.KS | Korean-market provider | Daily | Korean price-discovery treatment |
| USD/KRW | Massive Forex | 5 minutes | Currency/macroeconomic control |
| SKHY quotes | Massive Stocks | Tick level | Optional spread and imbalance controls |

### Current validated coverage

| Dataset | Rows | First observation | Latest observation |
|---|---:|---|---|
| SKHY 5-minute | 16,276 | 2026-07-13 08:00 UTC | 2026-11-10 00:55 UTC |
| SPY 5-minute | 16,485 | 2026-07-10 08:00 UTC | 2026-11-10 00:55 UTC |
| SOXX 5-minute | 15,985 | 2026-07-10 08:00 UTC | 2026-11-10 00:55 UTC |
| QQQ 5-minute | 16,505 | 2026-07-10 08:00 UTC | 2026-11-10 00:55 UTC |
| USD/KRW 5-minute | 23,784 | 2026-07-10 00:00 UTC | 2026-11-09 23:55 UTC |
| 000660.KS daily | 82 | 2026-07-10 | 2026-11-10 |

Historical SKHY quotes are not currently available because the Massive account does not include the required entitlement. Quote features are therefore optional and must not be required for the primary model.

### Persistent DuckDB

Build or refresh the database with:

```powershell
python database.py
```

The permanent file is `data/skhynix.duckdb`. It materializes the Parquet data rather than storing views that depend on the source files. The DuckDB connection is configured to display `TIMESTAMP WITH TIME ZONE` values in UTC.

Principal tables are:

```text
skhy_5min
spy_5min
soxx_5min
qqq_5min
usdkrw_5min
korea_daily
skhy_quotes
us_event_anchors
event_korea_asof
database_metadata
```

`event_korea_asof` contains both the 04:00 premarket and 09:30 regular-session anchors and the latest Korean close whose UTC availability timestamp is no later than the anchor.

Inspect a schema directly with:

```sql
DESCRIBE event_korea_asof;
```

#### Five-minute price tables

`spy_5min`, `soxx_5min`, `qqq_5min`, and `usdkrw_5min` have the same fields:

| Field | DuckDB type | Meaning |
|---|---|---|
| `ticker` | `VARCHAR` | Provider ticker. USD/KRW is stored as `C:USDKRW`. |
| `timestamp_utc` | `TIMESTAMP WITH TIME ZONE` | Start timestamp of the five-minute bar, stored as a UTC market instant. |
| `timestamp_ny` | `TIMESTAMP` | The same instant converted to New York local wall time. This field is daylight-saving aware but does not itself carry a zone suffix. |
| `open` | `DOUBLE` | First qualifying price in the five-minute interval. |
| `high` | `DOUBLE` | Highest qualifying price in the interval. |
| `low` | `DOUBLE` | Lowest qualifying price in the interval. |
| `close` | `DOUBLE` | Last qualifying price in the interval. |
| `vwap` | `DOUBLE` | Provider-calculated volume- or quote-weighted average price for the interval. |

`skhy_5min` contains all fields above plus:

| Field | DuckDB type | Meaning |
|---|---|---|
| `volume` | `DOUBLE` | Provider-reported SKHY trading volume in the five-minute bar. The adjusted aggregate feed can produce non-integer values. |
| `transactions` | `BIGINT` | Number of qualifying SKHY transactions contributing to the bar. |

#### `korea_daily`

This table contains the Korean daily observations and their information-availability timestamps.

| Field | DuckDB type | Meaning |
|---|---|---|
| `ticker` | `VARCHAR` | Korean security ticker, currently `000660.KS`. |
| `trading_date` | `DATE` | Korean exchange trading date. |
| `close_timestamp_kst` | `TIMESTAMP` | Korean close represented as 15:30 local `Asia/Seoul` wall time. |
| `availability_timestamp_utc` | `TIMESTAMP WITH TIME ZONE` | The Korean close converted to its UTC market instant; the observation must not be used before this timestamp. |
| `open` | `DOUBLE` | Korean daily opening price in KRW. |
| `high` | `DOUBLE` | Korean daily high price in KRW. |
| `low` | `DOUBLE` | Korean daily low price in KRW. |
| `close` | `DOUBLE` | Korean daily unadjusted closing price in KRW. |
| `adjusted_close` | `DOUBLE` | Korean close adjusted by the source for applicable corporate actions. |
| `volume` | `BIGINT` | Korean daily traded volume. |
| `prior_adjusted_close` | `DOUBLE` | Previous available Korean trading day's adjusted close. |
| `log_return` | `DOUBLE` | `ln(adjusted_close / prior_adjusted_close)`; null for the first observation or invalid non-positive prices. |

#### `skhy_quotes`

This table is currently empty because the Massive account is not entitled to historical quote data. Its stable schema allows downstream code to run when quote data is later added.

| Field | DuckDB type | Meaning |
|---|---|---|
| `ticker` | `VARCHAR` | Quoted security ticker. |
| `quote_timestamp_utc` | `TIMESTAMP WITH TIME ZONE` | SIP quote timestamp converted to UTC. |
| `participant_timestamp` | `BIGINT` | Raw nanosecond exchange/participant timestamp supplied by the provider. |
| `sequence_number` | `BIGINT` | Provider sequence number used to order quote events within a session. |
| `bid_price` | `DOUBLE` | National best bid price. |
| `bid_size` | `DOUBLE` | Provider-reported quantity available at the best bid. |
| `bid_exchange` | `BIGINT` | Provider exchange identifier for the best bid. |
| `ask_price` | `DOUBLE` | National best ask price. |
| `ask_size` | `DOUBLE` | Provider-reported quantity available at the best ask. |
| `ask_exchange` | `BIGINT` | Provider exchange identifier for the best ask. |
| `conditions` | `VARCHAR` | JSON-encoded list of provider quote-condition codes. |
| `indicators` | `VARCHAR` | JSON-encoded list of provider quote-indicator codes. |
| `tape` | `BIGINT` | Consolidated tape identifier. |

#### `us_event_anchors`

This table contains one event row per U.S. trading day and anchor type.

| Field | DuckDB type | Meaning |
|---|---|---|
| `event_date_ny` | `DATE` | U.S. event date in `America/New_York`. |
| `anchor_type` | `VARCHAR` | `premarket_0400` or `regular_0930`. |
| `anchor_timestamp_utc` | `TIMESTAMP WITH TIME ZONE` | UTC instant of the selected first valid SKHY bar at or after the scheduled anchor. |
| `anchor_timestamp_ny` | `TIMESTAMP` | Selected anchor instant expressed as New York local wall time. |
| `actual_anchor_time_ny` | `TIME` | Actual New York clock time of the selected bar. This can be later than the scheduled time if an exact bar is missing. |
| `exact_scheduled_anchor` | `BOOLEAN` | True when the selected bar occurs exactly at 04:00 or 09:30, as appropriate. |
| `skhy_ticker` | `VARCHAR` | SKHY ticker stored on the selected bar. |
| `skhy_open` | `DOUBLE` | SKHY open price of the selected five-minute anchor bar. |
| `skhy_high` | `DOUBLE` | SKHY high price of the anchor bar. |
| `skhy_low` | `DOUBLE` | SKHY low price of the anchor bar. |
| `skhy_close` | `DOUBLE` | SKHY close price of the anchor bar. |
| `skhy_vwap` | `DOUBLE` | SKHY VWAP of the anchor bar. |
| `skhy_volume` | `DOUBLE` | SKHY volume of the anchor bar. |
| `skhy_transactions` | `BIGINT` | Number of qualifying transactions in the anchor bar. |

#### `event_korea_asof`

This is the primary event-alignment table. It contains all `us_event_anchors` fields plus the latest Korean daily observation whose availability timestamp is no later than the U.S. anchor.

Anchor-side fields:

| Field | DuckDB type | Meaning |
|---|---|---|
| `event_date_ny` | `DATE` | U.S. event date in New York. |
| `anchor_type` | `VARCHAR` | Premarket or regular-session anchor identifier. |
| `anchor_timestamp_utc` | `TIMESTAMP WITH TIME ZONE` | Selected U.S. anchor instant in UTC. |
| `anchor_timestamp_ny` | `TIMESTAMP` | Selected anchor expressed in New York wall time. |
| `actual_anchor_time_ny` | `TIME` | Actual New York clock time of the selected bar. |
| `exact_scheduled_anchor` | `BOOLEAN` | Whether the selected bar exactly matches the scheduled anchor. |
| `skhy_ticker` | `VARCHAR` | U.S. instrument ticker. |
| `skhy_open` | `DOUBLE` | SKHY anchor-bar open. |
| `skhy_high` | `DOUBLE` | SKHY anchor-bar high. |
| `skhy_low` | `DOUBLE` | SKHY anchor-bar low. |
| `skhy_close` | `DOUBLE` | SKHY anchor-bar close. |
| `skhy_vwap` | `DOUBLE` | SKHY anchor-bar VWAP. |
| `skhy_volume` | `DOUBLE` | SKHY anchor-bar volume. |
| `skhy_transactions` | `BIGINT` | SKHY anchor-bar transaction count. |

Matched Korean fields:

| Field | DuckDB type | Meaning |
|---|---|---|
| `korea_ticker` | `VARCHAR` | Matched Korean ticker. |
| `korea_trading_date` | `DATE` | Trading date of the matched Korean observation. |
| `korea_close_timestamp_kst` | `TIMESTAMP` | Matched Korean close at 15:30 Seoul wall time. |
| `korea_availability_timestamp_utc` | `TIMESTAMP WITH TIME ZONE` | UTC instant when the matched Korean observation became available. |
| `korea_open` | `DOUBLE` | Matched Korean daily open in KRW. |
| `korea_high` | `DOUBLE` | Matched Korean daily high in KRW. |
| `korea_low` | `DOUBLE` | Matched Korean daily low in KRW. |
| `korea_close` | `DOUBLE` | Matched Korean unadjusted close in KRW. |
| `korea_adjusted_close` | `DOUBLE` | Matched Korean adjusted close. |
| `korea_volume` | `BIGINT` | Matched Korean daily volume. |
| `korea_log_return` | `DOUBLE` | Matched Korean adjusted-close log return. |
| `korea_stale_days` | `BIGINT` | Calendar-day difference between the Korean trading date and U.S. event date. Zero normally denotes a same-date new Korean close. |
| `korea_new_close` | `BOOLEAN` | True when the matched Korean trading date equals the U.S. event date. |
| `availability_check_passed` | `BOOLEAN` | True when the Korean availability timestamp is no later than the U.S. anchor; this must always be true for a valid match. |

#### `database_metadata`

This one-row table records how the current database file was built.

| Field | DuckDB type | Meaning |
|---|---|---|
| `built_at_utc` | `TIMESTAMP WITH TIME ZONE` | UTC time at which `database.py` last completed the build. |
| `data_directory` | `VARCHAR` | Absolute directory containing the source Parquet files. |
| `database_path` | `VARCHAR` | Absolute path of the persistent DuckDB file. |
| `timezone_policy` | `VARCHAR` | Human-readable description of the database timestamp policy. |
| `korean_close_policy` | `VARCHAR` | Human-readable description of the Korean information-availability rule. |
| `quotes_loaded` | `BOOLEAN` | Whether non-empty historical SKHY quote data was materialized. |

## 3. Event-time alignment

### 3.1 Korean information timestamp

For each Korean trading day, timestamp the daily observation at the actual Korean market close in `Asia/Seoul`, then convert it to UTC. The following variables become available only at that timestamp:

- Korean closing price
- Korean daily log return
- Korean daily high-low range
- Korean daily volume

Never join Korean data by calendar date alone. Join it to the U.S. event with a backward as-of match using the availability timestamp.

### 3.2 U.S. event anchor

The primary event anchor is the first valid SKHY five-minute bar at or after 04:00 `America/New_York`, when U.S. premarket trading begins. This captures the earliest U.S. response after the Korean close.

Use the 09:30 regular-session open as a robustness anchor because it has greater liquidity, although some Korean information may already have been incorporated during premarket trading.

For every U.S. event day (d), define the anchor time as (t_{0,d}). Match it to the latest Korean close whose availability timestamp is no later than (t_{0,d}).

### 3.3 Trading-calendar rules

Primary sample:

- Retain event days with a new Korean close and a valid U.S. event window.
- Drop days on which the required U.S. horizon is incomplete.
- Record Korean and U.S. holiday indicators.

Robustness sample:

- Permit an as-of matched stale Korean close.
- Add `korea_stale_days` and `korea_new_close` indicators.
- Do not silently treat a stale Korean close as new information.

## 4. Return definitions

Do not fit raw Korean or U.S. price levels in the primary ARDL. Use log returns to reduce non-stationarity and avoid scale differences between KRW and USD prices.

### 4.1 Korean treatment

The most recent Korean closing-price return is

\[
r^{KR}_{d} = \log\left(\frac{C^{KR}_{d}}{C^{KR}_{d-1}}\right).
\]

Create:

```text
kr_return_0       most recent Korean daily return known by the U.S. anchor
kr_return_lag1    previous Korean trading-day return
kr_return_lag2    Korean return two trading days earlier
```

`kr_return_0` is the principal treatment. Although it can have the same calendar date as the U.S. event, it is temporally prior because the Korean market has already closed.

### 4.2 U.S. future-return targets

For each horizon $h$ in 5, 10, 30, 60, 90, and 120 minutes, define

\[
r^{US,h}_{d} = \log\left(\frac{P^{US}_{d,t_0+h}}{P^{US}_{d,t_0}}\right).
\]

Create six targets:

```text
skhy_fwd_return_5m
skhy_fwd_return_10m
skhy_fwd_return_30m
skhy_fwd_return_60m
skhy_fwd_return_90m
skhy_fwd_return_120m
```

Use the SKHY close of the five-minute bar ending at the required timestamp. If no exact bar exists, do not use a later observation; either use a documented backward match within a narrow tolerance or mark the target missing.

### 4.3 Total-response robustness target

The primary target measures the post-anchor return. A secondary target should capture information incorporated before the anchor:

\[
r^{US,total,h}_{d} = \log\left(\frac{P^{US}_{d,t_0+h}}{C^{US}_{d-1}}\right).
\]

Decompose it into:

```text
overnight_gap       log(P_US_at_t0 / previous_US_close)
post_anchor_return  log(P_US_at_t0_plus_h / P_US_at_t0)
total_response      overnight_gap + post_anchor_return
```

This distinguishes price discovery that occurs before the U.S. anchor from adjustment during the measured U.S. horizon.

## 5. Autoregressive distributed-lag model

Estimate one single-equation ARDL for every horizon and every control ablation. Do not place all six nested horizon returns in one endogenous vector.

For horizon $h$, the unrestricted equation is

\[
r^{US,h}_{d} =
\alpha_h
+ \sum_{i=1}^{p}\phi_{i,h}r^{US,h}_{d-i}
+ \sum_{j=0}^{q}\beta_{j,h}r^{KR}_{d-j}
+ \rho_h r^{SKHY,pre}_{d}
+ \Gamma_h X_d
+ \epsilon_{d,h}.
\]

Where:

- $p$ is the autoregressive order of the previous daily event responses.
- $q$ is the distributed-lag order of the Korean daily return.
- $r^{SKHY,pre}_{d}$ is the most recent intraday SKHY return ending at the anchor.
- $X_d$ contains the selected ablation controls.

The short U.S. horizon defines the dependent variable. It does not determine the daily ARDL lag order.

### 5.1 Restricted equation

The restricted equation excludes the Korean-return block:

\[
r^{US,h}_{d} =
\alpha_h
+ \sum_{i=1}^{p}\phi_{i,h}r^{US,h}_{d-i}
+ \rho_h r^{SKHY,pre}_{d}
+ \Gamma_h X_d
+ \epsilon_{d,h}.
\]

### 5.2 Unrestricted-equation test

Test the Korean coefficients jointly:

\[
H_0: \beta_{0,h}=\beta_{1,h}=\cdots=\beta_{q,h}=0.
\]

Rejecting $H_0$ means the latest available Korean closing-price returns improve prediction of the future SKHY horizon return beyond SKHY's own history and the selected controls.

Because `kr_return_0` is same-date but temporally prior, describe the result as asynchronous-market or event-time Granger causality. Report the same-day coefficient separately from conventional lagged Korean coefficients.

## 6. Autoregressive features

The following variables measure different types of persistence and should not be confused:

| Feature | Frequency | Purpose |
|---|---:|---|
| `us_response_h_lag1d` | Daily event | Previous day's response at the same horizon; primary AR term |
| `skhy_pre_event_return_5m` | 5 minutes | Immediate momentum before the anchor |
| `skhy_pre_event_return_30m` | 30 minutes | More stable short-term momentum; preferred intraday control |
| `skhy_return_lag1d` | Daily | Previous full-session persistence; optional robustness control |

Use `skhy_pre_event_return_30m` as the primary recent-U.S. feature when the 09:30 anchor is used:

\[
r^{SKHY,pre30m}_{d} = \log\left(\frac{P^{SKHY}_{d,t_0}}{P^{SKHY}_{d,t_0-30m}}\right).
\]

Use `skhy_pre_event_return_5m` as a sensitivity specification. Do not include the five-minute return, the 30-minute cumulative return, and all six component lags simultaneously in the small current sample.

At the 04:00 premarket anchor, preceding same-day SKHY bars may not exist or may be illiquid. In that specification, use the previous U.S. close-to-anchor return and an opening-liquidity indicator instead of manufacturing missing intraday lags.

## 7. Control groups and ablation study

All predictive controls must have an information timestamp no later than the U.S. event anchor. Never use control returns measured during the future target interval in the predictive Granger model.

### 7.1 Microstructure controls

```text
skhy_pre_event_return_30m
skhy_pre_event_rv_30m
skhy_pre_event_volume_z
skhy_vwap_deviation
```

Optional when quote entitlement becomes available:

```text
skhy_relative_spread
skhy_bid_ask_imbalance
skhy_spread_volatility
```

Suggested transformations:

```python
skhy_range = np.log(skhy_high / skhy_low)
skhy_vwap_deviation = np.log(skhy_close / skhy_vwap)
skhy_log_volume = np.log1p(skhy_volume)
```

Normalize volume relative to the historical distribution for the same five-minute time-of-day bin. Raw volume has a strong intraday seasonal pattern.

### 7.2 Market controls

```text
soxx_pre_event_return_30m
spy_pre_event_return_30m
```

Use QQQ as a replacement robustness control. Do not include SPY, QQQ, and SOXX together in the primary small-sample model without checking multicollinearity.

### 7.3 Macroeconomic controls

```text
usdkrw_return_latest
vix_change_lag1d
macro_announcement
```

Optional extended controls:

```text
dxy_return_lag1d
treasury_2y_change
treasury_10y_change
```

USD/KRW is already available. VIX, Treasury, DXY, and announcement-calendar data require additional ingestion before they can enter the model.

### 7.4 Ablation matrix

For every horizon, estimate restricted and unrestricted versions of each specification:

| Model | AR base | Micro | Market | Macro |
|---|---:|---:|---:|---:|
| A0 | Yes | No | No | No |
| A1 | Yes | Yes | No | No |
| A2 | Yes | No | Yes | No |
| A3 | Yes | No | No | Yes |
| A4 | Yes | Yes | Yes | No |
| A5 | Yes | Yes | No | Yes |
| A6 | Yes | No | Yes | Yes |
| A7 | Yes | Yes | Yes | Yes |

The AR base is always present. The Korean block is present only in the unrestricted version.

Use the same complete-case event dates for all models in an ablation comparison. Otherwise, changes can be caused by different samples rather than different control groups.

Treat A7 as the primary confirmatory specification and A0-A6 as robustness ablations. If the full model is too large for the available sample, define A4 as the temporary primary model and treat macro controls as a later extension.

## 8. Stationarity and integration order

AIC and BIC select lag order; they do not determine integration order. Determine integration order using unit-root and stationarity tests.

Use ADF and KPSS together:

| Result | Interpretation |
|---|---|
| ADF rejects and KPSS does not reject | Treat as stationary, $I(0)$ |
| ADF does not reject and KPSS rejects | Difference or otherwise transform the series |
| Tests disagree | Inspect breaks and report sensitivity specifications |

Expected transformations:

| Variable | Primary form | Expected order |
|---|---|---:|
| Korean closing price | `log(kr_close)` | Often $I(1)$ |
| Korean daily return | `diff(log(kr_close))` | Usually $I(0)$ |
| SKHY price | `log(skhy_price)` | Often $I(1)$ |
| SKHY future return | `log(P[t+h] / P[t])` | Usually $I(0)$ |
| USD/KRW level | `log(usdkrw)` | Often $I(1)$ |
| USD/KRW return | `diff(log(usdkrw))` | Usually $I(0)$ |

Do not difference a return series again unless the diagnostics show that it remains non-stationary.

## 9. AIC/BIC lag selection

With the current 82-day sample, use the conservative candidate grid

```python
P_CANDIDATES = (0, 1, 2)
Q_CANDIDATES = (0, 1, 2)
```

Here, `q=0` includes `kr_return_0`. Larger values add prior Korean daily returns.

For every horizon and ablation:

1. Trim all candidate datasets to a common sample based on the maximum lag.
2. Fit every unrestricted ARDL candidate over the $(p,q)$ grid.
3. Select the primary $(p,q)$ using minimum BIC.
4. Select an AIC model as a sensitivity specification.
5. Fit the restricted equation using the same selected $p$, controls, and observations.
6. Remove only the Korean-return terms from the restricted model.

Do not select separate autoregressive orders for the restricted and unrestricted equations; they would no longer form a valid nested comparison.

BIC is primary because the sample is small and overfitting risk is high. AIC is reported as a less parsimonious robustness choice.

## 10. Statistical inference

For every horizon and ablation, report:

- Selected BIC and AIC lag orders
- Korean coefficient estimates and confidence intervals
- Joint Wald or nested-model F statistic
- Conventional and bootstrap p-values
- Restricted and unrestricted adjusted (R^2)
- Incremental (R^2)
- Restricted and unrestricted out-of-sample loss
- Directional accuracy
- Residual autocorrelation and heteroskedasticity diagnostics
- Number of event days

Lag selection precedes the hypothesis test, so ordinary post-selection p-values can be optimistic. Use a daily block bootstrap that repeats lag selection inside every bootstrap replication.

Correct the six primary A7 horizon p-values using Holm correction. If all ablation results are treated as confirmatory tests, adjust across the complete horizon-by-ablation family instead.

## 11. Walk-forward evaluation

Use expanding-window or rolling-window evaluation by event day. Never use a random train/test split.

Within each training fold:

1. Re-estimate normalization parameters.
2. Repeat AIC/BIC lag selection.
3. Fit restricted and unrestricted equations.
4. Forecast the next event day.

Compare:

```text
restricted_oos_mse
unrestricted_oos_mse
delta_oos_mse
restricted_directional_accuracy
unrestricted_directional_accuracy
```

Evidence for predictive causality is strongest when the Korean block is jointly significant and improves out-of-sample performance.

## 12. Required model output

Store one row per horizon, ablation, and selection criterion:

```text
horizon_minutes
event_anchor
ablation
selection_criterion
p_order
q_order
n_obs
beta_kr_current
beta_kr_lag1
beta_kr_lag2
wald_stat
wald_pvalue
bootstrap_pvalue
adjusted_pvalue
restricted_adj_r2
unrestricted_adj_r2
delta_adj_r2
restricted_oos_mse
unrestricted_oos_mse
delta_oos_mse
restricted_directional_accuracy
unrestricted_directional_accuracy
residual_diagnostics_pass
```

## 13. Price data versus quote data

Five-minute price bars summarize executed trades through open, high, low, close, volume, and VWAP. Quote data records available bid and ask prices and sizes before execution. They are related but not interchangeable.

When quote access becomes available, aggregate tick quotes into five-minute features:

```python
midpoint = (bid_price + ask_price) / 2
relative_spread = (ask_price - bid_price) / midpoint
imbalance = (bid_size - ask_size) / (bid_size + ask_size)
```

Quote features belong in the microstructure-control ablation. The primary model must continue to work without them.

## 14. Limitations and next data step

The current sample contains only 82 Korean daily observations. This is sufficient to construct and debug the event-time feature pipeline but is weak for six horizons, lag selection, and multiple control ablations.

Until the history is extended:

- Limit $p$ and $q$ to at most two.
- Keep the control groups parsimonious.
- Treat bootstrap intervals and out-of-sample results as exploratory.
- Avoid interpreting statistical insignificance as evidence of no transmission.

The next data priority is at least 12 months, preferably 24 months, of Korean daily and corresponding U.S. five-minute history. Macro-control ingestion should follow after the longer core sample is available.
