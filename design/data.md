# SKHY event-indexed feature design

## Research equation

For each U.S. event day `d`, anchor `t0`, and horizon `h`, estimate a separate
ARDL equation:

\[
r_d^{US,h}=\alpha_h
+\sum_{i=1}^{p}\phi_{i,h}r_{d-i}^{US,h}
+\sum_{j=0}^{q}\beta_{j,h}r_{d-j}^{KR}
+\rho_h r_d^{SKHY,pre}
+\Gamma_h X_d+\varepsilon_{d,h}.
\]

The model grid is:

- `t0 ∈ {04:00, 09:30}` in `America/New_York`.
- `h ∈ {5, 10, 30, 60, 90, 120}` minutes.
- One row per `(event_date_ny, anchor_type)` in the feature table.
- One separate model per anchor, horizon, response family, and control ablation.

`p` is the number of same-anchor, same-horizon U.S. event-response lags. `q`
is the number of Korean daily-return lags; `q = 0` includes the latest Korean
return. Select `p` and `q` within the training sample using BIC as the primary
criterion and AIC as a sensitivity criterion.

## Storage

```text
data/
├── raw/
│   ├── us/
│   │   ├── SKHY_1min.parquet
│   │   ├── SPY_1min.parquet
│   │   ├── SOXX_1min.parquet
│   │   ├── QQQ_1min.parquet
│   │   ├── SKHY_first_trades.parquet
│   │   └── SKHY_first_quotes.parquet
│   ├── korea/000660_daily.parquet
│   └── fx/USDKRW_1min.parquet
└── processed/
    ├── granger_panel.parquet
    └── arbitrage_panel_1min.parquet
```

`granger_panel.parquet` is the only ARDL/Granger/VARX feature table. There are
no database catalog or metadata tables. `arbitrage_panel_1min.parquet` contains
minute-level cross-market parity and premium-residual features.

Build both panels with:

```powershell
python ingestion/database.py
```

## Event indexing and information availability

| `anchor_type` | Scheduled New York time | Purpose |
|---|---:|---|
| `premarket_0400` | 04:00 | Earliest U.S. premarket response |
| `regular_0930` | 09:30 | Higher-liquidity robustness anchor |

`timestamp_utc` is the scheduled anchor converted to UTC. `t_first_minute_bar`
is the first qualifying SKHY minute bar at or after it. The corresponding bar
open is `skhy_anchor_price`, the implemented `P[d,t0]`. Its actual bar time is
retained, and `exact_minute_anchor` identifies exact 04:00 or 09:30 starts.

The timestamps `t_first_trade`, `t_first_quote`, and `t_first_minute_bar` remain
distinct. Unavailable trade or quote data remains null and is never replaced
by the minute-bar timestamp.

Korean daily data is joined backward by availability time. A close is eligible
only when:

```text
korea_availability_timestamp_utc <= timestamp_utc
```

The close is treated as available at 15:30 `Asia/Seoul` on its trading date.

## Return definitions

### Korean causal return

Using adjusted closing prices for corporate-action consistency:

\[
r_d^{KR}=\ln\left(\frac{C_d^{KR}}{C_{d-1}^{KR}}\right).
\]

| Field | Definition |
|---|---|
| `korea_adjusted_close` | \(C_d^{KR}\) |
| `korea_prior_adjusted_close` | \(C_{d-1}^{KR}\) |
| `kr_return_0` | Latest available \(r_d^{KR}\) |
| `kr_return_lag1` | \(r_{d-1}^{KR}\) |
| `kr_return_lag2` | \(r_{d-2}^{KR}\) |

### Post-anchor U.S. response

For every horizon:

\[
r_d^{US,h}=\ln\left(\frac{P_{d,t_0+h}^{US}}{P_{d,t_0}^{US}}\right).
\]

`P[d,t0]` is `skhy_anchor_price`. `P[d,t0+h]` is the close of the exact
one-minute bar ending `h` minutes after the actual anchor and is stored as
`skhy_target_price_{h}m`.

The six dependent-variable columns are:

```text
skhy_fwd_return_5m
skhy_fwd_return_10m
skhy_fwd_return_30m
skhy_fwd_return_60m
skhy_fwd_return_90m
skhy_fwd_return_120m
```

For each `h`, the panel also stores `skhy_target_timestamp_{h}m` and
`skhy_fwd_return_{h}m_missing`.

### Total response

For each anchor and horizon:

\[
r_d^{total,h}=\ln\left(\frac{P_{d,t_0+h}^{US}}{C_{d-1}^{US}}\right),
\]

where `C[d-1]` is `previous_us_regular_close`. The six columns are
`skhy_total_response_{h}m` for the same horizon set. Each has a corresponding
`skhy_total_response_{h}m_missing` flag.

The implementation verifies:

\[
r_d^{total,h}=\underbrace{\ln(P_{d,t_0}/C_{d-1}^{US})}_{overnight\_gap}
+\underbrace{\ln(P_{d,t_0+h}/P_{d,t_0})}_{skhy\_fwd\_return_h}.
\]

Thus total response measures the combined adjustment from the previous U.S.
regular close through the selected future horizon.

## ARDL feature blocks

For a chosen `(anchor_type, h)`, construct the model from named columns:

| Equation term | Feature columns |
|---|---|
| Dependent variable | `skhy_fwd_return_{h}m` or `skhy_total_response_{h}m` |
| U.S. AR block | `<dependent_variable>_lag1`, `<dependent_variable>_lag2` |
| Korean causal block | `kr_return_0`, `kr_return_lag1`, `kr_return_lag2` |
| Recent SKHY control | `skhy_pre_event_return` |
| Microstructure | `skhy_pre_event_rv_30m`, `skhy_pre_event_volume_z`, `skhy_vwap_deviation` |
| Market | `spy_pre_event_return`, `soxx_pre_event_return`, `qqq_pre_event_return` |
| Currency | `usdkrw_return_latest` |
| Calendar/quality | `iso_day_of_week`, `korea_stale_days`, `korea_new_close`, `exact_minute_anchor` |

Future SPY, SOXX, QQQ, and USD/KRW returns are stored for conditional
explanatory specifications. They must not be used as predictors in a real-time
forecast because they are not known at `t0`.

## Restricted test and control ablations

The unrestricted equation contains the Korean causal block. The restricted
equation uses the same observations, selected `(p,q)`, and controls but removes
all Korean-return terms. Test:

\[
H_0:\beta_{0,h}=\beta_{1,h}=\cdots=\beta_{q,h}=0.
\]

| Ablation | Controls added to AR base |
|---|---|
| A0 | None |
| A1 | Microstructure |
| A2 | Market |
| A3 | Currency |
| A4 | Microstructure + market |
| A5 | Microstructure + currency |
| A6 | Market + currency |
| A7 | All available controls |

Use a common complete-case sample for every restricted/unrestricted pair and
for ablations being compared.

## Missing-target policy

A target is valid only when the exact required future minute bar exists. The
builder does not substitute a later bar. Missing values are represented by:

- a null return and target price;
- the corresponding `_missing = true` flag;
- `missing_target_count` or `missing_total_response_count`;
- `all_targets_complete` or `all_total_responses_complete`.

Filter separately by anchor and horizon before estimation. Create lags within
each `anchor_type` only, so 04:00 and 09:30 responses are never mixed.
