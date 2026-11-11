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
│   │   ├── SKHY_5min.parquet
│   │   ├── SPY_5min.parquet
│   │   ├── SOXX_5min.parquet
│   │   ├── QQQ_5min.parquet
│   │   ├── SKHY_first_trades.parquet
│   │   ├── SKHY_first_quotes.parquet
│   │   ├── SKHY_trades.parquet        # optional full tick history
│   │   └── SKHY_quotes.parquet        # optional full NBBO history
│   ├── korea/000660_daily.parquet
│   └── fx/USDKRW_5min.parquet
└── processed/
    ├── granger_panel.parquet
    └── arbitrage_panel_5min.parquet
```

`granger_panel.parquet` is the only ARDL/Granger/VARX feature table. There are
no database catalog or metadata tables. `arbitrage_panel_5min.parquet` contains
five-minute cross-market parity and premium-residual features.

Build both panels with:

```powershell
python ingestion/database.py
```

## Korean/U.S. research-pair registry

`ingestion/download.py` supports the following asynchronous price-discovery
pairs. The U.S. ticker is the `--pairs` selector and raw-file prefix.

| Korean company | Korea ticker | U.S. ticker | U.S. exchange | Industry |
|---|---|---|---|---|
| SK hynix | `000660.KS` | `SKHY` | Nasdaq | Semiconductors |
| LG Display | `034220.KS` | `LPL` | NYSE | Displays |
| SK Telecom | `017670.KS` | `SKM` | NYSE | Telecommunications |
| KT Corporation | `030200.KS` | `KT` | NYSE | Telecommunications |
| POSCO Holdings | `005490.KS` | `PKX` | NYSE | Steel, materials |
| KB Financial | `105560.KS` | `KB` | NYSE | Banking |
| Shinhan Financial | `055550.KS` | `SHG` | NYSE | Banking |
| Woori Financial | `316140.KS` | `WF` | NYSE | Banking |
| Korea Electric Power | `015760.KS` | `KEP` | NYSE | Utilities |

The default remains `--pairs SKHY`. Examples:

```powershell
# Two selected pairs
python ingestion/download.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --pairs SKHY LPL

# All configured pairs, five-minute/daily data and first event ticks
python ingestion/download.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --pairs ALL

# All pairs plus complete trade and quote histories
python ingestion/download.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --pairs ALL --full-ticks

# Download selected target pairs only; skip SPY, SOXX, QQQ, and USD/KRW
python ingestion/download.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --pairs SKHY LPL --skip-controls
```

For U.S. ticker `T` and Korean code `K`, ingestion writes:

```text
data/raw/us/T_5min.parquet
data/raw/us/T_first_trades.parquet
data/raw/us/T_first_quotes.parquet
data/raw/us/T_trades.parquet          # only with --full-ticks
data/raw/us/T_quotes.parquet          # only with --full-ticks
data/raw/korea/K_daily.parquet
```

SPY, SOXX, QQQ, and USD/KRW are downloaded once as common controls. The
current processed feature builders remain SKHY-specific; the additional raw
pairs are ready for subsequent pair-indexed panel construction.

`--skip-controls` suppresses only the four common-control downloads. It does
not skip either side of a selected target pair: the U.S. five-minute bars and
the matched Korean daily prices are still downloaded. Building the complete
ARDL feature panel later requires existing control files or a subsequent run
without `--skip-controls`.

## Event indexing and information availability

| `anchor_type` | Scheduled New York time | Purpose |
|---|---:|---|
| `premarket_0400` | 04:00 | Earliest U.S. premarket response |
| `regular_0930` | 09:30 | Higher-liquidity robustness anchor |

`timestamp_utc` is the scheduled anchor converted to UTC. `t_first_minute_bar`
is the first qualifying SKHY five-minute bar at or after it. The corresponding bar
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
five-minute bar ending `h` minutes after the actual anchor and is stored as
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

## Complete model-variable dictionary

Let `a` denote the anchor, `h ∈ {5,10,30,60,90,120}` the return horizon,
and `w ∈ {1,5,10,30,60}` the trade/quote aggregation window. A separate model
is estimated for every `(a,h)` pair.

### Target variables

The six primary post-anchor targets are:

| Horizon | Target column | Mathematical definition |
|---:|---|---|
| 5 | `skhy_fwd_return_5m` | \(\ln(P_{d,t_0+5}/P_{d,t_0})\) |
| 10 | `skhy_fwd_return_10m` | \(\ln(P_{d,t_0+10}/P_{d,t_0})\) |
| 30 | `skhy_fwd_return_30m` | \(\ln(P_{d,t_0+30}/P_{d,t_0})\) |
| 60 | `skhy_fwd_return_60m` | \(\ln(P_{d,t_0+60}/P_{d,t_0})\) |
| 90 | `skhy_fwd_return_90m` | \(\ln(P_{d,t_0+90}/P_{d,t_0})\) |
| 120 | `skhy_fwd_return_120m` | \(\ln(P_{d,t_0+120}/P_{d,t_0})\) |

The six total-response robustness targets are:

| Horizon | Target column | Mathematical definition |
|---:|---|---|
| 5 | `skhy_total_response_5m` | \(\ln(P_{d,t_0+5}/C^{US}_{d-1})\) |
| 10 | `skhy_total_response_10m` | \(\ln(P_{d,t_0+10}/C^{US}_{d-1})\) |
| 30 | `skhy_total_response_30m` | \(\ln(P_{d,t_0+30}/C^{US}_{d-1})\) |
| 60 | `skhy_total_response_60m` | \(\ln(P_{d,t_0+60}/C^{US}_{d-1})\) |
| 90 | `skhy_total_response_90m` | \(\ln(P_{d,t_0+90}/C^{US}_{d-1})\) |
| 120 | `skhy_total_response_120m` | \(\ln(P_{d,t_0+120}/C^{US}_{d-1})\) |

For target family `y`, the estimated dependent variable is one `y_{d,a,h}` at
a time. Do not place the six nested horizons into one regression as if they
were independent observations.

Target audit fields are `skhy_anchor_price`, `previous_us_regular_close`,
`skhy_target_price_{h}m`, `skhy_target_timestamp_{h}m`, the two `_missing`
families, `missing_target_count`, `all_targets_complete`,
`missing_total_response_count`, and `all_total_responses_complete`. These
verify target construction and sample eligibility; they are not regressors.

### Autoregressive predictors

For each target and anchor, the U.S. AR block contains prior event-day values
of the same response, horizon, and anchor:

\[
AR_{d,a,h}(p)=\sum_{i=1}^{p}\phi_{i,h}y_{d-i,a,h},\qquad p\in\{0,1,2\}.
\]

The implemented columns are:

```text
skhy_fwd_return_{h}m_lag1
skhy_fwd_return_{h}m_lag2
skhy_total_response_{h}m_lag1
skhy_total_response_{h}m_lag2
```

for every `h ∈ {5,10,30,60,90,120}`, giving 24 AR predictor columns. Lags are
created within `anchor_type`; a 04:00 response is never used as the lag of a
09:30 response.

### Korean causal predictors

The unrestricted Korean block is:

\[
KR_d(q)=\sum_{j=0}^{q}\beta_{j,h}r^{KR}_{d-j},\qquad
r_d^{KR}=\ln(C_d^{KR}/C_{d-1}^{KR}),\quad q\in\{0,1,2\}.
\]

| Column | Formula or meaning | Role |
|---|---|---|
| `kr_return_0` | \(\ln(C_d^{KR}/C_{d-1}^{KR})\) | Main causal predictor |
| `kr_return_lag1` | \(r_{d-1}^{KR}\) | Distributed lag 1 |
| `kr_return_lag2` | \(r_{d-2}^{KR}\) | Distributed lag 2 |
| `korea_range` | \(\ln(H_d^{KR}/L_d^{KR})\) | Optional Korean volatility control |
| `korea_log_volume` | \(\ln(1+V_d^{KR})\) | Optional Korean liquidity control |
| `korea_stale_days` | U.S. event date minus matched Korean trading date | Information-age control |
| `korea_new_close` | 1 if matched Korean close is from the event date | Information-quality indicator |

`korea_open`, `korea_high`, `korea_low`, `korea_close`,
`korea_adjusted_close`, `korea_prior_adjusted_close`, and `korea_volume` are
audit/source fields. Do not include raw price levels in the primary return
model.

### Predetermined SKHY controls

Let `P0` be the anchor price and let `P-30` be the earliest qualifying price
in the 30-minute pre-anchor window. The implemented control is piecewise:

\[
r_d^{SKHY,pre}=\begin{cases}
\ln(P_0/P_{-30}), & a=09{:}30,\\
\ln(P_0/C^{US}_{d-1}), & a=04{:}00.
\end{cases}
\]

| Column | Formula or meaning |
|---|---|
| `skhy_pre_event_return` | Piecewise return above |
| `overnight_gap` | \(\ln(P_0/C^{US}_{d-1})\) |
| `skhy_pre_event_rv_30m` | \(\sqrt{\sum_{k=1}^{5}(\Delta\ln P_k)^2}\) across six complete five-minute bars |
| `skhy_pre_event_log_volume_30m` | \(\ln(1+\sum_{k=1}^{6}V_k)\) across six five-minute bars |
| `skhy_pre_event_volume_z` | \((x-\bar{x}_a)/s_a\), standardized within anchor type |
| `skhy_vwap_deviation` | \(\ln(P_{ref}/VWAP_{ref})\) for the last available pre-anchor reference bar |
| `pre_event_window_available` | 1 when all 30 required pre-anchor bars exist |

For a post-anchor target, these variables are known at `t0`. For a
total-response target, `overnight_gap` and the 04:00
`skhy_pre_event_return` are components of the dependent variable and must not
be used as controls.

### Predetermined market controls

For `m ∈ {SPY,SOXX,QQQ}`:

\[
r^{m,pre}_{d}=\begin{cases}
\ln(P^m_0/P^m_{-30}), & a=09{:}30,\\
\ln(P^m_0/C^m_{d-1}), & a=04{:}00.
\end{cases}
\]

| Column | Role |
|---|---|
| `spy_pre_event_return` | Broad-market return control |
| `soxx_pre_event_return` | Semiconductor-sector return control |
| `qqq_pre_event_return` | Technology-market robustness control |

SOXX is the preferred sector control. Use SPY as the broad-market control and
QQQ as a replacement robustness control; including all three requires a
multicollinearity check.

### Predetermined currency control

Let `F(t)` be the latest USD/KRW close available no later than `t`:

\[
usdkrw\_return\_latest=\ln\left(\frac{F(t_0)}{F(t_0-30m)}\right).
\]

This is the implemented FX predictor known at the anchor.

### Calendar, timing, and quality controls

| Column | Meaning | Model use |
|---|---|---|
| `iso_day_of_week` | ISO weekday, Monday = 1 | Calendar control or fixed effects |
| `anchor_type` | `premarket_0400` or `regular_0930` | Sample split; do not pool without interactions |
| `exact_minute_anchor` | First SKHY bar starts exactly at scheduled anchor | Quality control |
| `t_first_minute_bar` | First qualifying minute-bar timestamp | Timing diagnostic |
| `t_first_trade` | First actual trade timestamp | Liquidity-arrival outcome |
| `t_first_quote` | First valid quote timestamp | Liquidity-arrival outcome |
| `trade_ticks_available` | Full trade-tick input exists | Availability/filter flag |
| `quote_ticks_available` | Full quote-tick input exists | Availability/filter flag |

`event_date_ny`, `timestamp_utc`, `korea_trading_date`, and
`korea_availability_timestamp_utc` are index/alignment fields, not numerical
regressors. First trade/quote prices and sizes are descriptive post-anchor
outcomes, not predetermined controls.

### Horizon-matched contemporaneous controls

For `m ∈ {SPY,SOXX,QQQ}` and every horizon `h`:

\[
r^{m,h}_d=\ln(P^m_{d,t_0+h}/P^m_{d,t_0}).
\]

The columns are `spy_fwd_return_{h}m`, `soxx_fwd_return_{h}m`, and
`qqq_fwd_return_{h}m`. The horizon-matched FX return is:

\[
r^{FX,h}_d=\ln(F(t_0+h)/F(t_0)),
\]

stored as `usdkrw_fwd_return_{h}m`. Each family has six columns, one for every
target horizon. These variables are valid for conditional explanatory or
factor-adjusted specifications, but they are not real-time predictors because
they are observed after `t0`.

| Family | Implemented columns |
|---|---|
| SPY | `spy_fwd_return_5m`, `spy_fwd_return_10m`, `spy_fwd_return_30m`, `spy_fwd_return_60m`, `spy_fwd_return_90m`, `spy_fwd_return_120m` |
| SOXX | `soxx_fwd_return_5m`, `soxx_fwd_return_10m`, `soxx_fwd_return_30m`, `soxx_fwd_return_60m`, `soxx_fwd_return_90m`, `soxx_fwd_return_120m` |
| QQQ | `qqq_fwd_return_5m`, `qqq_fwd_return_10m`, `qqq_fwd_return_30m`, `qqq_fwd_return_60m`, `qqq_fwd_return_90m`, `qqq_fwd_return_120m` |
| USD/KRW | `usdkrw_fwd_return_5m`, `usdkrw_fwd_return_10m`, `usdkrw_fwd_return_30m`, `usdkrw_fwd_return_60m`, `usdkrw_fwd_return_90m`, `usdkrw_fwd_return_120m` |

## Post-anchor trade and quote windows

When full `SKHY_trades.parquet` and `SKHY_quotes.parquet` inputs are available,
the event table aggregates half-open windows `[t0, t0+w)` for
`w ∈ {1, 5, 10, 30, 60}` minutes. Every metric is created with suffix
`0_{w}m`:

```powershell
python ingestion/download.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --full-ticks
python ingestion/database.py
```

For trade `i`, let `p_i`, `v_i`, and `D_i ∈ {-1,+1}` denote price, size, and
direction. `D_i` uses the quote rule (`p_i` above/below the prevailing midpoint)
with a tick-rule fallback. For quote `q`, define:

\[
M_q=\frac{A_q+B_q}{2},\qquad
S_q=\frac{A_q-B_q}{M_q},\qquad
I_q=\frac{V^B_q-V^A_q}{V^B_q+V^A_q}.
\]

For the half-open window \(W=[t_0,t_0+w)\):

\[
\begin{aligned}
trade\_count_W &= |T_W|,\\
trade\_volume_W &= \sum_{i\in T_W}v_i,\\
trade\_vwap_W &= \frac{\sum_{i\in T_W}p_iv_i}{\sum_{i\in T_W}v_i},\\
signed\_volume_W &= \sum_{i\in T_W}D_iv_i,\\
quote\_count_W &= |Q_W|,\\
relative\_spread\_mean_W &= \frac{1}{|Q_W|}\sum_{q\in Q_W}S_q,\\
quote\_imbalance\_mean_W &= \frac{1}{|Q_W|}\sum_{q\in Q_W}I_q.
\end{aligned}
\]

With `M_i` the prevailing midpoint at trade `i` and `M_end` the last valid
midpoint at or before the window endpoint:

\[
effective\_spread\_bps_W=
\frac{\sum_i v_i\,2|p_i-M_i|M_i^{-1}\,10{,}000}{\sum_i v_i},
\]

\[
price\_impact\_bps_W=
\frac{\sum_i v_i\,2D_i(M_{end}-M_i)M_i^{-1}\,10{,}000}{\sum_i v_i}.
\]

| Column family | Definition |
|---|---|
| `trade_count_0_{w}m` | Number of valid trades |
| `trade_volume_0_{w}m` | Sum of trade sizes |
| `trade_vwap_0_{w}m` | Share-weighted trade price |
| `signed_volume_0_{w}m` | Sum of signed trade size using the quote rule with tick-rule fallback |
| `quote_count_0_{w}m` | Number of valid non-crossed quote updates |
| `relative_spread_mean_0_{w}m` | Mean `(ask-bid)/midpoint` across quote updates |
| `quote_imbalance_mean_0_{w}m` | Mean `(bid_size-ask_size)/(bid_size+ask_size)` |
| `effective_spread_bps_0_{w}m` | Share-weighted `2*abs(trade-midpoint)/midpoint*10,000` |
| `price_impact_bps_0_{w}m` | Share-weighted signed midpoint change from each trade to the last quote at the window endpoint |

`trade_ticks_available` and `quote_ticks_available` distinguish unavailable
tick files from genuine zero-activity windows. If a full-tick source is absent
or empty, its metrics are null rather than zero. The first-event files cannot
be used to calculate these aggregates.

These are post-anchor outcomes or mediators. They must not enter `X_d` when
predicting a return over an overlapping horizon, because doing so would use
future information.

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

The named control groups are:

```text
Own-history base:
    same-response lag1/lag2 selected by p
    skhy_pre_event_return                 # post-anchor response only

Microstructure:
    skhy_pre_event_rv_30m
    skhy_pre_event_volume_z
    skhy_vwap_deviation

Market:
    spy_pre_event_return
    soxx_pre_event_return
    qqq_pre_event_return                  # replacement robustness control

Currency:
    usdkrw_return_latest

Calendar/quality robustness:
    iso_day_of_week
    korea_stale_days
    korea_new_close
    exact_minute_anchor
```

The 45 `0_{w}m` trade/quote variables are not members of the predictive
microstructure control group: they are measured after `t0`. Analyze them as
liquidity outcomes, price-discovery mediators, or predictors of a strictly
later non-overlapping target.

For `skhy_total_response_{h}m`, exclude `overnight_gap` and
`skhy_pre_event_return`, because they overlap the dependent-variable interval.
Anchor-time controls that occur after the Korean close may also mediate the
Korean effect; report total-effect models without them and controlled/direct
effect specifications separately.

Use a common complete-case sample for every restricted/unrestricted pair and
for ablations being compared.

## Missing-target policy

A target is valid only when the exact required future five-minute bar exists. The
builder does not substitute a later bar. Missing values are represented by:

- a null return and target price;
- the corresponding `_missing = true` flag;
- `missing_target_count` or `missing_total_response_count`;
- `all_targets_complete` or `all_total_responses_complete`.

Filter separately by anchor and horizon before estimation. Create lags within
each `anchor_type` only, so 04:00 and 09:30 responses are never mixed.

## Physical column dictionaries

The following tables account for every physical column currently written to
the two processed Parquet files. Dtypes are the pandas/Parquet read dtypes in
the current build.

### `granger_panel.parquet` — 181 columns

| Column | Dtype | Definition and econometric role |
|---|---|---|
| `event_date_ny` | `object` | U.S. event trading date in `America/New_York`; event index, not a regressor. |
| `anchor_type` | `object` | Event-anchor label: `premarket_0400` or `regular_0930`; split or interaction variable. |
| `timestamp_utc` | `datetime64[ns, UTC]` | Scheduled 04:00 or 09:30 New York anchor converted to UTC. |
| `t_first_minute_bar` | `datetime64[ns, UTC]` | Start time of the first qualifying SKHY five-minute bar at or after the scheduled anchor; legacy-compatible field name. |
| `t_first_trade` | `datetime64[ns, UTC]` | SIP time of the first actual SKHY trade at or after the anchor; null without trade entitlement. |
| `t_first_quote` | `datetime64[ns, UTC]` | SIP time of the first valid two-sided, non-crossed SKHY quote at or after the anchor; null without quote entitlement. |
| `first_trade_price` | `float64` | Execution price of `t_first_trade`; descriptive post-anchor outcome. |
| `first_trade_size` | `int64` | Share size of `t_first_trade`; descriptive post-anchor outcome. |
| `first_bid_price` | `float64` | Bid price in `t_first_quote`; descriptive post-anchor outcome. |
| `first_ask_price` | `float64` | Ask price in `t_first_quote`; descriptive post-anchor outcome. |
| `first_bid_size` | `int64` | Displayed bid size in `t_first_quote`; descriptive post-anchor outcome. |
| `first_ask_size` | `int64` | Displayed ask size in `t_first_quote`; descriptive post-anchor outcome. |
| `skhy_anchor_price` | `float64` | Implemented `P[d,t0]`: open of the first qualifying SKHY five-minute bar. |
| `skhy_anchor_open` | `float64` | Open of the first qualifying SKHY anchor bar; equal to `skhy_anchor_price`. |
| `skhy_anchor_close` | `float64` | Close of the first qualifying SKHY five-minute anchor bar. |
| `skhy_anchor_volume` | `float64` | Volume of the first qualifying SKHY five-minute anchor bar. |
| `previous_us_regular_close` | `float64` | Last SKHY close in the previous U.S. regular session, `C[d-1,US]`. |
| `exact_minute_anchor` | `bool` | True when `t_first_minute_bar` equals the scheduled anchor exactly. |
| `iso_day_of_week` | `int64` | ISO weekday of `event_date_ny`, Monday = 1; optional calendar control. |
| `korea_trading_date` | `object` | Trading date of the backward-as-of matched Korean daily observation. |
| `korea_availability_timestamp_utc` | `datetime64[ns, UTC]` | UTC time when the matched Korean close became observable. |
| `korea_open` | `float64` | Matched Korean daily open in KRW; source/audit field. |
| `korea_high` | `float64` | Matched Korean daily high in KRW; source/audit field. |
| `korea_low` | `float64` | Matched Korean daily low in KRW; source/audit field. |
| `korea_close` | `float64` | Matched Korean unadjusted close in KRW; source/audit field. |
| `korea_adjusted_close` | `float64` | Matched corporate-action-adjusted Korean close, `C[d,KR]`. |
| `korea_prior_adjusted_close` | `float64` | Previous Korean trading-day adjusted close, `C[d-1,KR]`. |
| `korea_volume` | `int64` | Matched Korean daily share volume. |
| `kr_return_0` | `float64` | Latest Korean adjusted-close log return: `ln(C[d,KR]/C[d-1,KR])`; main causal predictor. |
| `kr_return_lag1` | `float64` | Previous Korean trading-day return, `r[d-1,KR]`; distributed-lag predictor. |
| `kr_return_lag2` | `float64` | Korean return two trading days earlier, `r[d-2,KR]`; distributed-lag predictor. |
| `korea_range` | `float64` | Korean daily log range: `ln(high/low)`; optional volatility control. |
| `korea_log_volume` | `float64` | Korean log volume: `ln(1 + korea_volume)`; optional liquidity control. |
| `korea_stale_days` | `int64` | Calendar-day age of the matched Korean observation at the U.S. event. |
| `korea_new_close` | `bool` | True when the matched Korean trading date equals the U.S. event date. |
| `skhy_pre_event_return` | `float64` | Predetermined SKHY return: `ln(P0/P-30)` at 09:30 and `ln(P0/previous close)` at 04:00. |
| `skhy_pre_event_rv_30m` | `float64` | Pre-anchor realized volatility: square root of squared five-minute log returns across six complete bars. |
| `skhy_pre_event_log_volume_30m` | `float64` | Pre-anchor log volume: `ln(1 + sum(volume over 30 minutes))`. |
| `skhy_vwap_deviation` | `float64` | Log price-to-VWAP deviation of the last available pre-anchor reference bar. |
| `pre_event_window_available` | `bool` | True when all six five-minute observations required for the 30-minute pre-event window exist. |
| `overnight_gap` | `float64` | SKHY previous-close-to-anchor return: `ln(skhy_anchor_price/previous_us_regular_close)`. |
| `spy_pre_event_return` | `float64` | Predetermined SPY return over the analogous pre-anchor interval; broad-market control. |
| `soxx_pre_event_return` | `float64` | Predetermined SOXX return over the analogous pre-anchor interval; preferred semiconductor control. |
| `qqq_pre_event_return` | `float64` | Predetermined QQQ return over the analogous pre-anchor interval; technology robustness control. |
| `usdkrw_return_latest` | `float64` | Predetermined FX return: `ln(F(t0)/F(t0-30m))` using latest available USD/KRW observations. |
| `trade_ticks_available` | `bool` | True when non-empty full SKHY trade-tick input is available. |
| `quote_ticks_available` | `bool` | True when non-empty full SKHY quote-tick input is available. |
| `trade_count_0_1m` | `float64` | Number of valid trades in the half-open post-anchor window `[t0,t0+1m)`; post-event outcome. |
| `trade_volume_0_1m` | `float64` | Sum of trade sizes in `[t0,t0+1m)`; post-event outcome. |
| `trade_vwap_0_1m` | `float64` | Share-weighted trade price in `[t0,t0+1m)`; post-event outcome. |
| `quote_count_0_1m` | `float64` | Number of valid non-crossed quote updates in `[t0,t0+1m)`; post-event outcome. |
| `relative_spread_mean_0_1m` | `float64` | Arithmetic mean of `(ask-bid)/midpoint` across quotes in `[t0,t0+1m)`. |
| `quote_imbalance_mean_0_1m` | `float64` | Mean `(bid_size-ask_size)/(bid_size+ask_size)` in `[t0,t0+1m)`. |
| `signed_volume_0_1m` | `float64` | Sum of quote-rule/tick-rule signed trade size in `[t0,t0+1m)`; post-event order-flow outcome. |
| `effective_spread_bps_0_1m` | `float64` | Share-weighted effective spread, `2*abs(trade-midpoint)/midpoint*10,000`, over 1 post-anchor minutes. |
| `price_impact_bps_0_1m` | `float64` | Share-weighted signed midpoint impact from each trade to the 1-minute window endpoint, in basis points. |
| `trade_count_0_5m` | `float64` | Number of valid trades in the half-open post-anchor window `[t0,t0+5m)`; post-event outcome. |
| `trade_volume_0_5m` | `float64` | Sum of trade sizes in `[t0,t0+5m)`; post-event outcome. |
| `trade_vwap_0_5m` | `float64` | Share-weighted trade price in `[t0,t0+5m)`; post-event outcome. |
| `quote_count_0_5m` | `float64` | Number of valid non-crossed quote updates in `[t0,t0+5m)`; post-event outcome. |
| `relative_spread_mean_0_5m` | `float64` | Arithmetic mean of `(ask-bid)/midpoint` across quotes in `[t0,t0+5m)`. |
| `quote_imbalance_mean_0_5m` | `float64` | Mean `(bid_size-ask_size)/(bid_size+ask_size)` in `[t0,t0+5m)`. |
| `signed_volume_0_5m` | `float64` | Sum of quote-rule/tick-rule signed trade size in `[t0,t0+5m)`; post-event order-flow outcome. |
| `effective_spread_bps_0_5m` | `float64` | Share-weighted effective spread, `2*abs(trade-midpoint)/midpoint*10,000`, over 5 post-anchor minutes. |
| `price_impact_bps_0_5m` | `float64` | Share-weighted signed midpoint impact from each trade to the 5-minute window endpoint, in basis points. |
| `trade_count_0_10m` | `float64` | Number of valid trades in the half-open post-anchor window `[t0,t0+10m)`; post-event outcome. |
| `trade_volume_0_10m` | `float64` | Sum of trade sizes in `[t0,t0+10m)`; post-event outcome. |
| `trade_vwap_0_10m` | `float64` | Share-weighted trade price in `[t0,t0+10m)`; post-event outcome. |
| `quote_count_0_10m` | `float64` | Number of valid non-crossed quote updates in `[t0,t0+10m)`; post-event outcome. |
| `relative_spread_mean_0_10m` | `float64` | Arithmetic mean of `(ask-bid)/midpoint` across quotes in `[t0,t0+10m)`. |
| `quote_imbalance_mean_0_10m` | `float64` | Mean `(bid_size-ask_size)/(bid_size+ask_size)` in `[t0,t0+10m)`. |
| `signed_volume_0_10m` | `float64` | Sum of quote-rule/tick-rule signed trade size in `[t0,t0+10m)`; post-event order-flow outcome. |
| `effective_spread_bps_0_10m` | `float64` | Share-weighted effective spread, `2*abs(trade-midpoint)/midpoint*10,000`, over 10 post-anchor minutes. |
| `price_impact_bps_0_10m` | `float64` | Share-weighted signed midpoint impact from each trade to the 10-minute window endpoint, in basis points. |
| `trade_count_0_30m` | `float64` | Number of valid trades in the half-open post-anchor window `[t0,t0+30m)`; post-event outcome. |
| `trade_volume_0_30m` | `float64` | Sum of trade sizes in `[t0,t0+30m)`; post-event outcome. |
| `trade_vwap_0_30m` | `float64` | Share-weighted trade price in `[t0,t0+30m)`; post-event outcome. |
| `quote_count_0_30m` | `float64` | Number of valid non-crossed quote updates in `[t0,t0+30m)`; post-event outcome. |
| `relative_spread_mean_0_30m` | `float64` | Arithmetic mean of `(ask-bid)/midpoint` across quotes in `[t0,t0+30m)`. |
| `quote_imbalance_mean_0_30m` | `float64` | Mean `(bid_size-ask_size)/(bid_size+ask_size)` in `[t0,t0+30m)`. |
| `signed_volume_0_30m` | `float64` | Sum of quote-rule/tick-rule signed trade size in `[t0,t0+30m)`; post-event order-flow outcome. |
| `effective_spread_bps_0_30m` | `float64` | Share-weighted effective spread, `2*abs(trade-midpoint)/midpoint*10,000`, over 30 post-anchor minutes. |
| `price_impact_bps_0_30m` | `float64` | Share-weighted signed midpoint impact from each trade to the 30-minute window endpoint, in basis points. |
| `trade_count_0_60m` | `float64` | Number of valid trades in the half-open post-anchor window `[t0,t0+60m)`; post-event outcome. |
| `trade_volume_0_60m` | `float64` | Sum of trade sizes in `[t0,t0+60m)`; post-event outcome. |
| `trade_vwap_0_60m` | `float64` | Share-weighted trade price in `[t0,t0+60m)`; post-event outcome. |
| `quote_count_0_60m` | `float64` | Number of valid non-crossed quote updates in `[t0,t0+60m)`; post-event outcome. |
| `relative_spread_mean_0_60m` | `float64` | Arithmetic mean of `(ask-bid)/midpoint` across quotes in `[t0,t0+60m)`. |
| `quote_imbalance_mean_0_60m` | `float64` | Mean `(bid_size-ask_size)/(bid_size+ask_size)` in `[t0,t0+60m)`. |
| `signed_volume_0_60m` | `float64` | Sum of quote-rule/tick-rule signed trade size in `[t0,t0+60m)`; post-event order-flow outcome. |
| `effective_spread_bps_0_60m` | `float64` | Share-weighted effective spread, `2*abs(trade-midpoint)/midpoint*10,000`, over 60 post-anchor minutes. |
| `price_impact_bps_0_60m` | `float64` | Share-weighted signed midpoint impact from each trade to the 60-minute window endpoint, in basis points. |
| `skhy_fwd_return_5m` | `float64` | Primary 5-minute SKHY target: `ln(P[t0+5]/P[t0])`. |
| `skhy_target_price_5m` | `float64` | SKHY price `P[t0+5]` used as the numerator of the 5-minute targets; audit field. |
| `skhy_target_timestamp_5m` | `datetime64[ns, UTC]` | Exact endpoint timestamp for the 5-minute SKHY target; audit field. |
| `skhy_fwd_return_5m_missing` | `bool` | True when the exact 5-minute SKHY post-anchor target cannot be constructed. |
| `skhy_total_response_5m` | `float64` | 5-minute total-response target: `ln(P[t0+5]/previous_us_regular_close)`. |
| `skhy_total_response_5m_missing` | `bool` | True when the 5-minute SKHY total-response target cannot be constructed. |
| `spy_fwd_return_5m` | `float64` | spy-minute forward log return for SPY: `ln(P[t0+5]/P[t0])`; contemporaneous factor control, not known at t0. |
| `soxx_fwd_return_5m` | `float64` | soxx-minute forward log return for SOXX: `ln(P[t0+5]/P[t0])`; contemporaneous factor control, not known at t0. |
| `qqq_fwd_return_5m` | `float64` | qqq-minute forward log return for QQQ: `ln(P[t0+5]/P[t0])`; contemporaneous factor control, not known at t0. |
| `usdkrw_fwd_return_5m` | `float64` | 5-minute USD/KRW forward log return; contemporaneous FX control, not known at t0. |
| `skhy_fwd_return_10m` | `float64` | Primary 10-minute SKHY target: `ln(P[t0+10]/P[t0])`. |
| `skhy_target_price_10m` | `float64` | SKHY price `P[t0+10]` used as the numerator of the 10-minute targets; audit field. |
| `skhy_target_timestamp_10m` | `datetime64[ns, UTC]` | Exact endpoint timestamp for the 10-minute SKHY target; audit field. |
| `skhy_fwd_return_10m_missing` | `bool` | True when the exact 10-minute SKHY post-anchor target cannot be constructed. |
| `skhy_total_response_10m` | `float64` | 10-minute total-response target: `ln(P[t0+10]/previous_us_regular_close)`. |
| `skhy_total_response_10m_missing` | `bool` | True when the 10-minute SKHY total-response target cannot be constructed. |
| `spy_fwd_return_10m` | `float64` | spy-minute forward log return for SPY: `ln(P[t0+10]/P[t0])`; contemporaneous factor control, not known at t0. |
| `soxx_fwd_return_10m` | `float64` | soxx-minute forward log return for SOXX: `ln(P[t0+10]/P[t0])`; contemporaneous factor control, not known at t0. |
| `qqq_fwd_return_10m` | `float64` | qqq-minute forward log return for QQQ: `ln(P[t0+10]/P[t0])`; contemporaneous factor control, not known at t0. |
| `usdkrw_fwd_return_10m` | `float64` | 10-minute USD/KRW forward log return; contemporaneous FX control, not known at t0. |
| `skhy_fwd_return_30m` | `float64` | Primary 30-minute SKHY target: `ln(P[t0+30]/P[t0])`. |
| `skhy_target_price_30m` | `float64` | SKHY price `P[t0+30]` used as the numerator of the 30-minute targets; audit field. |
| `skhy_target_timestamp_30m` | `datetime64[ns, UTC]` | Exact endpoint timestamp for the 30-minute SKHY target; audit field. |
| `skhy_fwd_return_30m_missing` | `bool` | True when the exact 30-minute SKHY post-anchor target cannot be constructed. |
| `skhy_total_response_30m` | `float64` | 30-minute total-response target: `ln(P[t0+30]/previous_us_regular_close)`. |
| `skhy_total_response_30m_missing` | `bool` | True when the 30-minute SKHY total-response target cannot be constructed. |
| `spy_fwd_return_30m` | `float64` | spy-minute forward log return for SPY: `ln(P[t0+30]/P[t0])`; contemporaneous factor control, not known at t0. |
| `soxx_fwd_return_30m` | `float64` | soxx-minute forward log return for SOXX: `ln(P[t0+30]/P[t0])`; contemporaneous factor control, not known at t0. |
| `qqq_fwd_return_30m` | `float64` | qqq-minute forward log return for QQQ: `ln(P[t0+30]/P[t0])`; contemporaneous factor control, not known at t0. |
| `usdkrw_fwd_return_30m` | `float64` | 30-minute USD/KRW forward log return; contemporaneous FX control, not known at t0. |
| `skhy_fwd_return_60m` | `float64` | Primary 60-minute SKHY target: `ln(P[t0+60]/P[t0])`. |
| `skhy_target_price_60m` | `float64` | SKHY price `P[t0+60]` used as the numerator of the 60-minute targets; audit field. |
| `skhy_target_timestamp_60m` | `datetime64[ns, UTC]` | Exact endpoint timestamp for the 60-minute SKHY target; audit field. |
| `skhy_fwd_return_60m_missing` | `bool` | True when the exact 60-minute SKHY post-anchor target cannot be constructed. |
| `skhy_total_response_60m` | `float64` | 60-minute total-response target: `ln(P[t0+60]/previous_us_regular_close)`. |
| `skhy_total_response_60m_missing` | `bool` | True when the 60-minute SKHY total-response target cannot be constructed. |
| `spy_fwd_return_60m` | `float64` | spy-minute forward log return for SPY: `ln(P[t0+60]/P[t0])`; contemporaneous factor control, not known at t0. |
| `soxx_fwd_return_60m` | `float64` | soxx-minute forward log return for SOXX: `ln(P[t0+60]/P[t0])`; contemporaneous factor control, not known at t0. |
| `qqq_fwd_return_60m` | `float64` | qqq-minute forward log return for QQQ: `ln(P[t0+60]/P[t0])`; contemporaneous factor control, not known at t0. |
| `usdkrw_fwd_return_60m` | `float64` | 60-minute USD/KRW forward log return; contemporaneous FX control, not known at t0. |
| `skhy_fwd_return_90m` | `float64` | Primary 90-minute SKHY target: `ln(P[t0+90]/P[t0])`. |
| `skhy_target_price_90m` | `float64` | SKHY price `P[t0+90]` used as the numerator of the 90-minute targets; audit field. |
| `skhy_target_timestamp_90m` | `datetime64[ns, UTC]` | Exact endpoint timestamp for the 90-minute SKHY target; audit field. |
| `skhy_fwd_return_90m_missing` | `bool` | True when the exact 90-minute SKHY post-anchor target cannot be constructed. |
| `skhy_total_response_90m` | `float64` | 90-minute total-response target: `ln(P[t0+90]/previous_us_regular_close)`. |
| `skhy_total_response_90m_missing` | `bool` | True when the 90-minute SKHY total-response target cannot be constructed. |
| `spy_fwd_return_90m` | `float64` | spy-minute forward log return for SPY: `ln(P[t0+90]/P[t0])`; contemporaneous factor control, not known at t0. |
| `soxx_fwd_return_90m` | `float64` | soxx-minute forward log return for SOXX: `ln(P[t0+90]/P[t0])`; contemporaneous factor control, not known at t0. |
| `qqq_fwd_return_90m` | `float64` | qqq-minute forward log return for QQQ: `ln(P[t0+90]/P[t0])`; contemporaneous factor control, not known at t0. |
| `usdkrw_fwd_return_90m` | `float64` | 90-minute USD/KRW forward log return; contemporaneous FX control, not known at t0. |
| `skhy_fwd_return_120m` | `float64` | Primary 120-minute SKHY target: `ln(P[t0+120]/P[t0])`. |
| `skhy_target_price_120m` | `float64` | SKHY price `P[t0+120]` used as the numerator of the 120-minute targets; audit field. |
| `skhy_target_timestamp_120m` | `datetime64[ns, UTC]` | Exact endpoint timestamp for the 120-minute SKHY target; audit field. |
| `skhy_fwd_return_120m_missing` | `bool` | True when the exact 120-minute SKHY post-anchor target cannot be constructed. |
| `skhy_total_response_120m` | `float64` | 120-minute total-response target: `ln(P[t0+120]/previous_us_regular_close)`. |
| `skhy_total_response_120m_missing` | `bool` | True when the 120-minute SKHY total-response target cannot be constructed. |
| `spy_fwd_return_120m` | `float64` | spy-minute forward log return for SPY: `ln(P[t0+120]/P[t0])`; contemporaneous factor control, not known at t0. |
| `soxx_fwd_return_120m` | `float64` | soxx-minute forward log return for SOXX: `ln(P[t0+120]/P[t0])`; contemporaneous factor control, not known at t0. |
| `qqq_fwd_return_120m` | `float64` | qqq-minute forward log return for QQQ: `ln(P[t0+120]/P[t0])`; contemporaneous factor control, not known at t0. |
| `usdkrw_fwd_return_120m` | `float64` | 120-minute USD/KRW forward log return; contemporaneous FX control, not known at t0. |
| `missing_target_count` | `int64` | Number of missing SKHY post-anchor targets across the six horizons. |
| `all_targets_complete` | `bool` | True when all six SKHY post-anchor targets are present. |
| `missing_total_response_count` | `int64` | Number of missing SKHY total-response targets across the six horizons. |
| `all_total_responses_complete` | `bool` | True when all six SKHY total-response targets are present. |
| `skhy_pre_event_volume_z` | `float64` | Within-anchor z-score of `skhy_pre_event_log_volume_30m`. |
| `skhy_fwd_return_5m_lag1` | `float64` | Lag 1 event-day value of the same-anchor 5-minute post-anchor target; AR predictor. |
| `skhy_fwd_return_5m_lag2` | `float64` | Lag 2 event-day value of the same-anchor 5-minute post-anchor target; AR predictor. |
| `skhy_fwd_return_10m_lag1` | `float64` | Lag 1 event-day value of the same-anchor 10-minute post-anchor target; AR predictor. |
| `skhy_fwd_return_10m_lag2` | `float64` | Lag 2 event-day value of the same-anchor 10-minute post-anchor target; AR predictor. |
| `skhy_fwd_return_30m_lag1` | `float64` | Lag 1 event-day value of the same-anchor 30-minute post-anchor target; AR predictor. |
| `skhy_fwd_return_30m_lag2` | `float64` | Lag 2 event-day value of the same-anchor 30-minute post-anchor target; AR predictor. |
| `skhy_fwd_return_60m_lag1` | `float64` | Lag 1 event-day value of the same-anchor 60-minute post-anchor target; AR predictor. |
| `skhy_fwd_return_60m_lag2` | `float64` | Lag 2 event-day value of the same-anchor 60-minute post-anchor target; AR predictor. |
| `skhy_fwd_return_90m_lag1` | `float64` | Lag 1 event-day value of the same-anchor 90-minute post-anchor target; AR predictor. |
| `skhy_fwd_return_90m_lag2` | `float64` | Lag 2 event-day value of the same-anchor 90-minute post-anchor target; AR predictor. |
| `skhy_fwd_return_120m_lag1` | `float64` | Lag 1 event-day value of the same-anchor 120-minute post-anchor target; AR predictor. |
| `skhy_fwd_return_120m_lag2` | `float64` | Lag 2 event-day value of the same-anchor 120-minute post-anchor target; AR predictor. |
| `skhy_total_response_5m_lag1` | `float64` | Lag 1 event-day value of the same-anchor 5-minute total-response target; AR predictor. |
| `skhy_total_response_5m_lag2` | `float64` | Lag 2 event-day value of the same-anchor 5-minute total-response target; AR predictor. |
| `skhy_total_response_10m_lag1` | `float64` | Lag 1 event-day value of the same-anchor 10-minute total-response target; AR predictor. |
| `skhy_total_response_10m_lag2` | `float64` | Lag 2 event-day value of the same-anchor 10-minute total-response target; AR predictor. |
| `skhy_total_response_30m_lag1` | `float64` | Lag 1 event-day value of the same-anchor 30-minute total-response target; AR predictor. |
| `skhy_total_response_30m_lag2` | `float64` | Lag 2 event-day value of the same-anchor 30-minute total-response target; AR predictor. |
| `skhy_total_response_60m_lag1` | `float64` | Lag 1 event-day value of the same-anchor 60-minute total-response target; AR predictor. |
| `skhy_total_response_60m_lag2` | `float64` | Lag 2 event-day value of the same-anchor 60-minute total-response target; AR predictor. |
| `skhy_total_response_90m_lag1` | `float64` | Lag 1 event-day value of the same-anchor 90-minute total-response target; AR predictor. |
| `skhy_total_response_90m_lag2` | `float64` | Lag 2 event-day value of the same-anchor 90-minute total-response target; AR predictor. |
| `skhy_total_response_120m_lag1` | `float64` | Lag 1 event-day value of the same-anchor 120-minute total-response target; AR predictor. |
| `skhy_total_response_120m_lag2` | `float64` | Lag 2 event-day value of the same-anchor 120-minute total-response target; AR predictor. |

### `arbitrage_panel_5min.parquet` — 46 columns

| Column | Dtype | Definition |
|---|---|---|
| `ticker` | `object` | Provider ticker for the U.S. instrument, normally `SKHY`. |
| `open` | `float64` | Open of the SKHY five-minute aggregate. |
| `high` | `float64` | High of the SKHY five-minute aggregate. |
| `low` | `float64` | Low of the SKHY five-minute aggregate. |
| `close` | `float64` | Close of the SKHY five-minute aggregate; U.S. price used in premium calculations. |
| `volume` | `float64` | SKHY share volume in the five-minute aggregate. |
| `vwap` | `float64` | Provider VWAP for the SKHY five-minute aggregate. |
| `transactions` | `int64` | Number of qualifying transactions in the SKHY five-minute aggregate. |
| `timestamp_utc` | `datetime64[ns, UTC]` | Start timestamp of the SKHY five-minute bar in UTC; primary row key. |
| `timestamp_ny` | `datetime64[ns, America/New_York]` | The same minute expressed in `America/New_York`. |
| `event_date_ny` | `object` | New York trading date of the minute. |
| `event_time_ny` | `object` | New York wall-clock time of the minute. |
| `availability_timestamp_utc` | `datetime64[ns, UTC]` | UTC availability time of the backward-as-of matched Korean close. |
| `trading_date` | `object` | Trading date of the matched Korean daily observation. |
| `korea_close` | `float64` | Matched Korean unadjusted close in KRW. |
| `korea_adjusted_close` | `float64` | Matched Korean adjusted close in KRW used for parity. |
| `kr_return_0` | `float64` | Latest available Korean adjusted-close log return. |
| `kr_return_lag1` | `float64` | Previous Korean trading-day adjusted-close log return. |
| `korea_range` | `float64` | Matched Korean log high-low range: `ln(high/low)`. |
| `korea_volume` | `int64` | Matched Korean daily share volume. |
| `usdkrw_close` | `float64` | Latest USD/KRW close no more than five minutes old; KRW per USD. |
| `ads_shares_per_ads` | `float64` | Korean common shares represented by one SKHY ADS; default `0.1`. |
| `korea_parity_usd_per_ads` | `float64` | FX-converted Korean parity: `korea_adjusted_close*ads_shares_per_ads/usdkrw_close`. |
| `arbitrage_premium_log` | `float64` | Log cross-market premium: `ln(SK_HY_close/korea_parity_usd_per_ads)`. |
| `arbitrage_premium_pct` | `float64` | Percentage premium in decimal form: `exp(arbitrage_premium_log)-1`. |
| `skhy_return_5m` | `float64` | SKHY close-to-close five-minute log return. |
| `usdkrw_return_5m` | `float64` | USD/KRW close-to-close five-minute log return. |
| `premium_residual_30m` | `float64` | Log premium minus its rolling 30-minute mean; minimum observations are `max(10, window/4)`. |
| `premium_zscore_30m` | `float64` | Rolling 30-minute standardized log premium: residual divided by rolling sample standard deviation. |
| `premium_residual_60m` | `float64` | Log premium minus its rolling 60-minute mean; minimum observations are `max(10, window/4)`. |
| `premium_zscore_60m` | `float64` | Rolling 60-minute standardized log premium: residual divided by rolling sample standard deviation. |
| `premium_residual_120m` | `float64` | Log premium minus its rolling 120-minute mean; minimum observations are `max(10, window/4)`. |
| `premium_zscore_120m` | `float64` | Rolling 120-minute standardized log premium: residual divided by rolling sample standard deviation. |
| `premium_residual_390m` | `float64` | Log premium minus its rolling 390-minute mean; minimum observations are `max(10, window/4)`. |
| `premium_zscore_390m` | `float64` | Rolling 390-minute standardized log premium: residual divided by rolling sample standard deviation. |
| `minutes_from_0400` | `int32` | New York clock minutes relative to 04:00; zero at the premarket anchor. |
| `minutes_from_0930` | `int32` | New York clock minutes relative to 09:30; zero at the regular-session anchor. |
| `session` | `object` | New York session label: `premarket`, `regular`, `afterhours`, or `outside`. |
| `is_0400_event` | `bool` | True for a bar whose New York start time is exactly 04:00. |
| `is_0930_event` | `bool` | True for a bar whose New York start time is exactly 09:30. |
| `premarket_t_first_trade` | `datetime64[ns, UTC]` | Date-level first actual SKHY trade at or after the 04:00 anchor. |
| `premarket_t_first_quote` | `datetime64[ns, UTC]` | Date-level first valid SKHY quote at or after the 04:00 anchor. |
| `premarket_t_first_minute_bar` | `datetime64[ns, UTC]` | Date-level first qualifying SKHY five-minute bar at or after 04:00; legacy-compatible field name. |
| `regular_t_first_trade` | `datetime64[ns, UTC]` | Date-level first actual SKHY trade at or after the 09:30 anchor. |
| `regular_t_first_quote` | `datetime64[ns, UTC]` | Date-level first valid SKHY quote at or after the 09:30 anchor. |
| `regular_t_first_minute_bar` | `datetime64[ns, UTC]` | Date-level first qualifying SKHY five-minute bar at or after 09:30; legacy-compatible field name. |

