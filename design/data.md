# SKHY event-indexed data design

This document describes the current raw-data layout and the processed features panels produced by `ingestion/features.py`.

## Research grid

The project estimates mixed-frequency event-time ARDL/Granger models on two daily U.S. events:

| Anchor | Definition | Purpose |
|---|---|---|
| `premarket_dynamic` | Earliest eligible five-minute decision timestamp from 07:00 through 09:20 ET | Detect Korean-driven price discovery during Schwab's premarket session |
| `regular_0930` | Fixed 09:30 ET research reference | Measure residual price discovery during regular trading |

Both anchors use exactly

$$
H=\{5,10,15,30,45,60,100,120\}\text{ minutes}.
$$

The wide Granger panel contains one row per `(event_date_ny, anchor_type)` and eight response columns per row. This represents 16 conceptual anchor--horizon observations per trading day before missing-data and execution-availability filters.

## Current raw-data layout

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
│   │   ├── SKHY_trades.parquet          # optional full trade ticks
│   │   ├── SKHY_quotes.parquet          # optional full NBBO ticks
│   │   ├── VIX_daily.parquet
│   │   └── US_TREASURY_daily.parquet
│   ├── korea/
│   │   ├── 000660_daily.parquet
│   │   ├── KOSPI_daily.parquet
│   │   ├── KOSDAQ_daily.parquet
│   │   ├── KOSPI_5min.parquet
│   │   ├── KOSPI200_5min.parquet
│   │   ├── KOSDAQ150_5min.parquet       # ETF proxy, explicitly labeled
│   │   ├── KRX300_5min.parquet          # ETF proxy, explicitly labeled
│   │   ├── KOSDAQ_5min.parquet
│   │   ├── KOREA_SEMICONDUCTOR_5min.parquet
│   │   └── KOREA_SEMICONDUCTOR_TOP10_5min.parquet
│   ├── fx/
│   │   └── USDKRW_5min.parquet
│   └── backup/                           # user-managed historical snapshots
└── processed/
    ├── granger_panel.parquet
    └── arbitrage_panel_5min.parquet
```

Other configured Korean/U.S. research pairs follow the same market-based
layout:

```text
data/raw/us/{US_TICKER}_5min.parquet
data/raw/us/{US_TICKER}_first_trades.parquet
data/raw/us/{US_TICKER}_first_quotes.parquet
data/raw/us/{US_TICKER}_trades.parquet       # with --full-ticks
data/raw/us/{US_TICKER}_quotes.parquet       # with --full-ticks
data/raw/korea/{KOREAN_CODE}_daily.parquet
```

## Acquisition workflow

Download the target pair and full SKHY ticks first:

```powershell
python ingestion/download.py `
  --start-date YYYY-MM-DD `
  --end-date YYYY-MM-DD `
  --pairs SKHY `
  --full-ticks `
  --skip-controls
```

Then refresh all controls and build both processed panels:

```powershell
python ingestion/features.py `
  --controls `
  --start-date YYYY-MM-DD `
  --end-date YYYY-MM-DD
```

When dates are omitted, `features.py --controls` infers the inclusive date
range from `data/raw/us/SKHY_5min.parquet`.

### Control downloads

`--controls` downloads the following datasets before building the panels:

| Market | Frequency | Series | Destination |
|---|---|---|---|
| U.S. | 5 minute | SPY, SOXX, QQQ | `data/raw/us` |
| FX | 5 minute | USD/KRW | `data/raw/fx` |
| U.S. | Daily | VIX, U.S. 2Y and 10Y Treasury yields | `data/raw/us` |
| Korea | Daily | SK hynix, KOSPI, KOSDAQ | `data/raw/korea` |
| Korea | 5 minute | KOSPI, KOSPI 200, KOSDAQ Composite | `data/raw/korea` |
| Korea | 5 minute | KOSDAQ 150 and KRX 300 tracking ETFs | `data/raw/korea` |
| Korea | 5 minute | KODEX Semiconductor and TIGER Fn Semiconductor TOP10 signals | `data/raw/korea` |

Five-minute U.S. and FX bars use Massive. Daily Korean/VIX data and Korean
five-minute indices or proxies use Yahoo Finance. Treasury yields use FRED.

Yahoo does not currently expose direct five-minute KOSDAQ 150 or KRX 300
symbols through the configured source. The files therefore use transparent
tracking proxies:

| Requested benchmark | Download symbol | `is_proxy` | File |
|---|---|---:|---|
| KOSPI | `^KS11` | `False` | `KOSPI_5min.parquet` |
| KOSPI 200 | `^KS200` | `False` | `KOSPI200_5min.parquet` |
| KOSDAQ 150 | `229200.KS` | `True` | `KOSDAQ150_5min.parquet` |
| KRX 300 | `292150.KS` | `True` | `KRX300_5min.parquet` |

Every Korean five-minute file includes `instrument_name`, `requested_index`,
and `is_proxy` metadata. A proxy must never be described as the direct index.

Yahoo retains only a rolling intraday window. The downloader uses a maximum
58-calendar-day request to stay within the provider's strict 60-day boundary
and prints a warning when the requested research start is older. Consequently,
the Korean five-minute controls can have a shorter sample than daily controls.

## Dynamic premarket anchor

The candidate grid is the set of SKHY five-minute bar starts from 07:00 through
09:20 ET:

$$
\mathcal T_d=\{t:07{:}00\leq t<09{:}25,\ t\text{ is observable}\}.
$$

At each candidate, selection uses only information available at that time:

$$
\begin{aligned}
Bid_t &> 0,\\
Ask_t &> Bid_t,\\
SpreadBps_t &\leq S_{\max},\\
Depth_t &\geq D_{\min},\\
QuoteAge_t &\leq A_{\max},\\
Volume_{t-L:t} &> 0.
\end{aligned}
$$

The selected timestamp is

$$
t_d^*=\min\{t\in\mathcal T_d:\text{all eligibility filters pass}\}.
$$

Defaults are `S_max = 100` bps, `D_min = 100` displayed-size units,`A_max = 60` seconds, and `L = 5` minutes. They can be changed with `features.py` arguments, but must be fixed before inspecting future returns. Full SKHY quote ticks are required. If quotes are unavailable or no candidate passes, the panel retains a `premarket_dynamic` row with
`anchor_available = False` and an `anchor_unavailable_reason`; it does not
invent an anchor or entry price.

For `regular_0930`, 09:30 is the fixed research timestamp. The exact bar open
is a statistical reference price, not a guaranteed fill. First trade and quote
fields remain separate execution diagnostics.

## Information availability

Korean daily closes become available at 15:30 `Asia/Seoul`. Daily joins obey

```text
korea_availability_timestamp_utc <= timestamp_utc
```

VIX closes are timestamped at 16:00 ET and Treasury observations at 17:00 ET.
Only observations available before the anchor are eligible.

For the Korean intraday controls, the builder takes the latest five-minute
observation no later than the U.S. anchor and computes a 30-minute log return
ending at that Korean observation. Each family stores:

```text
{control}_return_latest
{control}_timestamp_utc
{control}_is_proxy
```

Implemented control prefixes are:

```text
kospi_5m
kospi200_5m
kosdaq150_5m
krx300_5m
kosdaq_5m
korea_semiconductor_5m
korea_semiconductor_top10_5m
```

## Return definitions

### Korean causal return

Using adjusted Korean closes:

$$
r_d^{KR}=\ln\left(\frac{C_d^{KR}}{C_{d-1}^{KR}}\right).
$$

The causal block is `kr_return_0`, `kr_return_lag1`, and `kr_return_lag2`.
Source and audit fields include Korean OHLCV, `korea_range`,
`korea_log_volume`, `korea_stale_days`, and `korea_new_close`.

### Post-anchor response

For anchor `a` and horizon `h`:

$$
r_{d,a}^{US,h}=\ln\left(\frac{P_{d,a+h}^{US}}{P_{d,a}^{US}}\right).
$$

The response families are generated for every `h` in `H`:

```text
skhy_fwd_return_{h}m
skhy_target_price_{h}m
skhy_target_timestamp_{h}m
skhy_fwd_return_{h}m_missing
```

The endpoint must be the exact required five-minute bar. Missing endpoints are
null and flagged; they are not forward-filled.

### Total response

The robustness target measures adjustment from the previous U.S. regular
close:

$$
r_{d,a}^{total,h}=\ln\left(\frac{P_{d,a+h}^{US}}{C_{d-1}^{US}}\right).
$$

The implementation verifies

$$
r_{d,a}^{total,h}=overnight\_gap+r_{d,a}^{US,h}.
$$

Fields are `skhy_total_response_{h}m` and
`skhy_total_response_{h}m_missing`.

## Predictor families

### Autoregressive and Korean blocks

For every response family and horizon, the panel creates lag 1 and lag 2 within
the same `anchor_type`:

```text
skhy_fwd_return_{h}m_lag1
skhy_fwd_return_{h}m_lag2
skhy_total_response_{h}m_lag1
skhy_total_response_{h}m_lag2
```

Dynamic-premarket and regular-session lags are never mixed.

### Predetermined U.S. controls

```text
skhy_pre_event_return
skhy_pre_event_rv_30m
skhy_pre_event_log_volume_30m
skhy_pre_event_volume_z
skhy_vwap_deviation
spy_pre_event_return
spy_pre_event_return_5m
soxx_pre_event_return
soxx_pre_event_return_5m
qqq_pre_event_return
qqq_pre_event_return_5m
```

The five-minute market controls are exact returns ending at the selected
anchor. Thirty-minute statistics require six complete preceding bars.

### Daily and intraday macro controls

The existing Granger macro ablation uses:

```text
spy_pre_event_return_5m
kospi_return_latest
usdkrw_return_latest
vix_level_lag1d
vix_change_lag1d
us_2y_yield_change_lag1d
us_10y_yield_change_lag1d
```

The seven new Korean five-minute return signals are available in the panel for
screening and alternative specifications. They are not automatically added as
one block because the indices and tracking ETFs are highly correlated. Run
`models/feature_analysis.py` before choosing among them.

### Horizon-matched contemporaneous controls

For each horizon, the panel also creates:

```text
spy_fwd_return_{h}m
soxx_fwd_return_{h}m
qqq_fwd_return_{h}m
usdkrw_fwd_return_{h}m
```

These are post-anchor factor controls, not real-time predictors.

### Trade and quote outcomes

When full ticks exist, the panel aggregates trade and quote outcomes for
`w in {1,5,10,30,60}` minutes after the anchor:

```text
trade_count_0_{w}m
trade_volume_0_{w}m
trade_vwap_0_{w}m
signed_volume_0_{w}m
quote_count_0_{w}m
relative_spread_mean_0_{w}m
quote_imbalance_mean_0_{w}m
effective_spread_bps_0_{w}m
price_impact_bps_0_{w}m
```

These are outcomes or mediators and must not be used as predetermined
predictors.

## Core alignment and quality fields

| Field | Meaning |
|---|---|
| `event_date_ny` | U.S. event date in New York time |
| `anchor_type` | `premarket_dynamic` or `regular_0930` |
| `timestamp_utc` | Selected dynamic timestamp or fixed 09:30 timestamp |
| `research_reference_timestamp_utc` | Statistical reference timestamp |
| `anchor_available` | Whether the required anchor and exact reference bar exist |
| `anchor_unavailable_reason` | Explicit reason an anchor is unusable |
| `t_first_minute_bar` | Exact SKHY reference-bar start |
| `t_first_trade`, `t_first_quote` | Separate execution-arrival timestamps |
| `exact_minute_anchor` | Whether the reference bar exactly matches the selected timestamp |
| `pre_event_window_available` | Whether all six pre-anchor bars exist |
| `trade_ticks_available`, `quote_ticks_available` | Full-tick source availability |

Dynamic-anchor audit fields store the selected quote timestamp, bid, ask,
sizes, spread, depth, quote age, trailing volume, candidate number, and the
thresholds used in that run.

## Processed outputs

### `granger_panel.parquet`

One wide row per trading day and anchor. It contains anchor availability,
Korean causal returns, U.S. and Korean controls, eight forward-response
families, AR lags, and trade/quote outcomes.

### `arbitrage_panel_5min.parquet`

One row per SKHY five-minute bar. It contains as-of Korean/FX alignment,
ADS-parity values, log and percentage premium, rolling premium residuals and
z-scores, session labels, and merged event timestamps.

## Validation rules

`features.py` and `ingestion/test.py` enforce the following:

- required raw files must exist;
- raw SKHY event dates must match processed event dates;
- Korean adjusted-close return identities must hold;
- total responses must equal overnight gap plus post-anchor response;
- unavailable anchors remain explicit and unfilled;
- exact future endpoints are required for target construction;
- control proxies remain labeled with `is_proxy = True`;
- processed files are written atomically through temporary `.part` files.

After any raw-data, anchor, horizon, or control-definition change, run:

```powershell
python ingestion/features.py
python ingestion/test.py
python models/feature_analysis.py
python models/granger.py
```
