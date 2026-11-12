# Event-time ARDL Granger ablation design

## Objective

Test whether the latest available Korean SK hynix return predicts future U.S.
SKHY returns and identify which control group changes the contemporaneous
Korean coefficient most.

## Equation

For anchor `a` and horizon `h`:

$$
\begin{aligned}
r_{d,a}^{US,h}
&= \alpha_{a,h}
+ \sum_{i=1}^p \phi_{i,a,h} r_{d-i,a}^{US,h} \\
&\quad + \sum_{j=0}^q \beta_{j,a,h} r_{d-j}^{KR}
+ \rho_{a,h} r_{d,a}^{SKHY,pre} \\
&\quad + \boldsymbol{\gamma}_{a,h}^\mathsf{T} X_{d,a}
+ \varepsilon_{d,a,h} .
\end{aligned}
$$

The dependent variable is
`skhy_fwd_return_{h}m = ln(P[t0+h]/P[t0])` for

$$
H = \left\{5,10,15,30,45,60,100,120\right\}\ \text{minutes}.
$$

Models are estimated separately at `premarket_dynamic` and `regular_0930`,
yielding 16 conceptual anchor--horizon observations per U.S. trading day before
missing-price and execution-availability filters.

## Event anchors

| Anchor | Selection | Purpose |
|---|---|---|
| `premarket_dynamic` | Earliest eligible five-minute decision timestamp from 07:00 through 09:20 ET | Detect Korean-driven price discovery during the Schwab premarket session |
| `regular_0930` | Fixed 09:30 ET research timestamp | Measure residual price discovery during regular trading |

The dynamic candidate set is

$$
\mathcal{T}_d=\{t:07\!:\!00\leq t<09\!:\!25,\ t\text{ is observable}\}.
$$

At each candidate, the latest observable quote must have a positive bid, an ask
strictly above the bid, spread no greater than `S_max`, combined displayed depth
at least `D_min`, and quote age no greater than `A_max`. Completed-bar volume
over `[t-L,t)` must be positive. The selected anchor is

$$
t_d^*=\min\{t\in\mathcal{T}_d:\text{ all liquidity filters pass}\}.
$$

The default, predeclared thresholds are `S_max = 100` bps, `D_min = 100`
displayed-size units, `A_max = 60` seconds, and `L = 5` minutes. They are
configurable in `ingestion/features.py` and use no return or future-price
information. If full quote ticks are absent or no candidate qualifies, the
anchor is recorded as unavailable rather than replaced by an invented time or
price.

At 09:30, the five-minute bar open is only a research reference price. First
trade and quote fields remain separate execution diagnostics and are not
treated as guaranteed fills.

## Predictor blocks

- U.S. AR block: same-anchor, same-horizon response lags 1 and 2.
- Korean block: `kr_return_0`, `kr_return_lag1`, `kr_return_lag2`.
- Own-history control: `skhy_pre_event_return`.
- Macro-only: pre-event SPY, KOSPI return, USD/KRW return, lagged VIX level
  and change, and lagged U.S. 2Y/10Y yield changes.
- Korean-macro-only: KOSPI return and USD/KRW return.
- Sector-only: pre-event SOXX return.
- Liquidity-only: `skhy_pre_event_volume_z` and `skhy_vwap_deviation`.
- Microstructure-only: `skhy_pre_event_rv_30m` plus the two liquidity fields.

At both anchors, SPY and SOXX use exact five-minute returns ending at the
anchor. Korean 3Y/10Y yields and a Korean sector index are not
included because no validated daily source is currently ingested; they are
reported as unavailable rather than replaced with proxies.

Dynamic-anchor spread, depth, quote age, and trailing volume are eligibility
variables only. Stored post-anchor trade/quote windows are outcomes or mediators
and are excluded from predictive controls to prevent look-ahead bias.

The baseline contains the intercept, selected U.S. AR terms, Korean block, and
own-history control. Each ablation adds exactly one control group.

## Comparison protocol

For each anchor/horizon, all estimable specifications use the same complete-case
dates. Lag orders are selected once from the unrestricted baseline over
`p = 0..2` and `q = 0..2` by minimum BIC, then held
fixed across ablations. The reported change is

$$
\Delta \beta_0^g
= \widehat{\beta}_0^g
- \widehat{\beta}_0^{\mathrm{baseline}} .
$$

and the group with the largest $|\Delta\beta_0^g|$ changes the Korean
coefficient most for that anchor/horizon. Inference uses OLS coefficients with
HAC standard errors. The Korean-block Wald test evaluates

$$
H_0:\quad \beta_0 = \beta_1 = \cdots = \beta_q = 0 .
$$

Unavailable dynamic anchors and incomplete pre-event windows are marked
unavailable, not zero-filled. Results are exploratory because the current
sample is short and tests multiple horizons.
