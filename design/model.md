# Event-time ARDL Granger ablation design

## Objective

Test whether the latest available Korean SK hynix return predicts future U.S.
SKHY returns and identify which control group changes the contemporaneous
Korean coefficient most.

## Equation

For anchor `a` and horizon `h`:

\[
r_{d,a}^{US,h}=\alpha_h
+\sum_{i=1}^p\phi_{i,h}r_{d-i,a}^{US,h}
+\sum_{j=0}^q\beta_{j,h}r_{d-j}^{KR}
+\rho_h r_{d,a}^{SKHY,pre}
+\Gamma_h X_{d,a}+\varepsilon_{d,a,h}.
\]

The dependent variable is
`skhy_fwd_return_{h}m = ln(P[t0+h]/P[t0])` for
`h = 5, 10, 30, 60, 90, 120`. Models are estimated separately at 04:00 and
09:30 New York time.

## Predictor blocks

- U.S. AR block: same-anchor, same-horizon response lags 1 and 2.
- Korean block: `kr_return_0`, `kr_return_lag1`, `kr_return_lag2`.
- Own-history control: `skhy_pre_event_return`.
- Macro-only: pre-event SPY, KOSPI return, USD/KRW return, lagged VIX level
  and change, and lagged U.S. 2Y/10Y yield changes.
- Sector-only: pre-event SOXX return.
- Liquidity-only: `skhy_pre_event_volume_z` and `skhy_vwap_deviation`.
- Microstructure-only: `skhy_pre_event_rv_30m` plus the two liquidity fields.

At 09:30, SPY and SOXX use exact five-minute returns ending at the anchor. At
04:00, no preceding U.S. five-minute bar exists, so their previous-close-to-
anchor returns are used. Korean 3Y/10Y yields and a Korean sector index are not
included because no validated daily source is currently ingested; they are
reported as unavailable rather than replaced with proxies.

Historical pre-event spreads, depth, and order imbalance are also unavailable.
The stored trade/quote window variables begin at `t0`, so they are outcomes or
mediators and are excluded from predictive controls to prevent look-ahead bias.

The baseline contains the intercept, selected U.S. AR terms, Korean block, and
own-history control. Each ablation adds exactly one control group.

## Comparison protocol

For each anchor/horizon, all estimable specifications use the same complete-case
dates. Lag orders are selected once from the unrestricted baseline over
`p = 0..2` and `q = 0..2` by minimum BIC, then held
fixed across ablations. The reported change is

\[
\Delta\beta_0^g=\hat\beta_0^g-\hat\beta_0^{baseline},
\]

and the group with the largest `abs(Delta beta)` changes the Korean coefficient
most for that anchor/horizon. Inference uses OLS coefficients with HAC standard
errors. The Korean-block Wald test evaluates
`H0: beta_0 = ... = beta_q = 0`.

Liquidity and microstructure controls are unavailable at 04:00 because the six
required five-minute pre-event bars do not exist. Those cells are marked
unavailable, not zero-filled. Results are exploratory because the current
sample is short and tests multiple horizons.
