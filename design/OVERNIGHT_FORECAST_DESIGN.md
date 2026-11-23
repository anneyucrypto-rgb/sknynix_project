# Daily H1/H2 overnight-return design

## Target and primary predictor

For U.S. session date \(d\), both hypotheses use the SKHY close-to-open log
return:

\[
r^{ON}_d=\log(P^{SKHY}_{d,09:30\ open}/P^{SKHY}_{d-1,close}).
\]

The primary predictor is the daily log return of the Korean ordinary share
converted to U.S. dollars at the Korean close:

\[
r^{KR,FX}_d=\Delta\log(0.1P^{000660,KRW}_d/USDKRW_d).
\]

The Korean session is joined to the next eligible U.S. open by its 15:30 KST
availability. The 0.1 ADS ratio is a constant and therefore does not alter the
return, but it keeps the converted price economically interpretable.

## H1

H1 estimates the pre-open long-run level relation with the daily ARDL--UECM
defined in [`h1.md`](h1.md). The long-run slope is
`-rho_k/rho_a`, and the equilibrium error is oriented as SKHY minus its fitted
equilibrium. Positive ECT therefore means SKHY is above equilibrium, so
convergence in H2 requires a negative adjustment coefficient.

## H2 and control ablation

H2 estimates:

\[
r^{ON}_d=\alpha+\lambda ECT_d+\beta r^{KR,FX}_d+
\sum_{i=1}^{p}\phi_i r^{ON}_{d-i}+
\sum_{j=1}^{q}\delta_j r^{KR,FX}_{d-j}+\gamma'Controls_d+\epsilon_d.
\]

It tries these daily control groups:

- completed SOXX close-to-close return;
- prior completed daily normalized SKHY news sentiment;
- Korean semiconductor ETF, KOSDAQ, and KOSPI daily returns;
- daily Korean market-proxy returns; a within-group redundancy check compares
  EWY, FLKR, and USD/KRW, while the operational group retains FLKR and USD/KRW;
- released U.S. CPI YoY and core-CPI change;
- compact daily U.S. funding-condition spreads and changes;
- completed VIX daily return; and
- daily 2-year and 10-year Treasury yield changes.

The ablation study estimates each source group alone, the all-control model,
and every leave-one-group-out model. Selection is nested at every outer
forecast origin. Using prior rows only, H1 is re-estimated, the ECT is rebuilt,
and BIC selects p and q from 0 through 3 within each control specification. A
specification is eligible when \(-2<\lambda<0\), HAC \(p_\lambda<0.05\),
\(\beta_{KR,FX}>0\), and HAC \(p_\beta<0.05\). Eligible models are ranked by
inner expanding-window RMSE. The simplest model whose mean squared loss is
within one standard error of the best model is selected; MAE, direction
accuracy, coefficient stability, and BIC are ordered tie-breakers.

Grouping prevents individual fields from a multi-column source from being
cherry-picked. The complete candidate table is written to
`control_ablation.parquet`.

## Daily availability rules

The model matrix has one row per U.S. session date. Intraday timestamps are
used only while constructing the overnight target and enforcing availability;
they are removed from the feature table.

- Korean close data are eligible after 15:30 KST.
- U.S. market and economic daily observations are eligible the following day.
- A completed U.S. calendar day's sentiment is eligible the following midnight.
- Inflation source rows do not contain release timestamps, so the 15th of the
  following month is the explicit conservative availability convention.
- Missing supplied CPI YoY values are derived from the CPI index.

No 07:00 or 09:25 anchor, premarket gauge, or anchor-relative return is used.

## Current materialized run

Run `overnight-daily-ablation` uses a common 86-observation estimation sample
and 40 initial training observations for forecast diagnostics. The lambda
selection chooses the `vix` specification at all 46 outer origins. At the last
origin it uses \(p=0\), \(q=0\), \(\lambda=-0.326532\), and HAC
\(p_\lambda=3.998\times10^{-13}\). The nested forecasts have RMSE 0.020632,
MAE 0.016150, directional accuracy 89.13%, and zero-benchmark out-of-sample
\(R^2\) of 0.727496. These figures describe the current short
sample and must not be treated as a stable production estimate.

## Chronos-2 benchmark

Chronos-2 forecasts the same daily close-to-open target using four daily
covariates: the FX-adjusted Korean ordinary-share return, the VIX return, the
FLKR return, and the USD/KRW return. Its expanding-window run begins with 40
training observations. At every forecast origin, the LoRA adapter is fit using
prior observations only; the current-date covariates are supplied as known
future covariates for the one-step forecast.

### True nested expanding-window fine-tuning

Let (t) denote an outer forecast origin, (y_t=r^{ON}_t), and let
(x_t) contain the four covariates. The training information set is

\[
\mathcal H_t=\{(y_s,x_s):s<t\}.
\]

The histories are nested:

\[
\mathcal H_{t_1}\subset\mathcal H_{t_2}\subset\cdots,
\]

but (y_t) is never included before the forecast for (t) is fixed. The
implemented algorithm is:

1. Sort the panel by U.S. session date, require all four covariates, and admit
   an origin only after at least 40 earlier nonmissing targets.
2. Load the same pretrained `amazon/chronos-2` base checkpoint. Its base
   weights remain frozen; only LoRA adapter weights are optimized. The base is
   not replaced by a model trained on the full evaluation sample.
3. At the first origin, initialize a LoRA adapter over the base model. At every
   later origin, Chronos creates a fresh PEFT wrapper over that same base and a
   training callback restores the adapter state produced at the preceding
   origin. Thus the optimization is warm-started causally; no adapter trained
   at a later date is used for an earlier forecast.
4. Fine-tune the adapter on the complete prior-only history
   (mathcal H_t). The configured fit uses a one-step prediction length, 50
   optimizer steps, learning rate (10^{-5}), batch size 16, minimum past
   context 8, and seed 42. The historical target and covariate sequences form
   the training windows. Because the complete expanding history is presented
   again at every origin, earlier observations may contribute to multiple
   optimizer updates; this is intentional warm-started expanding-window
   training.
5. Copy the fitted LoRA weights to CPU memory as the only state carried into
   the next origin. Optimizer state is not carried forward.
6. Forecast (y_t) from (mathcal H_t) and the already available (x_t),
   supplied as one-step future covariates. Chronos returns the 10th, 50th, and
   90th percentiles; the median is the point forecast.
7. Store the forecast before exposing (y_t). The realized target is used only
   for scoring and becomes eligible for training at origin (t+1).
8. After the final historical forecast is fixed, perform one final adapter fit
   including that last realized target and save the adapter plus its
   `train_through` date. This deployment checkpoint does not alter any reported
   historical forecast.

In pseudocode:

```text
adapter = None
for t in eligible_forecast_dates:
    history = rows strictly before t with observed target
    model = fresh LoRA wrapper(pretrained_base)
    restore model adapter from adapter, if one exists
    model.fit(history, fixed_hyperparameters)
    adapter = copy(model.adapter_weights)
    forecast[t] = model.predict(history, known_covariates[t])
    reveal target[t] only after forecast[t] is stored
final_adapter = fit(history including the last revealed target)
```

Here “nested” refers to the strictly nested outer training histories and the
origin-by-origin refit. It does **not** mean that Chronos hyperparameters are
selected by an additional inner cross-validation loop: the LoRA mode, step
count, learning rate, batch size, minimum context, and quantiles are fixed
before the outer evaluation. Consequently, the run is a leakage-safe nested
walk-forward fine-tuning exercise, not nested hyperparameter optimization.

For a fair benchmark, `backtest.forecast_comparison` compares Chronos-2 with a
fixed VIX-only H1/H2 specification with p=q=0. H1 and the ECT are still
re-estimated at every H1/H2 forecast origin. Metrics are calculated only over
the intersection of forecast dates, so neither model benefits from a longer
evaluation sample.

The current 49-date common sample runs from 2026-09-09 through 2026-11-16.
H1/H2 has RMSE 0.020878, MAE 0.017021, directional accuracy 83.67%, and
zero-benchmark OOS \(R^2=0.715039\). Chronos-2 has RMSE 0.029204, MAE
0.022356, directional accuracy 73.47%, and OOS \(R^2=0.442446\). Thus H1/H2
has 28.5% lower RMSE and is the stronger forecast on this short sample.

```text
python -m src.features.h1_h2
python -m src.hypotheses.overnight_ablation --run-id overnight-daily-ablation --max-p 3 --max-q 3 --min-train 40
python -m backtest.backtest --run-id overnight-daily-ablation --min-train 40
python -m src.features.chronos_2
python -m backtest.chronos_2 --min-train 40 --steps 50
python -m backtest.forecast_comparison --min-train 40
```
