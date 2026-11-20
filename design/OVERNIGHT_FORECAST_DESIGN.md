# SKHY overnight forecasting

## Objective and information cutoff

For U.S. session date \(d\), H2 and Chronos-2 forecast the full SKHY
close-to-open gap:

\[
r^{ON}_{d}=\log\left(
\frac{P^{SKHY}_{d,09{:}30\ open}}
{P^{SKHY}_{d-1,16{:}00\ close}}
\right).
\]

The forecast is frozen at 09:25 ET. The 09:30 opening price is never available
to the model. The completed Korean session is daily. SOXX is intraday, but its
two retained controls are aggregate point-in-time gauges rather than lagged
five-minute blocks.

The bivariate-H1/SOXX-gauge equation is the retained H2 specification.
Chronos-2 uses the Korean day return and the two SOXX gauges in a separate
walk-forward forecast.

## H1: bivariate equilibrium

H1 is the daily bivariate ARDL–UECM between SKHY and the FX-adjusted Korean
ordinary share. BIC chooses the ARDL orders, subject to at least one lag of
each level variable. H1 estimates the long-run intercept \(\widehat c\) and
slope \(\widehat\theta\).

Let \(a_d=\log(P^{SKHY}_{d,close})\) and
\(k_d=\log(0.1P^{KR}_{d,close}/USDKRW_d)\), using the project's ADR conversion
factor of 0.1. The retained run selected ARDL order \((1,2)\), yielding the
unrestricted error-correction equation

\[
\Delta a_d=\alpha+\rho_a a_{d-1}+\rho_k k_{d-1}
+\gamma_0\Delta k_d+\gamma_1\Delta k_{d-1}+\varepsilon_d.
\]

| H1 measure | Result |
| --- | ---: |
| Long-run intercept \(\widehat c\) | 1.527827 |
| Long-run slope \(\widehat\theta\) | 0.745721 |
| Adjustment coefficient \(\rho_a\) | −0.440395 |
| Adjustment p-value | \(1.45\times10^{-4}\) |
| Bounds-test F statistic | 9.2129 |
| Bounds upper-tail p-value | \(3.71\times10^{-4}\) |

For H2 date \(d\), the equilibrium error is fixed before the current U.S.
opening:

\[
ECT_d=\log(P^{SKHY}_{d-1,close})
-\widehat c
-\widehat\theta\log(P^{KR,USD}_d).
\]

Here \(P^{SKHY}_{d-1,close}\) is the prior SKHY regular-session close and
\(P^{KR,USD}_d=0.1P^{KR}_{d,close}/USDKRW_{d,09{:}25}\). H1 supplies this
equilibrium benchmark. It does not directly generate the return forecast.

## H2: retained SOXX-gauge equation

H2 produces the overnight-return forecast:

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

where:

- \(r^{KR,day}_d=\log(P^{KR}_{d,close}/P^{KR}_{d,open})\) is known after the
  Korean market closes;
- \(r^{SOXX,close\rightarrow07{:}00}_d\) measures SOXX from its previous
  regular close to the last price available at 07:00 ET; and
- \(r^{SOXX,07{:}00\rightarrow09{:}25}_d\) measures SOXX from 07:00 to the
  last price available at the 09:25 cutoff.

BIC selected \(p=0\) and \(q=0\). Therefore, no lagged overnight-return terms
and no lagged SOXX-gauge terms enter the final equation.

## Retained H2 statistical results

The H1 and H2 estimates and validation metrics shown here come from
`overnight-best`, reported on 19 November 2026. Run-specific model artifacts
are stored as `h1_model.parquet` and `h2_model.parquet` under
`data/features/experiments/run_id=overnight-best/`.

| Measure | Result |
| --- | ---: |
| Complete observations | 88 |
| Expanding-window forecasts | 48 |
| RMSE | 0.013020 |
| MAE | 0.010313 |
| Directional accuracy | 97.9% |
| OOS \(R^2\) versus zero | 0.8912 |
| H2 adjustment coefficient \(\lambda\) | −0.226168 |
| H2 adjustment p-value | \(8.63\times10^{-7}\) |
| Selected \(p,q\) | 0, 0 |

The negative, highly significant H2 adjustment coefficient is consistent with
overnight correction toward the bivariate H1 equilibrium. Return forecasting
uses H2: H1 estimates the benchmark, while H2 combines it with current Korean
and SOXX information.

## Expanding-window validation

The displayed coefficients describe the full-sample fit. Forecast validation
selects the H2 lag order by BIC in the initial 40-observation window. The
selected forecast order is \(p=0,\ q=0\). At each forecast date, H1 is refitted
on earlier daily observations, and its coefficients are used to calculate ECT
for the training rows and the current forecast row. H2 is refitted on the
expanding history and predicts one session ahead at 09:25 ET. The realized
return enters training for the next forecast. The 48 forecasts are scored by
RMSE, MAE, directional accuracy, and OOS \(R^2\) against a zero-return forecast.

## Chronos-2 walk-forward forecasting with LoRA

[backtest/chronos_2.py](../backtest/chronos_2.py) forecasts one U.S. session
ahead with the pretrained Amazon Chronos-2 checkpoint. Its target is the same
SKHY log return \(r^{ON}_d\), stored as `overnight_return`. Its inputs are
historical target returns and the three covariates below. The feature builder
computes them from the original SKHY and SOXX one-minute Parquet files and the
Korean daily Parquet files, and writes a standalone panel under
`data/features/overnight/`.

| Covariate | Information available at the forecast cutoff |
| --- | --- |
| Korean day return | Completed Korean open-to-close return |
| SOXX overnight gauge | Prior regular close to the last price available at 07:00 ET |
| SOXX premarket gauge | 07:00 ET to the last price available at 09:25 ET |

1. Sort by U.S. session date and begin forecasting after 40 complete prior
   observations by default. A training row needs the realized overnight return
   and all predictor inputs. A forecast row needs the predictor inputs.
2. Start from the pretrained checkpoint and train a LoRA adapter on the initial
   history. Continue training the same adapter weights at each later origin
   using the expanding history and a new optimizer. Each default fit uses 50
   optimization steps, a learning
   rate of \(10^{-5}\), and a minimum training context of eight observations.
   Each historical training window supplies past returns and covariates as
   context and the covariates known at the forecasted step as conditioning
   inputs. The realized return at that step is the training label.
3. At inference, supply the historical context and the three covariate values
   for \(d\). Generate the 10th, 50th, and 90th percentiles of the one-step
   return forecast. Use the median as the point forecast. The current return
   and opening price are excluded from the model inputs.
4. In the historical simulation, a positive median signals a long entry and a
   negative median signals a short entry at the 09:30 ET open. A zero median
   produces no entry. Record the simulated entry price when the open is
   available. The opening price determines the realized forecast target, which
   joins the training history at the next origin if the row is complete.

The Chronos-2 output contains forecast quantiles, realized overnight returns,
training cutoffs, and simulated opening entries. The close-to-open return is a
forecast target. A position entered at the open would need a subsequent exit
price to measure its trading return.

After the last forecast, the final training update includes that session's
realized return if available. The adapter and its training cutoff are saved
under `.model_cache/chronos-2-finetuned/`. Each historical evaluation starts
from the pretrained base to keep the training cutoff valid at every origin.

Run from the repository root after preparing the raw market inputs listed
in [run_commands.txt](../run_commands.txt).

```powershell
python -m src.features.chronos_2
python -m backtest.chronos_2 --min-train 40 --steps 50 --output data/features/experiments/run_id=chronos-2/forecasts.parquet
```

## Executable premarket-to-open strategy

The trading return is narrower than the H2 target:

\[
r^{trade}_d=\log\left(
\frac{P^{SKHY}_{d,09{:}30\ open}}
{P^{SKHY}_{d,09{:}25\ entry}}
\right).
\]

The strategy proceeds as follows:

1. Freeze the retained H2 full-gap forecast at 09:25 ET.
2. Using earlier forecast/return pairs only, calibrate the H2 full-gap forecast
   to an expected remaining 09:25–09:30 return and residual volatility.
3. Enter at the 09:25 SKHY premarket open only if expected remaining profit
   exceeds transaction costs, two-sided slippage, square-root impact, and the
   safety buffer.
4. Scale the position by expected return divided by predicted variance, capped
   at 100% gross exposure.
5. Apply volatility-scaled stop-loss and take-profit levels through 09:27.
6. Submit an opposite-side MOO before 09:28 for every surviving position and
   exit in the 09:30 opening auction.
7. Enforce portfolio drawdown pauses, a hard drawdown stop, and a minimum-fund
   floor.

Only the executable 09:25–09:30 return is credited. The strategy never claims
the portion of the overnight gap already realized by 09:25. Transaction costs,
slippage, market impact, and risk controls are included.

The [trading results](../reports/TRADING_RESULTS.md) report the execution metrics for
the H2 premarket-to-open strategy.

Nasdaq's MOO timing is documented in the
[Nasdaq Opening/Closing Cross FAQ](https://beta.nasdaqtrader.com/content/ProductsServices/Trading/Crosses/openclose_faqs.pdf).
The Massive one-minute 09:30 open is used as an opening-auction proxy; official
Nasdaq opening prices and order-level fills are required for production-grade
execution validation.
