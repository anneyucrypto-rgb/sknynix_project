# H2 ablation study results

**Run ID:** `overnight-daily-ablation`  
**Initial outer training size:** 40 observations  
**Final evaluated origin:** 2026-11-16

## Nested selection criteria

Every outer forecast uses prior observations only. At each origin, H1 is
re-estimated, the training and forecast-date ECTs are rebuilt, and p and q
from 0 through 3 are selected by BIC within each control specification.
Eligible models must satisfy:

\[
-2<\lambda<0,\qquad p_\lambda<0.05,
\]

\[
\beta_{KR,FX}>0,\qquad p_\beta<0.05.
\]

The retained models are ranked by average inner expanding-window OOS RMSE.
The simplest model within one standard error of the lowest mean squared
error is selected. MAE, direction accuracy, coefficient stability, and
BIC are ordered tie-breakers.

Selection therefore uses four primary reported quantities:

- Average inner expanding-window OOS RMSE
- ECT λ and its p-value
- Korean FX-adjusted return β
- Korean β p-value

## Top model specifications

| Selected | One-SE set | Model | p | q | Average inner RMSE | λ(ECT) | λ p-value | Korean β | β p-value |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yes | yes | `add_vix` | 0 | 0 | 0.020938 | -0.326532 | 3.998e-13 | 0.312568 | 6.964e-12 |
|  | yes | `add_kr_market_proxies` | 0 | 0 | 0.022316 | -0.333101 | 1.135e-08 | 0.390191 | 8.199e-24 |
|  |  | `add_soxx` | 0 | 0 | 0.023717 | -0.297048 | 5.975e-08 | 0.371884 | 8.148e-14 |

## Korean market-proxy redundancy analysis

EWY and FLKR carry nearly identical Korean equity-market information:
their return correlation is 0.9977, with VIFs of 219.15 and 220.60. USD/KRW is materially less redundant, with VIF 1.09.

| Rank | Proxy controls | RMSE | BIC | Adj. R² | λ | λ p-value | Korean β | β p-value |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | FLKR + USDKRW | 0.021036 | -378.32 | 0.628 | -0.334516 | 8.315e-09 | 0.394476 | 1.295e-24 |
| 2 | EWY + USDKRW | 0.021052 | -377.42 | 0.624 | -0.335751 | 8.521e-09 | 0.390745 | 5.869e-24 |
| 3 | EWY + FLKR + USDKRW | 0.021159 | -374.62 | 0.627 | -0.335587 | 8.027e-09 | 0.395776 | 9.780e-26 |
| 4 | FLKR | 0.022155 | -377.78 | 0.611 | -0.312934 | 1.335e-08 | 0.386019 | 2.845e-21 |
| 5 | EWY | 0.022213 | -376.77 | 0.606 | -0.313932 | 1.276e-08 | 0.381780 | 1.102e-20 |
| 6 | EWY + FLKR | 0.022261 | -374.20 | 0.610 | -0.314504 | 1.350e-08 | 0.387609 | 2.447e-22 |
| 7 | USDKRW | 0.022933 | -360.23 | 0.522 | -0.417527 | 2.563e-12 | 0.246388 | 7.094e-09 |

The recommended nonduplicative set is **FLKR + USD/KRW**: RMSE 0.021036 and BIC -378.32. EWY is excluded because it duplicates FLKR; USD/KRW is retained because
it contributes a distinct currency component. The operational
`kr_market_proxies` group therefore contains FLKR and USD/KRW only.

## Final H1 and H2 equations

These equations are full-common-sample reference estimates for the
currently retained VIX specification. In the nested backtest, H1, the
ECT, p/q orders, and H2 coefficients are re-estimated at every origin
using prior observations only, so forecast-date coefficients can vary.

### H1: long-run level equilibrium

Define the FX-adjusted Korean ordinary-share price as:

\[
P^{KR,FX}_d=\frac{0.1P^{000660,KRW}_d}{USDKRW_d}.
\]

H1 estimates:

\[
\log P^{SKHY}_{d-1,close}=c+\theta\log P^{KR,FX}_d+u_d.
\]

The fitted equation is:

\[
\log P^{SKHY}_{d-1,close}=1.270661+0.801591\log P^{KR,FX}_d+u_d.
\]

H1 uses 89 observations; θ has HAC p-value 7.740e-57, R² is 0.814766, and BIC is -262.719183.

The pre-response equilibrium error is:

\[
ECT_d=\log P^{SKHY}_{d-1,close}-1.270661-0.801591\log P^{KR,FX}_d.
\]

Positive ECT means the prior SKHY close is above its fitted equilibrium;
therefore convergence requires a negative H2 adjustment coefficient.

### H2: overnight ECM with VIX control

The target and return predictors are:

\[
r^{ON}_d=\log\left(\frac{P^{SKHY}_{d,09:30\ open}}{P^{SKHY}_{d-1,close}}\right),\qquad r^{KR,FX}_d=\Delta\log P^{KR,FX}_d.
\]

For the retained p=0, q=0 VIX specification:

\[
r^{ON}_d=\alpha+\lambda ECT_d+\beta r^{KR,FX}_d+\gamma r^{VIX,known}_d+\epsilon_d.
\]

The fitted reference equation is:

\[
r^{ON}_d=0.002165-0.325514ECT_d+0.313171r^{KR,FX}_d+0.166556r^{VIX,known}_d+\epsilon_d.
\]

| H2 term | Coefficient | HAC p-value |
| --- | ---: | ---: |
| Intercept | 0.002165 | 3.920e-01 |
| ECT λ | -0.325514 | 6.881e-13 |
| Korean FX-adjusted return β | 0.313171 | 1.557e-12 |
| Known VIX daily return γ | 0.166556 | 1.424e-17 |

H2 uses 86 common-sample observations; adjusted R² is 0.621867 and BIC is -380.320151.
The VIX return is the latest completed daily VIX close-to-close return
available before the SKHY open; it is not a contemporaneous post-open value.

## Selected specification

`add_vix` is selected with p=0, q=0, average inner RMSE 0.020938, λ=-0.326532 (p=3.998e-13), and Korean β=0.312568 (p=6.964e-12).

## Nested out-of-sample performance

- Forecasts: 46
- Selection frequency: `add_vix` on 46 of 46 origins
- Selected p orders: p=0 on 23 origins, p=1 on 23 origins
- Selected q orders: q=0 on 46 origins
- RMSE: 0.020632
- MAE: 0.016150
- Direction accuracy: 89.13%
- OOS R²: 0.727496
