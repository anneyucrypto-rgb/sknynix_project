# H1 — Long-run cointegration

**Run ID:** `baseline-default`  
**Generated:** 2026-11-17T06:39:51.356733+00:00  
**Sample:** 2026-07-13 through 2026-11-16  
**Complete observations:** 90

## Result

The finite-sample ARDL bounds test rejects the null of no level relationship at the 5% level. H1 is therefore supported for this short baseline sample.

> This is statistical evidence within a 90-session sample, not proof of a stable structural equilibrium. SKHY history begins in July 2026, so long-run interpretation remains provisional.

## Specification

- Dependent variable: `log_adr`
- Long-run regressor: `log_kr_usd`
- Fixed I(0) controls: `us_soxx_return`, `kr_sector_return`
- Lag search: 0–5 by BIC; selected ARDL(1, 1)
- UECM deterministic case: unrestricted intercept, no trend (case 3)
- Inference: HAC with 3 lags
- ARDL AIC/BIC: -337.988 / -320.568

## Integration-order checks

| Variable | Transformation | ADF statistic | ADF p | KPSS statistic | KPSS p |
| --- | --- | ---: | ---: | ---: | ---: |
| `log_adr` | Level | -1.903 | 0.3306 | 0.319 | 0.1000 |
| `log_adr` | First difference | -10.331 | 2.85e-18 | 0.145 | 0.1000 |
| `log_adr` | Second difference | -8.039 | 1.87e-12 | 0.141 | 0.1000 |
| `log_kr_usd` | Level | -1.465 | 0.5508 | 0.492 | 0.0434 |
| `log_kr_usd` | First difference | -9.544 | 2.68e-16 | 0.107 | 0.1000 |
| `log_kr_usd` | Second difference | -7.490 | 4.53e-11 | 0.092 | 0.1000 |

ADF has a unit-root null; KPSS has a stationarity null. Both series are strongly stationary after first differencing, and neither shows evidence of being I(2). The level tests are mixed for `log_adr`.

## ARDL–UECM bounds test

| Calibration | F statistic | Lower p | Upper p | Decision at 5% |
| --- | ---: | ---: | ---: | --- |
| Asymptotic | 7.000 | 0.0011 | 0.0047 | Reject |
| Finite simulation (100,000) | 7.000 | 0.0119 | 0.0243 | Reject |

Finite-sample simulated critical values:

| Percentile | I(0) lower | I(1) upper |
| ---: | ---: | ---: |
| 90% | 4.128 | 4.897 |
| 95% | 5.054 | 5.933 |
| 99% | 7.269 | 8.244 |

## Long-run equilibrium and adjustment

The estimated equilibrium is:

`log_adr = 1.4010*** + 0.7750*** × log_kr_usd`

Equivalently in price levels:

`ADR price = exp(1.4010) × (FX-adjusted Korean price)^0.7750`

| Long-run equilibrium | Estimate | HAC SE | p-value | BH q-value | 95% CI | Significance |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Intercept, ĉ | 1.4010 | 0.4176 | 0.0008 | 0.0008 | [0.5824, 2.2195] | *** |
| Slope, θ̂ | 0.7750 | 0.0884 | 1.84e-18 | 3.68e-18 | [0.6018, 0.9483] | *** |
| Error correction, λ̂ | -0.3038 | 0.1196 | 0.0129 | 0.0194 | [-0.5417, -0.0660] | ** |

The slope θ̂=0.7750 is a long-run elasticity: a 1% increase in the FX-adjusted Korean price is associated with an estimated 0.775% increase in SKHY's long-run price. The intercept ĉ absorbs the persistent log-level wedge, including price-level and market-structure differences; it should not be read as a return.

The adjustment coefficient λ̂=-0.3038 means SKHY corrects approximately 30.4% of the previous disequilibrium per U.S. session. The implied half-life is 1.91 sessions; approximately 90% and 95% of an initial deviation decay after 6.36 and 8.27 sessions, respectively, if the estimated linear adjustment remains stable.

H1 is estimated at daily frequency, so it does **not** identify an 'after X minutes' convergence time. Minute-based convergence after dynamic or 09:30 anchors is an H2 estimand and must be obtained from the anchor × horizon conditional ECM.

A p-value measures evidence against a coefficient-specific null under the fitted model; it is not an effect size. The BH q-value adjusts the family of reported coefficient tests to control the expected false discovery rate. Economic magnitude is conveyed by θ̂ and λ̂, not by their p- or q-values. Significance: *** p<0.01, ** p<0.05, * p<0.10.

## UECM coefficients

| Term | Estimate | HAC SE | p-value | BH q-value | 95% CI | Sig. |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `const` | 0.4257 | 0.1524 | 0.0065 | 0.0130 | [0.1226, 0.7288] | *** |
| `log_adr.L1` | -0.3038 | 0.1196 | 0.0129 | 0.0194 | [-0.5417, -0.0660] | ** |
| `log_kr_usd.L1` | 0.2355 | 0.1086 | 0.0330 | 0.0396 | [0.0195, 0.4515] | ** |
| `D.log_kr_usd.L0` | 0.4655 | 0.1114 | 7.26e-05 | 0.0002 | [0.2438, 0.6871] | *** |
| `us_soxx_return` | 1.1937 | 0.1438 | 1.63e-12 | 9.79e-12 | [0.9077, 1.4797] | *** |
| `kr_sector_return` | -0.2321 | 0.1209 | 0.0583 | 0.0583 | [-0.4724, 0.0083] | * |

## Residual diagnostics

| Diagnostic | Statistic | p-value | Assessment |
| --- | ---: | ---: | --- |
| Ljung–Box Q(5) | 1.196 | 0.9453 | No evidence against no serial correlation |
| Ljung–Box Q(10) | 5.933 | 0.8209 | No evidence against no serial correlation |
| Breusch–Pagan | 11.906 | 0.0361 | Reject homoskedasticity |
| ARCH-LM(5) | 7.816 | 0.1667 | No evidence against no ARCH effects |
| Jarque–Bera | 67.190 | 2.57e-15 | Reject normality |
| CUSUM | 1.162 | 0.1343 | No evidence against parameter stability |
| Durbin–Watson | 1.681 | — | Descriptive |
| Equation RMSE | 0.0335 | — | Descriptive |

HAC inference is retained because Breusch–Pagan rejects homoskedasticity. Jarque–Bera also rejects residual normality. Ljung–Box, ARCH-LM, and CUSUM do not reject their respective nulls at 5%; these nonsignificant results are not proof that the assumptions hold.

## Anchor-time statistics (descriptive; used by H2)

Anchor timing does not enter the daily H1 bounds test. It is reported here to connect the established daily equilibrium to the subsequent minute-horizon H2 analysis.

| Measure | Dynamic premarket anchor | Fixed comparison anchor |
| --- | ---: | ---: |
| Valid dates | 87 | 87 |
| Invalid dates | 3 | 3 |
| Mean time | 07:11 ET | 09:30 ET |
| Median time | 07:10 ET | 09:30 ET |
| Earliest / latest | 07:10 ET / 07:35 ET | 09:30 / 09:30 ET |
| Standard deviation | 3.05 minutes | 0.00 minutes |
| Time distribution | 07:10 ET: 84, 07:20 ET: 2, 07:35 ET: 1 | 09:30 ET on all valid dates |

<details>
<summary>Daily anchor times and selection status</summary>

| U.S. date | Dynamic time/status | Fixed time/status |
| --- | --- | --- |
| 2026-07-13 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-07-14 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-07-15 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-07-16 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-07-17 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-07-20 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-07-21 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-07-22 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-07-23 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-07-24 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-07-27 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-07-28 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-07-29 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-07-30 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-07-31 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-08-03 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-08-04 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-08-05 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-08-06 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-08-07 | 07:20 ET — selected | 09:30 ET — selected |
| 2026-08-10 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-08-11 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-08-12 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-08-13 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-08-14 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-08-17 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-08-18 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-08-19 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-08-20 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-08-21 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-08-24 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-08-25 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-08-26 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-08-27 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-08-28 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-08-31 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-09-01 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-09-02 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-09-03 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-09-04 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-09-08 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-09-09 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-09-10 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-09-11 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-09-14 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-09-15 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-09-16 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-09-17 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-09-18 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-09-21 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-09-22 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-09-23 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-09-24 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-09-25 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-09-28 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-09-29 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-09-30 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-10-01 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-10-02 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-10-05 | 07:35 ET — selected | 09:30 ET — selected |
| 2026-10-06 | 07:20 ET — selected | 09:30 ET — selected |
| 2026-10-07 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-10-08 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-10-09 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-10-12 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-10-13 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-10-14 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-10-15 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-10-16 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-10-19 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-10-20 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-10-21 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-10-22 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-10-23 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-10-26 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-10-27 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-10-28 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-10-29 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-10-30 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-11-02 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-11-03 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-11-04 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-11-05 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-11-06 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-11-09 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-11-10 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-11-11 | 07:10 ET — selected | 09:30 ET — selected |
| 2026-11-12 | quotes_missing | fixed_open_stale_or_invalid |
| 2026-11-13 | quotes_missing | fixed_open_stale_or_invalid |
| 2026-11-16 | quotes_missing | fixed_open_stale_or_invalid |

</details>

## Final cointegration assessment

Valid statistical cointegration evidence **was obtained under the pre-specified baseline model** because: (1) neither price series shows evidence of being I(2); (2) the finite-sample bounds statistic exceeds the 5% I(1) upper critical value and has upper-bound p=0.0243; and (3) the error-correction coefficient is negative and significant (λ̂=-0.3038, p=0.0129, BH q=0.0194).

**H1 verdict: supported at the 5% level, but not conclusive as a structural long-run claim.** The result is internally valid for this specification and sample; it remains provisional because the sample contains only 90 sessions, the level unit-root tests are mixed for `log_adr`, and residual heteroskedasticity/non-normality is present.

## Interpretation constraints

- The effective history is too short for a strong structural long-run claim; results should be updated as SKHY history accumulates.
- The bounds conclusion is sensitive to specification, deterministic case, lag selection, and small-sample calibration.
- Korean prices are FX-adjusted and aligned point-in-time; this does not eliminate market-hours, trading-friction, or ADS-ratio risks.
- This H1 result supports estimating conditional H2/H3 ECMs, but those models should use coefficients trained on an earlier or otherwise separated sample to avoid generated-regressor look-ahead.
