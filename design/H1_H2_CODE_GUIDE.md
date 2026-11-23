# H1/H2 code guide

## Purpose

The H1/H2 pipeline models SKHY's previous-close-to-open return using the latest
FX-adjusted Korean ordinary-share information. H1 estimates the long-run price
relation. H2 is a daily error-correction model (ECM) whose controls and lag
orders are selected inside a nested expanding-window procedure.

The implementation is designed to prevent selection leakage: every forecast
date re-estimates H1, reconstructs the ECT, selects lag orders, tests model
eligibility, and compares controls using earlier observations only.

## Code map

| File | Responsibility |
| --- | --- |
| `src/features/h1_h2.py` | Builds the daily target, Korean predictor, and candidate controls from raw Parquet data |
| `src/hypotheses/h1.py` | Estimates the daily ARDL--UECM, performs bounds inference, constructs the ECT, and evaluates rolling stability |
| `src/hypotheses/overnight_ablation.py` | Imports H1 and estimates H2, performs nested selection, proxy redundancy analysis, and writes H2 artifacts |
| `backtest/backtest.py` | Reproduces the nested expanding-window forecasts from a materialized run |
| `tests/test_overnight_ablation.py` | Tests ECM construction, lag handling, selection, proxy subsets, and no-lookahead behavior |
| `reports/H2_ABLATION_STUDY_RESULTS.md` | Concise current results, equations, top specifications, and performance |

## Feature construction

`python -m src.features.h1_h2` writes the same unified daily panel to:

- `data/features/daily/part-00000.parquet`
- `data/features/overnight_h2/part-00000.parquet`

The target is:

\[
r^{ON}_d=\log\left(
\frac{P^{SKHY}_{d,09:30\ open}}{P^{SKHY}_{d-1,close}}
\right).
\]

The FX-adjusted Korean price and primary return predictor are:

\[
P^{KR,FX}_d=\frac{0.1P^{000660,KRW}_d}{USDKRW_d},
\qquad
r^{KR,FX}_d=\Delta\log P^{KR,FX}_d.
\]

Korean data are joined after the 15:30 KST close. Completed U.S. daily controls
become available the following day. The final model table contains daily rows;
intraday timestamps are used only to construct the target and enforce timing.

### Candidate control groups

`CONTROL_GROUPS` in `src/features/h1_h2.py` defines the selectable information
sets:

- SOXX completed daily return
- Prior completed daily normalized SKHY sentiment
- Korean semiconductor ETF return
- KOSDAQ return
- KOSPI return
- Korean market proxies: FLKR and USD/KRW returns
- U.S. headline-inflation YoY and core-CPI monthly change
- U.S. funding-condition spreads and changes
- VIX completed daily return
- Two-year and ten-year Treasury yield changes

The raw Korean proxy dataset still contains EWY. A separate redundancy analysis
tests every EWY/FLKR/USD-KRW subset; EWY is not in the operational group because
it nearly duplicates FLKR.

## H1 implementation

The H1 design, UECM equation, long-run transformation, Narayan and bootstrap
bounds inference, and rolling-stability diagnostics are specified in
[`design/h1.md`](h1.md). `src/hypotheses/h1.py` is the only H1 estimator;
H2 imports its `fit_h1_equilibrium()` and `equilibrium_error()` functions.

## H2 implementation

For control vector \(Z_d\), lag orders \(p,q\), and coefficients \(\phi_i,\delta_j\),
H2 is:

\[
r^{ON}_d=\alpha+\lambda ECT_d+\beta r^{KR,FX}_d
+\sum_{i=1}^{p}\phi_i r^{ON}_{d-i}
+\sum_{j=1}^{q}\delta_j r^{KR,FX}_{d-j}
+\gamma'Z_d+\epsilon_d.
\]

`prepare_ecm_panel()` creates the ECT and candidate lags. `_design()` constructs
the regression matrix. `fit_daily()` uses OLS coefficients with HAC standard
errors.

## Nested model selection

`nested_walk_forward_selection()` is the operative forecasting procedure.
For every outer forecast origin it:

1. Uses dates strictly before the forecast date.
2. Re-estimates H1 and reconstructs training and test ECTs.
3. Searches \(p,q\in\{0,1,2,3\}\) by BIC within every control specification.
4. Retains a specification only when:

   \[
   -2<\lambda<0,\quad p_\lambda<0.05,
   \quad \beta_{KR,FX}>0,\quad p_\beta<0.05.
   \]

5. Ranks eligible models by average inner expanding-window OOS RMSE.
6. Computes the one-standard-error threshold from squared forecast losses.
7. Selects the simplest model inside the one-SE set.
8. Uses MAE, direction accuracy, coefficient stability, and BIC as ordered
   tie-breakers.

The audit table stores every candidate at every outer origin, including H1
parameters, selected orders, eligibility, inner metrics, the one-SE decision,
and the selected model.

## Korean proxy redundancy analysis

`analyze_kr_proxy_redundancy()` evaluates all seven nonempty subsets of EWY,
FLKR, and USD/KRW on one common ECM sample with \(p=q=0\). It reports:

- Pairwise return correlations
- Variance inflation factors
- BIC and adjusted \(R^2\)
- Walk-forward RMSE, MAE, and direction accuracy
- ECT \(\lambda\), Korean-return \(\beta\), and their HAC p-values

The current result retains FLKR plus USD/KRW. EWY and FLKR have near-unit
correlation and VIFs above 200, so including both destabilizes their individual
coefficients without improving forecast error.

## Full-sample reference estimates

`build_full_sample_comparison()` produces a descriptive common-sample table.
It is useful for coefficient inspection but does not select outer forecasts.
The nested selector is the only source of reported OOS forecasts.

The H1 and H2 equations in the Markdown report are full-common-sample reference
estimates for the current final-origin specification. Origin-specific forecast
coefficients differ because the nested pipeline re-estimates them using prior
data only.

## Outputs

The default run writes to
`data/features/experiments/run_id=overnight-daily-ablation/`:

| Artifact | Contents |
| --- | --- |
| `overnight.parquet` | Model panel with ECT and candidate lags |
| `h1_model.parquet` | Full-sample H1 reference estimate |
| `h2_model.parquet` | Final-origin H2 reference estimate and nested metrics |
| `control_ablation.parquet` | Descriptive common-sample control comparison |
| `nested_selection_audit.parquet` | Origin-by-origin nested selection audit |
| `overnight_backtest.parquet` | Selected outer forecasts and realized returns |
| `overnight_backtest_summary.parquet` | Backtest summary metrics |
| `proxy_redundancy_analysis.parquet` | Seven Korean proxy subset models |
| `proxy_redundancy_diagnostics.parquet` | Proxy correlations and VIFs |

The human-readable report is `reports/H2_ABLATION_STUDY_RESULTS.md`.

## Usage

Install dependencies and build the feature panel:

```powershell
python -m pip install -r requirements.txt
python -m src.features.h1_h2
```

Run H1, the nested H2 study, proxy analysis, artifacts, and report:

```powershell
python -m src.hypotheses.overnight_ablation `
  --run-id overnight-daily-ablation `
  --max-p 3 `
  --max-q 3 `
  --min-train 40
```

Reproduce the nested backtest from the materialized run:

```powershell
python -m backtest.backtest `
  --run-id overnight-daily-ablation `
  --h1-spec daily_ardl_uecm `
  --h2-spec ablation_selected `
  --min-train 40
```

Run automated checks:

```powershell
python -m pytest -q
```

## Extending the study

To add a daily control source:

1. Create its point-in-time transformation in `src/features/h1_h2.py`.
2. Add its model columns as one coherent entry in `CONTROL_GROUPS`.
3. Add required synthetic columns and behavior checks to
   `tests/test_overnight_ablation.py`.
4. Rebuild features and rerun the study.

Controls with several related columns should remain grouped unless a dedicated
within-group redundancy analysis justifies a smaller set. Do not use current-day
U.S. closing information to forecast that same day's SKHY opening return.

## Data assumptions and limitations

- The sample is short, so coefficient and model-selection stability must be
  monitored as new sessions arrive.
- Inflation is monthly. When release timestamps are absent, availability is
  conservatively assigned to the 15th of the following month.
- CPI YoY uses the supplied field when present and otherwise derives
  \(100(CPI_t/CPI_{t-12}-1)\).
- Core-CPI change is a monthly fractional change, not a percentage-point value.
- Korean market proxies are market-implied controls, not direct Korean macro
  measurements.
